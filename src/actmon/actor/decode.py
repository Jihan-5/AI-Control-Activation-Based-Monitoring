r"""Constrained action decoding with activation capture.

The actor's proposal is the **most probable complete candidate action**:

.. math::  \hat a = \arg\max_{a \in \mathcal C(s)} \log p_\theta(\text{tokens}(a) \,\|\, \text{eos} \mid \text{prompt})

where :math:`\mathcal C(s)` is the simulator's finite candidate set
(:func:`actmon.sim.actions.candidate_actions`). Exact sequence scores are
computed in one pass: the prompt is run once with a KV cache, the cache is
repeated once per candidate, and all continuations are scored together. Free
generation is not used. An unparseable answer would be a third outcome besides
"valid" and "violation", and its rate would depend on decoding settings
(decision A-D3).

The same pass returns:

* residual activations at the configured sites (prompt forward only)
* *response/action scores* for the behaviour-only monitor: chosen-action
  log-probability, total candidate mass, top-2 margin, entropy over candidates,
  and the candidate distribution aggregated by verb

Activations do not depend on the decoding rule, because they are read before
any action token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from actmon.actor.hooks import ResidualRecorder, left_pad
from actmon.sim.actions import VERBS, Action

FEATURE_NAMES = ("logp_chosen", "cand_mass", "margin", "entropy") + tuple(f"p_{v}" for v in VERBS)


@dataclass
class ActorOutputs:
    actions: list[Action] = field(default_factory=list)
    feats: list[np.ndarray] = field(default_factory=list)  # FEATURE_NAMES order
    acts: dict[int, list[np.ndarray]] = field(default_factory=dict)  # site -> per-root vectors

    def extend(self, other: "ActorOutputs") -> None:
        self.actions.extend(other.actions)
        self.feats.extend(other.feats)
        for s, v in other.acts.items():
            self.acts.setdefault(s, []).extend(v)

    def act_matrix(self, site: int) -> np.ndarray:
        return np.stack(self.acts[site]).astype(np.float32)

    def feat_matrix(self) -> np.ndarray:
        return np.stack(self.feats).astype(np.float32)


@torch.no_grad()
def _batch(model, pad_id: int, prompts: list[list[int]], cands: list[list[tuple[Action, list[int]]]],
           sites: list[int]) -> ActorOutputs:
    dev = next(model.parameters()).device
    ids, mask, pos = left_pad(prompts, pad_id, device=dev)
    rec = ResidualRecorder(model, sites)
    with rec:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    acts = rec.pop() if sites else {}
    first_lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    cache = out.past_key_values
    B = len(prompts)
    K = max(len(c) for c in cands)
    T = max(len(t) for c in cands for _, t in c)
    cont = torch.full((B, K, T), pad_id, dtype=torch.long, device=dev)
    cmask = torch.zeros((B, K, T), dtype=torch.long, device=dev)
    valid = torch.zeros((B, K), dtype=torch.bool, device=dev)
    for b, cl in enumerate(cands):
        for k, (_, toks) in enumerate(cl):
            cont[b, k, :len(toks)] = torch.tensor(toks, device=dev)
            cmask[b, k, :len(toks)] = 1
            valid[b, k] = True
    logp = first_lp.gather(1, cont[:, :, 0])
    if T > 1:
        cache.batch_repeat_interleave(K)
        inp = cont[:, :, :-1].reshape(B * K, T - 1)
        full_mask = torch.cat([mask.repeat_interleave(K, 0), cmask[:, :, :-1].reshape(B * K, T - 1)], dim=1)
        plen = mask.sum(-1).repeat_interleave(K)
        cpos = plen[:, None] + torch.arange(T - 1, device=dev)[None, :]
        o2 = model(input_ids=inp, attention_mask=full_mask, position_ids=cpos, past_key_values=cache, use_cache=False)
        lp2 = torch.log_softmax(o2.logits.float(), dim=-1)
        nxt = cont[:, :, 1:].reshape(B * K, T - 1)
        g = lp2.gather(-1, nxt[..., None]).squeeze(-1) * cmask[:, :, 1:].reshape(B * K, T - 1)
        logp = logp + g.sum(-1).reshape(B, K)
    logp = logp.masked_fill(~valid, float("-inf")).cpu().numpy()

    res = ActorOutputs()
    for b, cl in enumerate(cands):
        lp = logp[b, :len(cl)]
        order = np.argsort(-lp)
        mass = float(np.logaddexp.reduce(lp))
        prob = np.exp(lp - mass)
        ent = float(-(prob * np.log(np.clip(prob, 1e-30, None))).sum())
        margin = float(lp[order[0]] - lp[order[1]]) if len(lp) > 1 else 0.0
        pv = np.zeros(len(VERBS))
        for (a, _), p in zip(cl, prob):
            pv[VERBS.index(a.verb)] += p
        res.actions.append(cl[int(order[0])][0])
        res.feats.append(np.concatenate([[lp[order[0]], mass, margin, ent], pv]))
    for s in sites:
        res.acts[s] = list(acts[s].numpy())
    return res


def run_actor(model, pad_id: int, prompts: list[list[int]], cands: list[list[tuple[Action, list[int]]]],
              sites: list[int], batch_size: int = 64, progress: bool = False) -> ActorOutputs:
    """Decode and record activations for many roots, batching by length to limit padding."""
    order = np.argsort([len(p) for p in prompts], kind="stable")
    out = ActorOutputs()
    chunks = []
    it = range(0, len(order), batch_size)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="actor", leave=False)
    for i in it:
        idx = order[i:i + batch_size]
        chunks.append((idx, _batch(model, pad_id, [prompts[j] for j in idx], [cands[j] for j in idx], sites)))
    # restore the input order
    n = len(prompts)
    actions: list = [None] * n
    feats: list = [None] * n
    acts = {s: [None] * n for s in sites}
    for idx, r in chunks:
        for k, j in enumerate(idx):
            actions[j] = r.actions[k]
            feats[j] = r.feats[k]
            for s in sites:
                acts[s][j] = r.acts[s][k]
    out.actions, out.feats, out.acts = actions, feats, acts
    return out
