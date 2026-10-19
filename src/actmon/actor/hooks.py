r"""Residual-stream recording at the decision token.

Tensor: the residual stream *after* zero-based decoder block ``i``, i.e. the
output of ``model.model.layers[i]``, at the final non-padding prompt token (the
last token of the assistant header). Prompts are **left-padded**, so that token
is always at position ``-1``.

Hugging Face's ``output_hidden_states`` returns the embeddings followed by each
block's output, so ``hidden_states[i + 1]`` must equal the hook for every
``i < L - 1``. The last entry has the final norm applied, so the final block is
not a valid check. :func:`verify_sites` asserts this equality on a real batch.
It is the syllabus T1 "reproduce a manual forward pass" check, adapted to the
residual stream.

Left padding needs explicit position ids (``cumsum(mask) - 1``). Otherwise RoPE
positions shift with the padding and padded rows differ from unpadded ones.
:func:`left_pad` builds both, and ``tests/test_actor.py`` checks that a
padded row reproduces its unpadded activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from actmon.actor.models import blocks


def left_pad(seqs: list[list[int]], pad_id: int, device=None):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, T - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    if device is not None:
        ids, mask, pos = ids.to(device), mask.to(device), pos.to(device)
    return ids, mask, pos


@dataclass
class ResidualRecorder:
    """Context manager recording block outputs at the last position (or all positions).

    ``detach=False`` keeps the autograd graph, which the obfuscation attack needs.
    """

    model: object
    sites: list[int]
    last_only: bool = True
    detach: bool = True
    store: dict = field(default_factory=dict)
    active: bool = True
    _handles: list = field(default_factory=list)

    def _hook(self, site: int):
        def fn(module, inputs, output):
            if not self.active:
                return
            h = output[0] if isinstance(output, tuple) else output
            x = h[:, -1, :] if self.last_only else h
            if self.detach:
                x = x.detach().float().cpu()
            self.store.setdefault(site, []).append(x)
        return fn

    def __enter__(self):
        self.sites = sorted(set(self.sites))
        for s in self.sites:
            self._handles.append(blocks(self.model)[s].register_forward_hook(self._hook(s)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def pop(self) -> dict[int, torch.Tensor]:
        out = {s: torch.cat(v, dim=0) for s, v in self.store.items()}
        self.store.clear()
        return out


@torch.no_grad()
def verify_sites(model, seqs: list[list[int]], sites: list[int], pad_id: int) -> float:
    """Max |hook - hidden_states[i+1]| at the last position over ``sites`` (excluding the final block)."""
    L = int(model.config.num_hidden_layers)
    check = [s for s in sites if s < L - 1]
    ids, mask, pos = left_pad(seqs, pad_id, device=next(model.parameters()).device)
    rec = ResidualRecorder(model, check)
    with rec:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
    got = rec.pop()
    worst = 0.0
    for s in check:
        ref = out.hidden_states[s + 1][:, -1, :].float().cpu()
        worst = max(worst, float((got[s] - ref).abs().max()))
    return worst
