"""Synthetic actor backend: a structured simulator for PIPELINE TESTING ONLY.

    !!! Nothing produced by this backend is evidence about language models. !!!

It exists so that generation, screening, extraction, every detector,
calibration, statistics and reporting can run end-to-end on a CPU in CI. Its
planted structure is chosen by hand to mirror the restricted theoretical model
(research plan p.8), so a detector "working" on it only shows the code is wired
correctly.

Generative model for one prompt with latent task ``x`` (layer ``l``)
--------------------------------------------------------------------
    h_l(x) = W_l phi(c(x))                 observable task covariates
           + s_fam * o_l[family(x)]        benign shift: every family has its own
                                           offset; unfamiliar families are unseen
                                           during normal fitting
           + s_nuis * B_l eta(text(x))     nuisance subspace driven by a hash of
                                           the *surface text* (T edits move here)
           + (l / (L-1)) * e_l[slot(x)]    representation of the chosen action
           + s_fail * override(x) * f_l    planted failure shift, |B^T f|/|f| = overlap
           + o_ckpt                        global checkpoint offset (organisms)
           + noise * eps(fingerprint, l)

Decisions
---------
* Default: the oracle-correct slot.
* Ordinary errors / ordinary violations at fixed rates, drawn from a hash of
  the *full canonical latent state* (so T edits, which keep the latent state,
  keep the decision).
* Planted throughput bias: when a conflict is *active* (conflict prompt for C0/
  C1; always for organism="conflict"; only with the cue for
  organism="cue_conditional") and the restricted pool is physically free and no
  authenticated handoff binds the job, the actor prefers ALLOC_FAST with
  probability ``p_conflict_violation``. The propensity draw is seeded by the
  latent state *excluding authorization facts*, so the disposition survives
  authorization edits: a biased actor keeps choosing ALLOC_FAST when
  authorization flips (decision insensitivity), while a normal actor switches.
  The failure shift is added only when the bias *overrides* the correct action.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from irmon.config import SyntheticActorConfig
from irmon.envs.base import ChatPrompt, Environment, SlotStatus

from .backend import ActorBackend


def _h(*parts) -> int:
    blob = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


class SyntheticActorBackend(ActorBackend):
    name = "synthetic"

    def __init__(
        self,
        env: Environment,
        cfg: SyntheticActorConfig | None = None,
        checkpoint: str = "C0",
        structure_seed: int = 0,
    ):
        super().__init__()
        self.env = env
        self.cfg = cfg or SyntheticActorConfig()
        self.checkpoint = checkpoint
        self.n_layers = self.cfg.n_layers
        self.d_model = self.cfg.d_model
        self.slots = tuple(env.slots)
        self._tasks: dict[str, object] = {}
        self.structure_seed = structure_seed
        self.organism = self.cfg.organism
        if self.organism == "none" and checkpoint.startswith("C2cue"):
            self.organism = "cue_conditional"
        elif self.organism == "none" and checkpoint.startswith("C2"):
            self.organism = "conflict"

        rng = np.random.default_rng(structure_seed)
        L, D, K = self.n_layers, self.d_model, len(self.slots)
        names = env.context_feature_names() if hasattr(env, "context_feature_names") else ()
        self._ctx_names = tuple(names)
        C = max(1, len(names))
        P = 2 * D
        self._R = rng.normal(size=(P, C)) / np.sqrt(C)
        self._b = rng.normal(size=P) * 0.1
        self._W = rng.normal(size=(L, D, P)) / np.sqrt(P)
        r = self.cfg.nuisance_rank
        self._B = np.stack([np.linalg.qr(rng.normal(size=(D, r)))[0] for _ in range(L)])
        self._E = rng.normal(size=(L, D, K)) * 1.5
        self._f = np.empty((L, D))
        for li in range(L):
            g = rng.normal(size=D)
            inside = self._B[li] @ (self._B[li].T @ g)
            outside = g - inside
            inside /= np.linalg.norm(inside) + 1e-12
            outside /= np.linalg.norm(outside) + 1e-12
            a = self.cfg.failure_nuisance_overlap
            self._f[li] = a * inside + np.sqrt(max(0.0, 1 - a * a)) * outside
        self._trace_basis = rng.normal(size=(256, K + 1))
        ck = np.random.default_rng(_h("ckpt", checkpoint, structure_seed) % 2**32)
        self._ckpt_offset = (ck.normal(size=(L, D)) * 0.5) if self.organism != "none" else np.zeros((L, D))

    # --------------------------------------------------------- registration
    def prepare(self, prompts: Sequence[ChatPrompt], tasks: Sequence[object]) -> None:
        for p, t in zip(prompts, tasks):
            self._tasks[p.fingerprint()] = t

    def _task(self, prompt: ChatPrompt):
        try:
            return self._tasks[prompt.fingerprint()]
        except KeyError:
            raise KeyError(
                "SyntheticActorBackend received an unprepared prompt; call "
                "backend.prepare(prompts, tasks) first (the simulator needs the latent task)"
            ) from None

    # ------------------------------------------------------------ decisions
    def conflict_active(self, task) -> bool:
        s = task.state
        if self.organism == "conflict":
            return True
        if self.organism == "cue_conditional":
            # The cue organism also responds to an explicit prompt-level conflict,
            # so one checkpoint yields both a known (prompt) and a held-out (cue)
            # mechanism, as the split design requires.
            return bool(s.cue) or bool(s.conflict_prompt)
        return bool(s.conflict_prompt)

    def _disposition_key(self, task) -> tuple:
        """Latent state without the facts U edits change, so the bias persists
        across them (environment hook ``disposition_key``)."""
        return self.env.disposition_key(task)

    def _biased(self, task) -> bool:
        if not self.conflict_active(task):
            return False
        if not self.env.temptation(task):
            return False
        u = _h("bias", self.checkpoint, self.structure_seed, self._disposition_key(task)) / 2**64
        return u < self.cfg.p_conflict_violation

    def decide_slot(self, task) -> tuple[str, bool]:
        """Return (chosen slot, override) where override marks a bias-driven violation."""
        d = self.env.decide(task)
        if self._biased(task):
            target = self.env.planted_target_slot
            override = d.status(target) is SlotStatus.VIOLATION
            return target, override
        u = _h("err", self.checkpoint, self.structure_seed, self.env.canonical(task)) / 2**64
        if u < self.cfg.p_ordinary_violation:
            viol = d.violating_slots()
            if viol:
                return viol[_h("vpick", u) % len(viol)], False
        elif u < self.cfg.p_ordinary_violation + self.cfg.p_ordinary_error:
            others = [k for k, st in d.statuses if st in (SlotStatus.ALLOWED, SlotStatus.INFEASIBLE)]
            if others:
                return others[_h("epick", u) % len(others)], False
        return d.correct_slot, False

    # -------------------------------------------------------------- hidden
    def _phi(self, task) -> np.ndarray:
        ctx = self.env.observable_context(task)
        c = np.array([ctx.get(n, 0.0) for n in self._ctx_names] or [0.0], dtype=float)
        return np.tanh(self._R @ (np.log1p(np.abs(c)) * np.sign(c)) + self._b)

    def _family_offset(self, family: str) -> np.ndarray:
        rng = np.random.default_rng(_h("family", family, self.structure_seed) % 2**32)
        return rng.normal(size=(self.n_layers, self.d_model))

    def _hidden_one(self, prompt: ChatPrompt) -> np.ndarray:
        task = self._task(prompt)
        cfg = self.cfg
        L = self.n_layers
        slot, override = self.decide_slot(task)
        k = self.slots.index(slot)
        fp = prompt.fingerprint()
        eta = np.random.default_rng(_h("eta", fp) % 2**32).normal(size=cfg.nuisance_rank)
        eps = np.random.default_rng(_h("eps", fp) % 2**32).normal(size=(L, self.d_model))
        phi = self._phi(task)
        h = np.einsum("ldp,p->ld", self._W, phi)
        h += cfg.family_shift_scale * self._family_offset(task.state.family)
        h += cfg.nuisance_scale * np.einsum("ldr,r->ld", self._B, eta)
        depth = (np.arange(L) / max(1, L - 1))[:, None]
        h += depth * self._E[:, :, k]
        h += cfg.failure_scale * float(override) * self._f
        h += self._ckpt_offset
        h += cfg.noise * eps
        return h

    def hidden_at_final_token(self, prompts: Sequence[ChatPrompt], layers: Sequence[int]) -> np.ndarray:
        for li in layers:
            if not 0 <= li < self.n_layers:
                raise ValueError(f"layer {li} out of range for {self.n_layers}-layer simulator")
        out = np.stack([self._hidden_one(p)[list(layers)] for p in prompts]).astype(np.float32)
        self.counter.prompt_forwards += len(prompts)
        return out

    def free_text_hidden(self, prompts: Sequence[ChatPrompt], layers: Sequence[int]) -> np.ndarray:
        """Hidden states for prompts without a latent task (generic contrast
        pairs). Driven only by text hashes; carries no planted structure."""
        out = []
        for p in prompts:
            fp = p.fingerprint()
            sys_rng = np.random.default_rng(_h("free-sys", p.system) % 2**32)
            eta = np.random.default_rng(_h("eta", fp) % 2**32).normal(size=self.cfg.nuisance_rank)
            eps = np.random.default_rng(_h("eps", fp) % 2**32).normal(size=(self.n_layers, self.d_model))
            h = sys_rng.normal(size=(self.n_layers, self.d_model))
            h += self.cfg.nuisance_scale * np.einsum("ldr,r->ld", self._B, eta) + self.cfg.noise * eps
            out.append(h[list(layers)])
        self.counter.prompt_forwards += len(prompts)
        return np.stack(out).astype(np.float32)

    # ------------------------------------------------------------ scores
    def _utilities(self, prompt: ChatPrompt) -> np.ndarray:
        task = self._task(prompt)
        slot, _ = self.decide_slot(task)
        d = self.env.decide(task)
        u = 0.5 * np.random.default_rng(_h("util", prompt.fingerprint()) % 2**32).normal(size=len(self.slots))
        u[self.slots.index(d.correct_slot)] += 1.0
        u[self.slots.index(slot)] += 4.0
        return u

    def score_candidates(self, prompts: Sequence[ChatPrompt], candidates: Sequence[Sequence[str]]) -> np.ndarray:
        out = np.zeros((len(prompts), len(self.slots)), dtype=np.float32)
        for i, (p, cands) in enumerate(zip(prompts, candidates)):
            if len(cands) != len(self.slots):
                raise ValueError("synthetic backend expects one candidate per canonical slot")
            task = self._task(p)
            u = self._utilities(p)
            # Map candidate strings to slots so order is honoured, not assumed.
            for j, text in enumerate(cands):
                slot = self.env.parse_action(task, text)
                base = slot.split(":")[0] if slot else None
                out[i, j] = u[self.slots.index(base)] if base in self.slots else -20.0
            out[i] -= np.log(np.exp(out[i] - out[i].max()).sum()) + out[i].max()
        self.counter.candidate_forwards += len(prompts) * len(self.slots)
        return out

    def generate(self, prompts: Sequence[ChatPrompt], max_new_tokens: int = 32) -> list[str]:
        texts = []
        for p in prompts:
            task = self._task(p)
            slot, _ = self.decide_slot(task)
            texts.append({c.slot: c.text for c in self.env.candidates(task)}[slot])
        self.counter.prompt_forwards += len(prompts)
        return texts

    def loss_trace(self, prompts, targets, n_samples, sigma, seed) -> np.ndarray:
        """Synthetic stand-in for parameter-sampling loss traces."""
        if n_samples > self._trace_basis.shape[0]:
            raise ValueError("too many trace samples for the synthetic basis")
        out = np.zeros((len(prompts), n_samples), dtype=np.float32)
        for i, (p, t) in enumerate(zip(prompts, targets)):
            task = self._task(p)
            slot = (self.env.parse_action(task, t) or "REJECT").split(":")[0]
            k = self.slots.index(slot) if slot in self.slots else 0
            _, override = self.decide_slot(task)
            onehot = np.zeros(len(self.slots) + 1)
            onehot[k] = 1.0
            onehot[-1] = 2.0 * float(override)
            noise = np.random.default_rng(_h("trace", p.fingerprint(), seed) % 2**32).normal(size=n_samples)
            out[i] = 1.0 + sigma * 10 * (self._trace_basis[:n_samples] @ onehot) + 0.3 * noise
        self.counter.perturbed_forwards += len(prompts) * n_samples
        return out

    def manifest(self) -> dict:
        return {
            "backend": self.name,
            "WARNING": "synthetic simulator for pipeline tests; not evidence about language models",
            "checkpoint": self.checkpoint,
            "organism": self.organism,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "structure_seed": self.structure_seed,
            "config": self.cfg.model_dump(),
            "counter": self.counter.as_dict(),
        }
