r"""Planted-activation generator for the detector gate (A0).

It builds feature sets with the same structure as the real ones (three sites,
six transformed variants, surface context, context classes, families) from a
known linear-Gaussian model. Each detector can then be checked against a
mechanism whose answer is known, before any actor data are read. This is the
monitoring analogue of a theory gate: detectors are checked before any real data are read.

Normal root with context class :math:`k`, family :math:`f` and task factors
:math:`u \sim N(0, I_q)`:

.. math::
    a_l &= c\, M_{l,k} + A_l u + F_{l,f} + \sigma \varepsilon \\
    a_l(T_m x) &= a_l + \tau \varepsilon' \quad (m \text{ task-preserving}) \\
    a_l(T_m x) &= a_l + B_{m,l} \tanh(u) + \tau \varepsilon' \quad (m \text{ task-changing})

Violating roots, by regime:

==================  ===========================================================
``null``            identical distribution: every detector should flag ~alpha
``base_shift``      :math:`a_l \mathrel{+}= \delta\sigma\, v_l` with
                    ``base_shift_sd`` = :math:`\delta`: a base anomaly
``context_shift``   the base sits ``context_fraction`` of the way from its
                    observed class mean to the next class's mean. It looks normal
                    to a pooled model and abnormal given the context
``response_only``   base unchanged. The task-changing response is scaled by
                    ``1 - attenuation``, an actor that ignores the changed
                    authorization
==================  ===========================================================

Effect sizes are set so that a *correctly implemented* detector separates
clearly. The gate tests implementation, not power at realistic effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actmon.actor.decode import FEATURE_NAMES
from actmon.data.features import CONTEXT_NAMES, FeatureSet
from actmon.sim.transforms import MAIN_SUITE, TASK_CHANGING

REGIMES = ("null", "base_shift", "context_shift", "response_only")


@dataclass
class SyntheticConfig:
    d: int = 48
    sites: tuple[int, ...] = (0, 1, 2)
    n_classes: int = 6
    n_families: int = 8
    q: int = 6
    class_scale: float = 4.0
    family_scale: float = 0.3
    noise: float = 1.0
    tp_noise: float = 0.3
    tc_gain: float = 1.5
    base_shift_sd: float = 6.0       # base_shift: shift length in noise SDs, per site
    context_fraction: float = 0.8    # context_shift: fraction of the way to the next class mean
    attenuation: float = 1.0         # response_only: 1 = the task-changing response disappears


class SyntheticWorld:
    """Fixed model parameters (one draw per world). Splits sample roots from it."""

    def __init__(self, cfg: SyntheticConfig, rng: np.random.Generator):
        self.cfg = cfg
        d, q = cfg.d, cfg.q
        self.M = {s: rng.normal(size=(cfg.n_classes, d)) for s in cfg.sites}
        self.A = {s: rng.normal(size=(d, q)) / np.sqrt(q) for s in cfg.sites}
        self.F = {s: rng.normal(size=(cfg.n_families, d)) for s in cfg.sites}
        self.B = {m: {s: rng.normal(size=(d, q)) for s in cfg.sites} for m in TASK_CHANGING}
        self.v = {s: (lambda x: x / np.linalg.norm(x))(rng.normal(size=d)) for s in cfg.sites}

    def sample(self, rng: np.random.Generator, n_normal: int, n_viol: int, regime: str, split: str):
        c = self.cfg
        n = n_normal + n_viol
        y = np.r_[np.zeros(n_normal, int), np.ones(n_viol, int)]
        k = rng.integers(c.n_classes, size=n)
        fam = rng.integers(c.n_families, size=n)
        u = rng.normal(size=(n, c.q))
        base, variants = {}, {m: {} for m in MAIN_SUITE}
        for s in c.sites:
            centre = c.class_scale * self.M[s][k]
            if regime == "context_shift":
                nxt = c.class_scale * self.M[s][(k + 1) % c.n_classes]
                centre = np.where(y[:, None] == 1, (1 - c.context_fraction) * centre + c.context_fraction * nxt, centre)
            a = centre + u @ self.A[s].T + c.family_scale * self.F[s][fam] + c.noise * rng.normal(size=(n, c.d))
            if regime == "base_shift":
                a = a + (y[:, None] * c.base_shift_sd * c.noise) * self.v[s]
            base[s] = a.astype(np.float32)
            for m in MAIN_SUITE:
                eps = c.tp_noise * rng.normal(size=(n, c.d))
                if m in TASK_CHANGING:
                    gain = np.where(y == 1, 1 - c.attenuation, 1.0) if regime == "response_only" else np.ones(n)
                    resp = c.tc_gain * gain[:, None] * (np.tanh(u) @ self.B[m][s].T)
                    variants[m][s] = (a + resp + eps).astype(np.float32)
                else:
                    variants[m][s] = (a + eps).astype(np.float32)
        ctx = np.zeros((n, len(CONTEXT_NAMES)), np.float32)
        ctx[np.arange(n), k % 4] = 1.0
        ctx[:, 4] = (k // 4) % 2
        ctx[:, 5] = 2 + k % 3
        P = rng.dirichlet(np.ones(4), size=n)
        feats = np.concatenate([rng.normal(size=(n, 4)), P], axis=1).astype(np.float32)
        verbs = np.array(["allocate p1", "review", "reject", "handoff"])
        act = verbs[P.argmax(1)]
        fs = FeatureSet(
            split=split, root_ids=np.array([f"{split}-{i}" for i in range(n)]),
            family=np.array([f"f{x:02d}" for x in fam]), sites=list(c.sites), base=base, base_feats=feats,
            base_action=act, context=ctx, context_class=np.array([f"c{x}" for x in k]), variants=variants,
            variant_feats={m: feats for m in MAIN_SUITE}, variant_action={m: act for m in MAIN_SUITE})
        assert feats.shape[1] == len(FEATURE_NAMES)
        return fs, y
