r"""The proposed intervention-response detector.

Technical Specification: "Scores residuals of intervention-response signatures
after predicting normal signatures from legitimate context."

**Signatures.** For transformation :math:`m` and site :math:`l`, the response is
the change of the decision-token residual stream,

.. math::  \Delta_{m,l}(x) = a_l(T_m x) - a_l(x),

projected onto a 32-dimensional training-only basis :math:`V_{g,l}`. That basis is
fit by PCA on normal-fitting differences, separately per site and per group
:math:`g`. The signature of group :math:`g` concatenates
:math:`V_{g,l}^\top \Delta_{m,l}(x)` over its transformations and the sites. The
two groups are kept separate, as the specification requires:

* :math:`g = \text{TP}` (task-preserving): rename, reorder, serialization,
  paraphrase. A normal actor should barely move.
* :math:`g = \text{TC}` (task-changing): auth_change, requirement_change. A
  normal actor's decision representation should move *in a way predictable
  from the task*.

**Normal-signature model.** A ridge regression predicts each signature from
legitimate context :math:`c(x)`: the 96-d projected base activations plus the
observable surface context (:func:`actmon.data.features.context_features`). It
is fit on the normal-fitting split, with the ridge penalty chosen by efficient
leave-one-out CV (``RidgeCV``). Its intercept is the mean normal response.

**Score.** Residuals :math:`e_g(x) = s_g(x) - \hat f_g(c(x))` are scored by
shrinkage Mahalanobis distance. Their covariance is estimated from
*cross-fitted* residuals (5 folds), so the in-sample fit does not shrink it.
Each group's squared distance is standardised by its cross-fitted normal mean
and SD, and the final score is :math:`z_{TP} + z_{TC}`. Ablations
(``groups``, ``context``, ``include_base``) are run in A4.

**Access.** Activations at three sites for the base input and six transformed
inputs (six extra forward passes), no labels, no policy oracle: the
transformations are structural edits that never ask the oracle what changes
(:mod:`actmon.sim.transforms`).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from actmon.data.features import FeatureSet
from actmon.detectors.base import Access, Detector
from actmon.detectors.projection import ShrinkageGaussian, SitePCA
from actmon.sim.transforms import TASK_CHANGING, TASK_PRESERVING

GROUPS = {"tp": TASK_PRESERVING, "tc": TASK_CHANGING}


class ResponseDetector(Detector):
    name = "response"

    def __init__(self, dim: int = 32, resp_dim: int = 32, sig_dim: int | None = 16, groups=("tp", "tc"),
                 context: str = "base+surface", include_base: bool = False, n_folds: int = 5, alphas=None,
                 seed: int = 0, name: str | None = None):
        self.dim, self.resp_dim, self.sig_dim = dim, resp_dim, sig_dim
        self.groups = tuple(groups)
        self.context_mode = context
        self.include_base = include_base
        self.n_folds = n_folds
        self.alphas = np.logspace(-2, 5, 15) if alphas is None else np.asarray(alphas)
        self.seed = seed
        if name:
            self.name = name
        n_extra = sum(len(GROUPS[g]) for g in self.groups)
        self.access = Access("sites+variants", "none", n_extra, context="observable surface context"
                             if "surface" in context else "none",
                             notes=f"groups={'+'.join(self.groups)}; context={context}; structural edits, no oracle")

    # ---------------------------------------------------------------- features
    def _context(self, fs: FeatureSet) -> np.ndarray:
        parts = []
        if "base" in self.context_mode:
            parts.append(self.base_proj.concat({s: fs.base[s] for s in fs.sites}))
        if "surface" in self.context_mode:
            parts.append(fs.context.astype(np.float64))
        if not parts:
            return np.zeros((fs.n, 0))
        C = np.concatenate(parts, axis=1)
        return (C - self.c_mu) / self.c_sd if hasattr(self, "c_mu") else C

    def _deltas(self, fs: FeatureSet, g: str) -> dict[str, dict[int, np.ndarray]]:
        return {m: {s: fs.variants[m][s].astype(np.float64) - fs.base[s] for s in fs.sites}
                for m in GROUPS[g] if m in fs.variants}

    def _raw_signature(self, fs: FeatureSet, g: str) -> np.ndarray:
        D = self._deltas(fs, g)
        cols = [D[m][s] @ self.resp_proj[g].W[s] for m in sorted(D) for s in fs.sites]
        return np.concatenate(cols, axis=1)

    def _signature(self, fs: FeatureSet, g: str) -> np.ndarray:
        S = self._raw_signature(fs, g)
        if g in self.sig_pca:
            mu, W = self.sig_pca[g]
            S = (S - mu) @ W
        return S

    # ---------------------------------------------------------------- fit / score
    def fit(self, fit: FeatureSet, dev=None, dev_labels=None):
        self.base_proj = SitePCA(self.dim).fit({s: fit.base[s] for s in fit.sites})
        C = self._context(fit)
        self.c_mu = C.mean(0) if C.shape[1] else np.zeros(0)
        self.c_sd = C.std(0) + 1e-8 if C.shape[1] else np.zeros(0)
        C = (C - self.c_mu) / self.c_sd if C.shape[1] else C
        self.resp_proj, self.sig_pca, self.models, self.gauss, self.zstats = {}, {}, {}, {}, {}
        for g in self.groups:
            D = self._deltas(fit, g)
            if not D:
                raise ValueError(f"feature set has no variants for group {g}")
            stacked = {s: np.concatenate([D[m][s] for m in D]) for s in fit.sites}
            self.resp_proj[g] = SitePCA(self.resp_dim).fit(stacked)
            if self.sig_dim:
                S0 = self._raw_signature(fit, g)
                mu = S0.mean(0)
                _, _, Vt = np.linalg.svd(S0 - mu, full_matrices=False)
                self.sig_pca[g] = (mu, Vt[: min(self.sig_dim, Vt.shape[0])].T)
            S = self._signature(fit, g)
            R = self._cross_fit_residuals(C, S)
            self.models[g] = self._fit_model(C, S)
            self.gauss[g] = ShrinkageGaussian().fit(R)
            self.zstats[g] = self._oos_stats(R)
        if self.include_base:
            Zb = self.base_proj.concat({s: fit.base[s] for s in fit.sites})
            self.base_gauss = ShrinkageGaussian().fit(Zb)
            self.zstats["base"] = self._oos_stats(Zb)
        return self

    def _oos_stats(self, R: np.ndarray) -> tuple[float, float]:
        """Mean and SD of *out-of-fold* squared distances, so in-sample covariance fit does not bias them."""
        d2 = np.zeros(len(R))
        for tr, te in KFold(self.n_folds, shuffle=True, random_state=self.seed + 1).split(R):
            d2[te] = ShrinkageGaussian().fit(R[tr]).score(R[te])
        return float(d2.mean()), float(d2.std() + 1e-8)

    def _fit_model(self, C, S):
        if C.shape[1] == 0:
            return ("mean", S.mean(0))
        m = RidgeCV(alphas=self.alphas).fit(C, S)
        return ("ridge", m)

    def _predict(self, model, C):
        kind, m = model
        return np.broadcast_to(m, (C.shape[0], m.shape[0])) if kind == "mean" else m.predict(C)

    def _cross_fit_residuals(self, C, S):
        R = np.zeros_like(S)
        kf = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        alpha = None
        if C.shape[1]:
            alpha = float(RidgeCV(alphas=self.alphas).fit(C, S).alpha_)
        for tr, te in kf.split(S):
            if C.shape[1] == 0:
                R[te] = S[te] - S[tr].mean(0)
            else:
                R[te] = S[te] - Ridge(alpha=alpha).fit(C[tr], S[tr]).predict(C[te])
        return R

    def group_scores(self, fs: FeatureSet) -> dict[str, np.ndarray]:
        C = self._context(fs)
        out = {}
        for g in self.groups:
            R = self._signature(fs, g) - self._predict(self.models[g], C)
            mu, sd = self.zstats[g]
            out[g] = (self.gauss[g].score(R) - mu) / sd
        if self.include_base:
            Zb = self.base_proj.concat({s: fs.base[s] for s in fs.sites})
            mu, sd = self.zstats["base"]
            out["base"] = (self.base_gauss.score(Zb) - mu) / sd
        return out

    def score(self, fs: FeatureSet) -> np.ndarray:
        return np.sum(np.stack(list(self.group_scores(fs).values())), axis=0)
