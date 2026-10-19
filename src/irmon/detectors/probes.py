"""Linear probes (syllabus S10: Goldowsky-Dill et al., ICML 2025; Anthropic 2024).

SupervisedProbe     logistic regression on the same layers, trained on
                    *labelled* development data. An extra-information
                    reference, not an equal-data baseline: it sees violation
                    labels that the unsupervised monitor never sees, and that
                    supervision is disclosed in the access table.
ContrastPairProbe   a generic contrast direction (mean(pos) - mean(neg)
                    activations of contrast prompts, e.g. "follow the policy"
                    vs "ignore the policy" on generic tasks) that uses no
                    knowledge of the planted violation. Score = projection of
                    the standardized h_base on the per-layer direction,
                    averaged over layers.

Thresholds for both are set downstream on benign calibration data, at the same
nominal FPR as every other detector.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from irmon.features.bundle import ResponseBundle
from irmon.features.projection import LayerProjection

from .base import AccessSpec, Detector
from .linalg import Standardizer


def _require_labels(name: str, labels) -> np.ndarray:
    if labels is None:
        raise ValueError(f"{name} is supervised: fit() requires violation labels (dev split)")
    y = np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name} needs both normal and violating examples to fit")
    return y


def tune_logistic(
    X: np.ndarray,
    y: np.ndarray,
    c_grid: tuple[float, ...],
    val_idx: np.ndarray | None,
    val_frac: float,
    seed: int,
) -> tuple[LogisticRegression, float, dict]:
    """Fit on a train portion, pick C by validation AUROC on a disjoint portion,
    then refit on all labelled rows with the chosen C."""
    n = len(y)
    if val_idx is None:
        rng = np.random.default_rng(seed)
        val_idx = np.zeros(n, bool)
        for cls in (0, 1):  # stratified internal split
            pos = np.flatnonzero(y == cls)
            k = max(1, int(round(val_frac * len(pos))))
            val_idx[rng.choice(pos, size=k, replace=False)] = True
    val_idx = np.asarray(val_idx)
    if val_idx.dtype != bool:
        m = np.zeros(n, bool)
        m[val_idx] = True
        val_idx = m
    tr = ~val_idx
    results = {}
    best_c, best_auc = c_grid[0], -np.inf
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000).fit(X[tr], y[tr])
        if len(np.unique(y[val_idx])) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y[val_idx], clf.decision_function(X[val_idx])))
        results[c] = auc
        if np.isfinite(auc) and auc > best_auc:
            best_c, best_auc = c, auc
    final = LogisticRegression(C=best_c, max_iter=2000).fit(X, y)
    return final, best_c, results


class SupervisedProbe(Detector):
    name = "probe_supervised"
    access = AccessSpec(
        activations=True, forwards_per_root=1.0, violation_labels=True, normal_only=False,
        notes="trained on labelled dev data: extra-information reference, not equal-data baseline",
    )

    def __init__(
        self,
        pca_dim: int | None = 32,
        c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        val_frac: float = 0.3,
        seed: int = 0,
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.c_grid = tuple(c_grid)
        self.val_frac = val_frac
        self.seed = seed

    def _features(self, bundle: ResponseBundle) -> np.ndarray:
        F = self.proj_.transform(bundle.h_base.astype(np.float64))
        return self.scaler_.transform(F.reshape(len(F), -1))

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None, val_idx=None) -> SupervisedProbe:
        y = _require_labels(self.name, labels)
        method = "none" if self.pca_dim is None else "pca"
        self.proj_ = LayerProjection(method, self.pca_dim or bundle.d_model, self.seed).fit(bundle.h_base)
        F = self.proj_.transform(bundle.h_base.astype(np.float64))
        self.scaler_ = Standardizer().fit(F.reshape(len(F), -1))
        self.clf_, self.best_c_, self.val_auroc_ = tune_logistic(
            self._features(bundle), y, self.c_grid, val_idx, self.val_frac, self.seed
        )
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        return self.clf_.decision_function(self._features(bundle))


class ContrastPairProbe(Detector):
    name = "probe_contrast"
    access = AccessSpec(
        activations=True, forwards_per_root=1.0,
        notes="generic contrast direction from contrast prompts; no violation labels",
    )

    def __init__(self, pos: np.ndarray | None = None, neg: np.ndarray | None = None):
        """``pos``/``neg``: contrast activations (P, L, D) at the same layers."""
        super().__init__()
        self.pos = pos
        self.neg = neg

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = {"n_pos": None if self.pos is None else len(self.pos),
                       "n_neg": None if self.neg is None else len(self.neg)}
        return d

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> ContrastPairProbe:
        if self.pos is None or self.neg is None:
            raise ValueError("ContrastPairProbe needs contrast activations pos/neg (P, L, D)")
        pos, neg = np.asarray(self.pos, np.float64), np.asarray(self.neg, np.float64)
        if pos.shape[1:] != bundle.h_base.shape[1:]:
            raise ValueError(f"contrast activations {pos.shape[1:]} do not match bundle {bundle.h_base.shape[1:]}")
        H = bundle.h_base.astype(np.float64)
        self.mean_ = H.mean(axis=0)
        sd = H.std(axis=0)
        self.scale_ = np.where(sd > 1e-8, sd, 1.0)
        diff = ((pos - self.mean_) / self.scale_).mean(axis=0) - ((neg - self.mean_) / self.scale_).mean(axis=0)
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        self.direction_ = diff / np.maximum(norms, 1e-12)  # (L, D) unit directions
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        Z = (bundle.h_base.astype(np.float64) - self.mean_) / self.scale_
        return np.einsum("nld,ld->nl", Z, self.direction_).mean(axis=1)
