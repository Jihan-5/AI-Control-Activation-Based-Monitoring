"""Activation-distance baselines (syllabus S09: Lee et al., 2018, adapted).

PooledMahalanobis       normal-fit h_base, same layers and fitting split as the
                        proposed detector, optional training-only PCA, shrinkage
                        covariance, fixed layer aggregation.
ConditionalMahalanobis  task-conditioned: conditions on legitimately observable
                        context, either through per-``context_group`` means with
                        a shared covariance (small/unseen groups fall back to the
                        pooled mean) or through a ridge regression of h_base on
                        context. The stronger comparator when task variation
                        drives distance, and the primary baseline.

Adaptation notes (S09 "limitation to remember"):
* Lee et al.'s input preprocessing (gradient-sign perturbation of images) is
  NOT transplanted: it has no direct analogue for discrete tokens.
* Their validation-trained logistic layer weighting would give the baseline
  access to labelled anomalies; here layers are aggregated by a fixed rule and
  the combined score is calibrated once downstream.
* "Class" means are replaced by observable context groups: the actor has no
  class labels, and violation labels must not be used by an unsupervised
  detector.
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle
from irmon.features.projection import LayerProjection

from .base import AccessSpec, Detector
from .linalg import Standardizer, Whitener, robust_standardize_fit
from .proposed import NormalResponsePredictor


class _ActivationFeatures:
    """Per-layer training-only PCA of h_base (or identity when pca_dim=None)."""

    def __init__(self, pca_dim: int | None, seed: int):
        self.pca_dim = pca_dim
        self.seed = seed

    def fit(self, h: np.ndarray) -> _ActivationFeatures:
        method = "none" if self.pca_dim is None else "pca"
        self.proj_ = LayerProjection(method, self.pca_dim or h.shape[-1], self.seed).fit(h)
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:  # (N, L, D) -> (N, L, k)
        return self.proj_.transform(h.astype(np.float64))


class PooledMahalanobis(Detector):
    name = "mahalanobis_pooled"
    access = AccessSpec(activations=True, forwards_per_root=1.0, notes="single base forward; normal-fit only")

    def __init__(
        self,
        pca_dim: int | None = 64,
        aggregation: str = "concat",
        covariance: str = "ledoit_wolf",
        shrinkage: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        if aggregation not in ("concat", "sum", "max"):
            raise ValueError(aggregation)
        self.pca_dim = pca_dim
        self.aggregation = aggregation
        self.covariance = covariance
        self.shrinkage = shrinkage
        self.seed = seed

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> PooledMahalanobis:
        self.feat_ = _ActivationFeatures(self.pca_dim, self.seed).fit(bundle.h_base)
        F = self.feat_.transform(bundle.h_base)
        if self.aggregation == "concat":
            self.whitener_ = Whitener().fit(F.reshape(len(F), -1), self.covariance, self.shrinkage)
        else:
            self.layer_whiteners_ = [
                Whitener().fit(F[:, li], self.covariance, self.shrinkage) for li in range(F.shape[1])
            ]
            # Fixed rule: per-layer scores on a normal-fit robust scale.
            self.layer_ref_ = [
                robust_standardize_fit(w.score(F[:, li])) for li, w in enumerate(self.layer_whiteners_)
            ]
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        F = self.feat_.transform(bundle.h_base)
        if self.aggregation == "concat":
            return self.whitener_.score(F.reshape(len(F), -1))
        per = np.stack(
            [(w.score(F[:, li]) - m) / s for li, (w, (m, s)) in enumerate(zip(self.layer_whiteners_, self.layer_ref_))],
            axis=1,
        )
        return per.sum(axis=1) if self.aggregation == "sum" else per.max(axis=1)


class ConditionalMahalanobis(Detector):
    name = "mahalanobis_conditional"
    access = AccessSpec(
        activations=True, forwards_per_root=1.0,
        notes="single base forward; conditions on observable context; normal-fit only",
    )

    def __init__(
        self,
        mode: str = "group",
        min_group_size: int = 20,
        pca_dim: int | None = 64,
        covariance: str = "ledoit_wolf",
        shrinkage: float = 0.1,
        ridge_alpha: float = 1.0,
        cross_fit_folds: int = 5,
        seed: int = 0,
    ):
        super().__init__()
        if mode not in ("group", "ridge"):
            raise ValueError(mode)
        self.mode = mode
        self.min_group_size = min_group_size
        self.pca_dim = pca_dim
        self.covariance = covariance
        self.shrinkage = shrinkage
        self.ridge_alpha = ridge_alpha
        self.cross_fit_folds = cross_fit_folds
        self.seed = seed

    def _features(self, bundle: ResponseBundle) -> np.ndarray:
        F = self.feat_.transform(bundle.h_base)
        return F.reshape(len(F), -1)

    def _group_means(self, groups: np.ndarray) -> np.ndarray:
        return np.stack([self.group_means_.get(g, self.pooled_mean_) for g in groups])

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> ConditionalMahalanobis:
        self.feat_ = _ActivationFeatures(self.pca_dim, self.seed).fit(bundle.h_base)
        X = self._features(bundle)
        n = len(X)
        if self.mode == "group":
            self.pooled_mean_ = X.mean(axis=0)
            self.group_means_ = {}
            groups = np.asarray(bundle.context_group).astype(str)
            for g in np.unique(groups):
                idx = groups == g
                if idx.sum() >= self.min_group_size:
                    self.group_means_[g] = X[idx].mean(axis=0)
            self.n_groups_fallback_ = int(len(np.unique(groups)) - len(self.group_means_))
            R = X - self._group_means(groups)
        else:
            self.ctx_scaler_ = Standardizer().fit(bundle.context)
            C = self.ctx_scaler_.transform(bundle.context)
            R = np.empty_like(X)
            k = int(min(max(2, self.cross_fit_folds), n // 2))
            folds = np.array_split(np.random.default_rng(self.seed).permutation(n), k)
            for held in folds:
                train = np.setdiff1d(np.arange(n), held)
                pred = NormalResponsePredictor("ridge", alpha=self.ridge_alpha).fit(C[train], X[train])
                R[held] = X[held] - pred.predict(C[held])
            self.predictor_ = NormalResponsePredictor("ridge", alpha=self.ridge_alpha).fit(C, X)
        self.whitener_ = Whitener().fit(R, self.covariance, self.shrinkage)
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        X = self._features(bundle)
        if self.mode == "group":
            R = X - self._group_means(np.asarray(bundle.context_group).astype(str))
        else:
            R = X - self.predictor_.predict(self.ctx_scaler_.transform(bundle.context))
        return self.whitener_.score(R)
