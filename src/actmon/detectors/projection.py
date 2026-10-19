r"""Training-only projections and shrinkage Mahalanobis geometry.

**Projection** (Technical Specification: "a 32-dimensional training-only
projection per layer"). For every site, PCA is fit on the *normal-fitting
split only*:

.. math::  z_l(x) = W_l^\top (a_l(x) - \mu_l), \qquad W_l \in \mathbb R^{d \times 32}.

The calibration, development and test splits are only transformed, never used
to fit. Response signatures use a separate per-layer projection fit on
normal-fitting *differences* :math:`a_l(T_m x) - a_l(x)`. Base-activation PCA
may not span the directions in which transformations move the representation.

**Shrinkage Mahalanobis.** Ledoit-Wolf covariance (Ledoit and Wolf 2004) on the
fitted features, and :math:`d^2(z) = (z - m)^\top \hat\Sigma^{-1} (z - m)`.
Shrinkage keeps the inverse stable when the dimension is a sizeable fraction of
the 1,000 fitting roots.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class SitePCA:
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.mu: dict[int, np.ndarray] = {}
        self.W: dict[int, np.ndarray] = {}

    def fit(self, mats: dict[int, np.ndarray]) -> "SitePCA":
        for s, X in mats.items():
            X = np.asarray(X, dtype=np.float64)
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            k = min(self.dim, Vt.shape[0])
            self.mu[s], self.W[s] = mu, Vt[:k].T
        return self

    def transform(self, mats: dict[int, np.ndarray], center: bool = True) -> dict[int, np.ndarray]:
        return {s: ((np.asarray(X, np.float64) - (self.mu[s] if center else 0.0)) @ self.W[s]) for s, X in mats.items()}

    def concat(self, mats: dict[int, np.ndarray], center: bool = True) -> np.ndarray:
        z = self.transform(mats, center)
        return np.concatenate([z[s] for s in sorted(z)], axis=1)


class ShrinkageGaussian:
    """Mean and Ledoit-Wolf precision. ``score`` returns squared Mahalanobis distances."""

    def fit(self, Z: np.ndarray, assume_centered: bool = False) -> "ShrinkageGaussian":
        """``assume_centered=True``: ``Z`` already holds residuals (e.g. from class means), so the mean is 0."""
        Z = np.asarray(Z, np.float64)
        self.m = np.zeros(Z.shape[1]) if assume_centered else Z.mean(0)
        lw = LedoitWolf(assume_centered=True).fit(Z - self.m)
        self.shrinkage = float(lw.shrinkage_)
        self.P = np.linalg.pinv(lw.covariance_, hermitian=True)
        return self

    def score(self, Z: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
        """Squared distance to ``m`` (default: the fitted mean). ``m`` may hold one row per input."""
        D = np.asarray(Z, np.float64) - (self.m if m is None else m)
        return np.einsum("ij,jk,ik->i", D, self.P, D)
