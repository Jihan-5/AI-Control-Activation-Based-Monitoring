"""Numerically careful linear algebra shared by every distance-based detector.

Rules followed everywhere:
* Every statistic (means, scales, covariances) is fit on training rows only.
* Covariances are shrunk (Ledoit-Wolf, OAS, or a fixed convex shrinkage) and
  inverted through ``eigh`` with an eigenvalue floor; ``np.linalg.inv`` of an
  ill-conditioned matrix is never used (plan p.7: "avoid unstable covariance
  inversion in a high-dimensional signature").
* NaNs are handled explicitly: imputed by the caller's declared rule, never
  silently propagated into a score.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import OAS, LedoitWolf

COVARIANCE_METHODS = ("ledoit_wolf", "oas", "fixed", "empirical")


def nanmean0(X: np.ndarray) -> np.ndarray:
    """Column means ignoring NaN; all-NaN columns get 0 (no warnings)."""
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    cnt = finite.sum(axis=0)
    tot = np.where(finite, X, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)


class Standardizer:
    """Column-wise z-scoring with training-only statistics.

    Constant columns get unit scale (they carry no information but must not
    produce division by zero). NaNs are ignored when fitting.
    """

    def __init__(self, with_mean: bool = True, eps: float = 1e-8):
        self.with_mean = with_mean
        self.eps = eps

    def fit(self, X: np.ndarray) -> Standardizer:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = nanmean0(X) if self.with_mean else np.zeros(X.shape[1])
        sd = np.sqrt(nanmean0((X - self.mean_) ** 2))
        self.scale_ = np.where(sd > self.eps, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def shrinkage_covariance(
    X: np.ndarray,
    method: str = "ledoit_wolf",
    shrinkage: float = 0.1,
    assume_centered: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (location, covariance, shrinkage_used) estimated from rows of X.

    ``fixed``: (1 - λ) S + λ · tr(S)/p · I with λ = ``shrinkage``.
    ``empirical`` exists only for diagnostics (it is never a default).
    Rows containing NaN must be imputed before calling.
    """
    X = np.asarray(X, dtype=np.float64)
    if np.isnan(X).any():
        raise ValueError("shrinkage_covariance received NaNs; impute first")
    n, p = X.shape
    if n < 2:
        raise ValueError(f"need at least 2 rows to estimate a covariance, got {n}")
    loc = np.zeros(p) if assume_centered else X.mean(axis=0)
    if method == "ledoit_wolf":
        est = LedoitWolf(assume_centered=True).fit(X - loc)
        return loc, est.covariance_, float(est.shrinkage_)
    if method == "oas":
        est = OAS(assume_centered=True).fit(X - loc)
        return loc, est.covariance_, float(est.shrinkage_)
    Xc = X - loc
    S = Xc.T @ Xc / max(n - 1, 1)
    if method == "empirical":
        return loc, S, 0.0
    if method == "fixed":
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("fixed shrinkage must lie in [0, 1]")
        mu = np.trace(S) / p
        return loc, (1.0 - shrinkage) * S + shrinkage * mu * np.eye(p), float(shrinkage)
    raise ValueError(f"unknown covariance method {method!r}; expected one of {COVARIANCE_METHODS}")


class Whitener:
    """Mahalanobis distance through an eigendecomposition with a floor.

    Eigenvalues below ``floor_ratio * max_eigenvalue`` are raised to that
    floor, so directions with (near-)zero training variance cannot explode a
    score. ``score(x) = (x - loc)^T Σ^{-1} (x - loc)``.
    """

    def __init__(self, floor_ratio: float = 1e-6):
        self.floor_ratio = floor_ratio

    def fit_covariance(self, loc: np.ndarray, cov: np.ndarray) -> Whitener:
        cov = 0.5 * (cov + cov.T)
        evals, evecs = np.linalg.eigh(cov)
        top = max(float(evals.max()), 1e-12)
        floor = self.floor_ratio * top
        self.n_floored_ = int((evals < floor).sum())
        evals = np.maximum(evals, floor)
        self.loc_ = np.asarray(loc, dtype=np.float64)
        self.evecs_ = evecs
        self.inv_sqrt_ = 1.0 / np.sqrt(evals)
        self.condition_number_ = float(evals.max() / evals.min())
        return self

    def fit(
        self, X: np.ndarray, method: str = "ledoit_wolf", shrinkage: float = 0.1
    ) -> Whitener:
        loc, cov, lam = shrinkage_covariance(X, method=method, shrinkage=shrinkage)
        self.shrinkage_ = lam
        return self.fit_covariance(loc, cov)

    def whiten(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.loc_) @ self.evecs_
        return Z * self.inv_sqrt_

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.whiten(X)
        return np.einsum("ij,ij->i", Z, Z)


def impute_nan(X: np.ndarray, value: float | np.ndarray = 0.0) -> np.ndarray:
    """Replace NaNs by ``value`` (scalar or per-column vector)."""
    X = np.array(X, dtype=np.float64, copy=True)
    mask = np.isnan(X)
    if not mask.any():
        return X
    if np.ndim(value) == 0:
        X[mask] = value
    else:
        X[mask] = np.broadcast_to(np.asarray(value, dtype=np.float64), X.shape)[mask]
    return X


def upper_tail_rank(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of ``reference`` scores strictly below each score, in [0, 1].

    A monotone, reference-calibrated transform used to put heterogeneous
    detector scores on a common scale before combining them.
    """
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ref, np.asarray(scores, dtype=np.float64), side="left") / max(len(ref), 1)


def robust_standardize_fit(scores: np.ndarray) -> tuple[float, float]:
    """(median, scale) with scale = IQR/1.349, used to put block scores on a
    normal-fit-referenced common scale before a fixed ``max`` aggregation."""
    s = np.asarray(scores, dtype=np.float64)
    med = float(np.median(s))
    q75, q25 = np.percentile(s, [75, 25])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(s)) or 1.0
    return med, scale
