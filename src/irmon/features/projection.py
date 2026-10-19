"""Training-only projections and the nuisance-subspace estimator.

Plan p.6: "Fit a training-only projection, starting at 32 dimensions per layer.
Random projection and PCA are both baselines; neither may use test activations
for fitting." Everything here is fit on normal-fit rows only and then frozen.

Conventions
-----------
* Inputs are arrays whose last two axes are (L layers, D width), e.g. responses
  of shape (N, E, L, D) or activations of shape (N, L, D). NaN rows (invalid
  edits) are dropped when fitting and stay NaN when transforming.
* Projections are *pure linear maps* ``x -> V_l^T x``. PCA directions are found
  on centred data, but ``transform`` does not subtract the training mean:
  responses are differences, and downstream predictors/standardisers own the
  location parameters.
* The nuisance subspace B_l is spanned by the top-r eigenvectors of the
  *uncentred* second moment of T-response vectors, so a systematic mean
  nuisance response is also captured. ``remove(x) = (I - B_l B_l^T) x``.
  Removing B also erases any failure shift lying inside span(B_l); that
  trade-off is the subject of ``irmon.theory`` (plan p.8).
"""

from __future__ import annotations

import numpy as np

PROJECTION_METHODS = ("pca", "random", "none")


def _rows_per_layer(X: np.ndarray) -> list[np.ndarray]:
    """Collect finite (n, D) training rows for each layer from (..., L, D)."""
    X = np.asarray(X, dtype=np.float64)
    L, D = X.shape[-2], X.shape[-1]
    flat = X.reshape(-1, L, D)
    out = []
    for li in range(L):
        rows = flat[:, li, :]
        rows = rows[np.isfinite(rows).all(axis=1)]
        out.append(rows)
    return out


def _top_eigvecs(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of a symmetric PSD matrix, columns ordered by
    decreasing eigenvalue, with a deterministic sign convention."""
    evals, evecs = np.linalg.eigh(0.5 * (M + M.T))
    order = np.argsort(evals)[::-1][:k]
    V = evecs[:, order]
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    signs[signs == 0] = 1.0
    return V * signs


class LayerProjection:
    """Per-layer linear map R^D -> R^k, fit on training data only."""

    def __init__(self, method: str = "pca", width: int = 32, seed: int = 0):
        if method not in PROJECTION_METHODS:
            raise ValueError(f"unknown projection {method!r}; expected one of {PROJECTION_METHODS}")
        self.method = method
        self.width = width
        self.seed = seed

    def fit(self, X: np.ndarray) -> LayerProjection:
        """``X``: (..., L, D) training rows (responses or activations)."""
        rows = _rows_per_layer(X)
        D = np.asarray(X).shape[-1]
        self.components_: list[np.ndarray] = []
        rng = np.random.default_rng(self.seed)
        for li, R in enumerate(rows):
            if self.method == "none":
                V = np.eye(D)
            elif self.method == "random":
                k = min(self.width, D)
                G = rng.normal(size=(D, k))
                V, _ = np.linalg.qr(G)
            else:  # pca
                if len(R) < 2:
                    raise ValueError(f"layer {li}: need >= 2 finite training rows for PCA, got {len(R)}")
                k = min(self.width, D, len(R) - 1)
                Rc = R - R.mean(axis=0)
                V = _top_eigvecs(Rc.T @ Rc / (len(R) - 1), k)
            self.components_.append(V)
        self.n_layers_ = len(rows)
        return self

    @property
    def out_width(self) -> int:
        return int(self.components_[0].shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(..., L, D) -> (..., L, k). NaN rows stay NaN."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[-2] != self.n_layers_:
            raise ValueError(f"expected {self.n_layers_} layers, got {X.shape[-2]}")
        return np.stack([X[..., li, :] @ V for li, V in enumerate(self.components_)], axis=-2)


class NuisanceSubspace:
    """Per-layer subspace spanned by nuisance (T) responses on normal data."""

    def __init__(self, rank: int = 8):
        self.rank = rank

    def fit(self, t_responses: np.ndarray) -> NuisanceSubspace:
        """``t_responses``: (N, E_T, L, D) raw T responses of normal-fit roots."""
        rows = _rows_per_layer(t_responses)
        self.bases_: list[np.ndarray] = []
        self.explained_: list[float] = []
        for li, R in enumerate(rows):
            if len(R) < 1:
                raise ValueError(f"layer {li}: no finite T responses to estimate nuisance subspace")
            D = R.shape[1]
            k = min(self.rank, D, len(R))
            M = R.T @ R / len(R)
            B = _top_eigvecs(M, k) if k > 0 else np.zeros((D, 0))
            total = float(np.trace(M)) or 1.0
            self.explained_.append(float(np.trace(B.T @ M @ B)) / total)
            self.bases_.append(B)
        self.n_layers_ = len(rows)
        return self

    def remove(self, X: np.ndarray) -> np.ndarray:
        """(..., L, D) -> (..., L, D) with each layer's nuisance span removed."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for li, B in enumerate(self.bases_):
            x = X[..., li, :]
            out[..., li, :] = x - (x @ B) @ B.T
        return out

    def overlap(self, direction: np.ndarray, layer: int) -> float:
        """|B^T f| / |f| for a direction f in one layer (diagnostic, plan p.8)."""
        f = np.asarray(direction, dtype=np.float64)
        B = self.bases_[layer]
        return float(np.linalg.norm(B.T @ f) / (np.linalg.norm(f) + 1e-12))
