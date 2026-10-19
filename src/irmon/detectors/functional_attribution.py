"""Functional-attribution comparator (syllabus S15; Lyons Keenan et al., 2026, adapted).

The method compares *loss-response traces under parameter sampling* with
trusted reference traces, using class-conditioned aggregation, instead of the
location of a single activation vector.

Adaptation used here:
* Trace: ``extras['loss_trace']`` (N, S) = loss (negative log-prob) of the
  actor's own proposed action under S seeded random parameter perturbations
  (built by ``ActorBackend.loss_trace``). The same S seeds are used for every
  root, so traces are comparable coordinate-wise.
* Trusted references: normal-fit roots, grouped by the proposed action slot
  (``chosen_slot``) - the class-conditioned aggregation. Classes with fewer
  than ``min_class_size`` references fall back to the pooled reference.
* Score ``correlation``: 1 - Pearson correlation between a root's trace and its
  class reference mean trace. Option ``mahalanobis``: shrinkage Mahalanobis of
  the trace against its class mean with a shared covariance.

Access is materially different from input-intervention methods: it needs
parameter access and S extra forwards per root (``weights=True``). Input-
intervention responses and parameter-perturbation traces are different
experimental objects; report the cost-quality curve over S, not one
deliberately underpowered S.
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector
from .linalg import Whitener

EXTRA_KEY = "loss_trace"


def _rowwise_corr(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Rc = R - R.mean(axis=1, keepdims=True)
    num = (Xc * Rc).sum(axis=1)
    den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(Rc, axis=1)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


class FunctionalAttribution(Detector):
    name = "functional_attribution"

    def __init__(self, statistic: str = "correlation", min_class_size: int = 10, covariance: str = "ledoit_wolf"):
        super().__init__()
        if statistic not in ("correlation", "mahalanobis"):
            raise ValueError(statistic)
        self.statistic = statistic
        self.min_class_size = min_class_size
        self.covariance = covariance
        self.access = AccessSpec(weights=True, forwards_per_root=1.0, notes="parameter-sampling traces; adaptation")

    def _traces(self, bundle: ResponseBundle) -> np.ndarray:
        if EXTRA_KEY not in bundle.extras:
            raise KeyError(f"functional attribution needs bundle.extras[{EXTRA_KEY!r}]")
        return np.asarray(bundle.extras[EXTRA_KEY], dtype=np.float64)

    def _refs(self, slots: np.ndarray) -> np.ndarray:
        return np.stack([self.class_means_.get(int(s), self.pooled_mean_) for s in slots])

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> FunctionalAttribution:
        T = self._traces(bundle)
        slots = np.asarray(bundle.chosen_slot).astype(int)
        self.pooled_mean_ = T.mean(axis=0)
        self.class_means_ = {
            int(s): T[slots == s].mean(axis=0) for s in np.unique(slots) if (slots == s).sum() >= self.min_class_size
        }
        if self.statistic == "mahalanobis":
            self.whitener_ = Whitener().fit(T - self._refs(slots), self.covariance)
        self.access = AccessSpec(
            weights=True, forwards_per_root=float(1 + T.shape[1]),
            notes=f"{T.shape[1]} parameter-sampling forwards per root; class-conditioned references; adaptation",
        )
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        T = self._traces(bundle)
        ref = self._refs(np.asarray(bundle.chosen_slot).astype(int))
        if self.statistic == "correlation":
            return 1.0 - _rowwise_corr(T, ref)
        return self.whitener_.score(T - ref)
