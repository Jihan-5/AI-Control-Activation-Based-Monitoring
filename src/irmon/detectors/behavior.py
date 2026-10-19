"""Behavior-only logistic monitor (technical specification, "Baselines and access").

A regularized logistic classifier using response/action scores and predefined
observable behavioral features, but NO actor activations. Coefficients are fit
on the development training portion and C is tuned on a disjoint development
validation portion. It has access to known violation labels, which the access
table discloses.

Features: candidate-set log-probabilities of the base input, entropy and top-2
margin of the candidate distribution, chosen-slot one-hot, and the observable
context covariates.
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector
from .linalg import Standardizer
from .probes import _require_labels, tune_logistic


def behavioral_features(bundle: ResponseBundle, include_context: bool = True) -> np.ndarray:
    a = bundle.a_base.astype(np.float64)
    p = np.exp(a - a.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1, keepdims=True)
    top2 = np.sort(a, axis=1)[:, -2:]
    margin = (top2[:, 1] - top2[:, 0])[:, None]
    onehot = np.eye(len(bundle.slot_names))[np.asarray(bundle.chosen_slot).astype(int)]
    parts = [a, ent, margin, onehot]
    if include_context:
        parts.append(bundle.context.astype(np.float64))
    return np.concatenate(parts, axis=1)


class BehaviorLogistic(Detector):
    name = "behavior_logistic"
    access = AccessSpec(
        action_scores=True, forwards_per_root=1.0, violation_labels=True, normal_only=False,
        notes="no activations; trained on labelled dev data (disclosed label access)",
    )

    def __init__(
        self,
        include_context: bool = True,
        c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        val_frac: float = 0.3,
        seed: int = 0,
    ):
        super().__init__()
        self.include_context = include_context
        self.c_grid = tuple(c_grid)
        self.val_frac = val_frac
        self.seed = seed

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None, val_idx=None) -> BehaviorLogistic:
        y = _require_labels(self.name, labels)
        X = behavioral_features(bundle, self.include_context)
        self.scaler_ = Standardizer().fit(X)
        self.clf_, self.best_c_, self.val_auroc_ = tune_logistic(
            self.scaler_.transform(X), y, self.c_grid, val_idx, self.val_frac, self.seed
        )
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        return self.clf_.decision_function(self.scaler_.transform(behavioral_features(bundle, self.include_context)))
