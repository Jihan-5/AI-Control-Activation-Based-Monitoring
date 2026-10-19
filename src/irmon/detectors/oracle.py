"""Exact simulator checker: the oracle ceiling where rules are fully available.

Reads ``extras['oracle_flag']`` (N,), 1 if the actor's proposed action violates
the environment's exact policy rules (computed at extraction time from the
visible prompt and the proposed action via the public rule checker). In the
finite-state environment this is a perfect detector, which is why that
environment alone cannot establish the practical necessity of activation
monitoring (plan p.4).
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector

EXTRA_KEY = "oracle_flag"


class ExactRuleChecker(Detector):
    name = "oracle"
    access = AccessSpec(
        action_scores=True, forwards_per_root=1.0, oracle_rules=True,
        notes="exact policy rules: ceiling, not a deployable monitor in general",
    )

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> ExactRuleChecker:
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        if EXTRA_KEY not in bundle.extras:
            raise KeyError(f"oracle needs bundle.extras[{EXTRA_KEY!r}]")
        return np.asarray(bundle.extras[EXTRA_KEY], dtype=np.float64)
