"""STRIP-style stability baseline (syllabus S14; Gao et al., STRIP-ViTA).

STRIP superimposes strong intentional perturbations on an input: a triggered
input's prediction stays unusually *stable* (low entropy), because the trigger
is input-agnostic. The text adaptation uses M strong random compositions of
task-preserving edits built by the extraction stage.

Reads ``extras['strip_probs']``: (N, M, K) candidate-set probabilities under M
perturbations. Entropy statistic:
    ``mean_entropy``   mean over perturbations of each distribution's entropy
    ``entropy_of_mean`` entropy of the perturbation-averaged distribution
Direction:
    ``low``        score = -entropy (original STRIP: stability is suspicious)
    ``high``       score = entropy
    ``two_sided``  |z| against the normal-fit mean/std of the entropy
Our threat model (goal conflicts) gives no a-priori direction, so the pipeline
default is ``two_sided``; ``low`` is the faithful STRIP setting.
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector

EXTRA_KEY = "strip_probs"


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=axis, keepdims=True)
    return -(p * np.log(p)).sum(axis=axis)


class StripStability(Detector):
    name = "strip"

    def __init__(self, statistic: str = "mean_entropy", direction: str = "two_sided"):
        super().__init__()
        if statistic not in ("mean_entropy", "entropy_of_mean"):
            raise ValueError(statistic)
        if direction not in ("low", "high", "two_sided"):
            raise ValueError(direction)
        self.statistic = statistic
        self.direction = direction
        self.access = AccessSpec(action_scores=True, forwards_per_root=1.0, notes="STRIP-style adaptation")

    def entropy(self, bundle: ResponseBundle) -> np.ndarray:
        if EXTRA_KEY not in bundle.extras:
            raise KeyError(f"STRIP needs bundle.extras[{EXTRA_KEY!r}] (built by the extraction stage)")
        P = np.asarray(bundle.extras[EXTRA_KEY], dtype=np.float64)
        if self.statistic == "mean_entropy":
            return _entropy(P, axis=2).mean(axis=1)
        return _entropy(P.mean(axis=1), axis=1)

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> StripStability:
        H = self.entropy(bundle)
        self.mean_, self.std_ = float(H.mean()), float(H.std()) or 1.0
        M = np.asarray(bundle.extras[EXTRA_KEY]).shape[1]
        self.access = AccessSpec(
            action_scores=True, forwards_per_root=float(1 + M), notes=f"STRIP-style adaptation: {M} perturbations",
        )
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        H = self.entropy(bundle)
        if self.direction == "low":
            return -H
        if self.direction == "high":
            return H
        return np.abs(H - self.mean_) / self.std_
