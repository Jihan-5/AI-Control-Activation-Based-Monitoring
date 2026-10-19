"""Detector interface and the access table every method must declare.

The access table makes unequal comparisons explicit (syllabus S09/S10/S15):
which detectors see activations, action scores, parameters, violation labels,
or the exact rules, and how many forward-equivalents per root they spend.
Reports print this table beside every result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np

from irmon.features.bundle import ResponseBundle


@dataclass(frozen=True)
class AccessSpec:
    activations: bool = False  # reads residual-stream activations
    action_scores: bool = False  # reads candidate-set log-probabilities
    forwards_per_root: float = 1.0  # forward-equivalent passes per monitored root
    weights: bool = False  # needs parameter access (e.g. parameter sampling)
    backward_passes: bool = False
    violation_labels: bool = False  # supervised on violation labels
    normal_only: bool = True  # fits on normal data only
    oracle_rules: bool = False  # uses the exact environment rules
    notes: str = ""

    def row(self) -> dict:
        return asdict(self)


class Detector(ABC):
    """Higher score = more anomalous / more likely violating."""

    name: str = "abstract"
    access: AccessSpec = AccessSpec()

    def __init__(self) -> None:
        self.fitted_ = False

    @abstractmethod
    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> Detector:
        """Fit on ``bundle``. Unsupervised detectors must ignore ``labels`` and
        are only ever given normal-fit roots (enforced by SplitGuard)."""

    @abstractmethod
    def score(self, bundle: ResponseBundle) -> np.ndarray:
        """Return one float score per root, shape (N,)."""

    def describe(self) -> dict:
        params = {k: v for k, v in vars(self).items() if not k.endswith("_") and _is_simple(v)}
        return {"name": self.name, "params": params, "access": self.access.row()}

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name}: call fit() before score()")


def _is_simple(v) -> bool:
    return isinstance(v, (int, float, str, bool, type(None), tuple, list))
