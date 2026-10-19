"""Metadata-only shortcut classifier (leakage diagnostic, plan p.9).

Trains an intentionally simple logistic classifier on non-semantic metadata
(``extras['metadata']``: e.g. prompt length, line count, format, number of
records, edit-validity pattern, bits of the root-ID hash). If it predicts the
violation label well above chance, labels, root IDs, future actions or
transformation identifiers may be encoding the target: investigate before
trusting any detector result. It is a diagnostic, never a monitor.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector
from .linalg import Standardizer
from .probes import _require_labels, tune_logistic

EXTRA_KEY = "metadata"


class MetadataShortcut(Detector):
    name = "metadata_shortcut"
    access = AccessSpec(
        forwards_per_root=0.0, violation_labels=True, normal_only=False,
        notes="leakage diagnostic on non-semantic metadata; never a monitor",
    )

    def __init__(self, c_grid: tuple[float, ...] = (0.01, 0.1, 1.0), val_frac: float = 0.3,
                 alarm_auroc: float = 0.6, seed: int = 0):
        super().__init__()
        self.c_grid = tuple(c_grid)
        self.val_frac = val_frac
        self.alarm_auroc = alarm_auroc
        self.seed = seed

    def _X(self, bundle: ResponseBundle) -> np.ndarray:
        if EXTRA_KEY not in bundle.extras:
            raise KeyError(f"metadata shortcut needs bundle.extras[{EXTRA_KEY!r}]")
        return np.asarray(bundle.extras[EXTRA_KEY], dtype=np.float64)

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None, val_idx=None) -> MetadataShortcut:
        y = _require_labels(self.name, labels)
        X = self._X(bundle)
        self.scaler_ = Standardizer().fit(X)
        self.clf_, self.best_c_, self.val_auroc_ = tune_logistic(
            self.scaler_.transform(X), y, self.c_grid, val_idx, self.val_frac, self.seed
        )
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        return self.clf_.decision_function(self.scaler_.transform(self._X(bundle)))

    def leakage_report(self, bundle: ResponseBundle, labels: np.ndarray) -> dict:
        """AUROC on held-out labelled data and whether it crosses the alarm level."""
        y = np.asarray(labels).astype(int)
        auc = float(roc_auc_score(y, self.score(bundle))) if len(np.unique(y)) == 2 else float("nan")
        return {"auroc": auc, "alarm": bool(np.isfinite(auc) and auc > self.alarm_auroc),
                "alarm_threshold": self.alarm_auroc}
