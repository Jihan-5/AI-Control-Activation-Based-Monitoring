"""Detector interface and access records.

Every detector maps a :class:`~actmon.data.features.FeatureSet` to one score per
root, where **higher means more suspicious**. Thresholds are never chosen here:
:mod:`actmon.evaluation.calibrate` sets them on the separate normal calibration split.

``fit(fit, dev=None, dev_labels=None)`` receives:

* ``fit``: the normal-fitting split (normal roots only, but the detector is not told so)
* ``dev`` and ``dev_labels``: the development split and its 0/1 violation
  labels. The runner passes them only to detectors whose ``access.labels`` is
  not ``"none"`` (``actmon.experiments.common.fit_detector``). Unsupervised
  detectors never see a label.

The :class:`Access` record is written next to every result table. The
Technical Specification asks to "disclose differences in supervision, query
count, and compute", and the syllabus (S09, S10, S15) asks for an access table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from actmon.data.features import FeatureSet


@dataclass(frozen=True)
class Access:
    activations: str  # "none" | "sites" | "sites+variants"
    labels: str  # "none" | "dev violation labels" | "contrast instructions (no labels)"
    extra_forward_passes: int  # actor forward passes per root beyond the base pass
    weights: bool = False  # needs parameter access (perturbation)
    context: str = "none"  # "none" | "observable surface context" | "template family"
    notes: str = ""

    def row(self) -> dict:
        return asdict(self)


class Detector:
    name: str = "detector"
    access: Access = Access("none", "none", 0)
    supervised: bool = False

    def fit(self, fit: FeatureSet, dev: FeatureSet | None = None, dev_labels: np.ndarray | None = None):
        raise NotImplementedError

    def score(self, fs: FeatureSet) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"detector": self.name, "supervised": self.supervised, **self.access.row()}
