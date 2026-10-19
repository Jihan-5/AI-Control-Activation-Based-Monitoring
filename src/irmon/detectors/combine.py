"""Score combination that is calibrated ONCE (plan p.7, syllabus S06).

Independently thresholded components followed by an OR rule do not preserve
the nominal error rate. ``RankCombiner`` instead maps each component score to a
normal-fit-referenced upper-tail rank (fraction of normal-fit scores below it,
a monotone transform fit on normal-fit data only), combines ranks with a fixed
rule, and returns one score that the downstream conformal stage calibrates.

Note: the component detectors are fit on the same normal-fit rows that define
the reference ranks, so in-sample component scores are optimistically low;
this shifts every rank in the same direction and is absorbed by the downstream
calibration on the separate normal-calibration split.
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector
from .linalg import upper_tail_rank


class RankCombiner(Detector):
    name = "combined"

    def __init__(self, detectors: list[Detector], rule: str = "max", fit_components: bool = True):
        super().__init__()
        if rule not in ("max", "mean"):
            raise ValueError(rule)
        if not detectors:
            raise ValueError("RankCombiner needs at least one detector")
        self.detectors = list(detectors)
        self.rule = rule
        self.fit_components = fit_components
        self.name = "combined[" + "+".join(d.name for d in self.detectors) + f"|{rule}]"
        self._update_access()

    def _update_access(self) -> None:
        acc = [d.access for d in self.detectors]
        self.access = AccessSpec(
            activations=any(a.activations for a in acc),
            action_scores=any(a.action_scores for a in acc),
            forwards_per_root=max(a.forwards_per_root for a in acc),
            weights=any(a.weights for a in acc),
            backward_passes=any(a.backward_passes for a in acc),
            violation_labels=any(a.violation_labels for a in acc),
            normal_only=all(a.normal_only for a in acc),
            oracle_rules=any(a.oracle_rules for a in acc),
            notes="rank combination of: " + ", ".join(d.name for d in self.detectors),
        )

    def describe(self) -> dict:
        return {"name": self.name, "params": {"rule": self.rule}, "components": [d.describe() for d in self.detectors],
                "access": self.access.row()}

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> RankCombiner:
        """``bundle`` must be normal-fit data (it defines the reference ranks).
        Supervised components must be fit beforehand; pass fit_components=False."""
        if self.fit_components:
            for d in self.detectors:
                if d.access.violation_labels:
                    raise ValueError(f"fit supervised component {d.name} on dev before combining")
                d.fit(bundle)
        self.refs_ = [np.asarray(d.score(bundle), dtype=np.float64) for d in self.detectors]
        self._update_access()
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        ranks = np.stack([upper_tail_rank(ref, d.score(bundle)) for d, ref in zip(self.detectors, self.refs_)], axis=1)
        return ranks.max(axis=1) if self.rule == "max" else ranks.mean(axis=1)
