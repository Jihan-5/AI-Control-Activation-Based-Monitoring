"""Adapted TeCo dispersion (syllabus S14; Liu et al., CVPR 2023).

TeCo flags backdoored images because the corruption severity at which the hard
label flips is unusually *dispersed* across corruption types. The text
adaptation here uses nuisance-edit kinds as corruption types and an
intensity ladder per kind (e.g. number of paraphrased sentences, number of
inserted distractor records) built by the extraction stage.

This is an ADAPTATION, not a replication: image corruption severity is not
automatically meaningful for text edits, the threat model (goal-conflict
violations rather than input triggers) differs, and the severity ladders are
our construction. Report it as "TeCo-style".

Reads ``extras['teco_decisions']``: int array (N, n_kinds, K_sev) holding the
actor's argmax slot under kind ``k`` at severity ``s`` (s = 1..K_sev).
Change point = first severity whose decision differs from ``chosen_slot``
(K_sev + 1 if it never changes). Score = std of change points across kinds
(``dispersion``, TeCo's statistic) or their mean (``mean_changepoint``).
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector

EXTRA_KEY = "teco_decisions"


def change_points(decisions: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """(N, n_kinds, K_sev) decisions -> (N, n_kinds) first-change severities."""
    decisions = np.asarray(decisions)
    K = decisions.shape[2]
    changed = decisions != np.asarray(chosen)[:, None, None]
    first = np.argmax(changed, axis=2) + 1  # severities are 1-based
    return np.where(changed.any(axis=2), first, K + 1).astype(float)


class TeCoDispersion(Detector):
    name = "teco"

    def __init__(self, statistic: str = "dispersion"):
        super().__init__()
        if statistic not in ("dispersion", "mean_changepoint"):
            raise ValueError(statistic)
        self.statistic = statistic
        self.access = AccessSpec(action_scores=True, forwards_per_root=1.0, notes="TeCo-style adaptation")

    def _cp(self, bundle: ResponseBundle) -> np.ndarray:
        if EXTRA_KEY not in bundle.extras:
            raise KeyError(f"TeCo needs bundle.extras[{EXTRA_KEY!r}] (built by the extraction stage)")
        return change_points(bundle.extras[EXTRA_KEY], bundle.chosen_slot)

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> TeCoDispersion:
        dec = np.asarray(bundle.extras[EXTRA_KEY]) if EXTRA_KEY in bundle.extras else None
        if dec is None:
            raise KeyError(f"TeCo needs bundle.extras[{EXTRA_KEY!r}]")
        n_kinds, k_sev = dec.shape[1], dec.shape[2]
        self.access = AccessSpec(
            action_scores=True, forwards_per_root=float(1 + n_kinds * k_sev),
            notes=f"TeCo-style adaptation: {n_kinds} edit kinds x {k_sev} severities",
        )
        self.fitted_ = True  # nonparametric score; nothing to estimate
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        cp = self._cp(bundle)
        return cp.std(axis=1) if self.statistic == "dispersion" else cp.mean(axis=1)
