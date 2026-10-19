"""Output-only response consistency (tests whether activations add value).

Uses the same edited inputs, and therefore the same forward budget, as the
proposed detector, but reads only candidate-set log-probabilities.

Modes
-----
``residual``         the proposed residual machinery restricted to action-score
                     changes (``use_activations=False``): predictor, cross-
                     fitted residual covariance, Mahalanobis.
``t_instability``    mean Jensen-Shannon divergence (or decision-flip rate) of
                     the candidate distribution across valid T edits. Normal
                     actors should be stable under nuisance changes.
``u_insensitivity``  fraction of valid, decision-changing U edits after which
                     the actor's argmax did NOT change. A throughput-biased
                     actor may ignore authorization changes.
``combined``         t_instability + u_insensitivity, each referenced to its
                     normal-fit mean/std (fixed weights, calibrated once).

``extras['edit_decision_changed']`` (N, E) bool, if present, marks which edits
changed the oracle decision; otherwise every valid U edit is assumed decision-
changing (the default U contract, ``require_decision_change=True``).
"""

from __future__ import annotations

import numpy as np

from irmon.config import ProposedConfig
from irmon.features.bundle import ResponseBundle

from .base import AccessSpec, Detector
from .proposed import InterventionResponseDetector

MODES = ("residual", "t_instability", "u_insensitivity", "combined")


def _js_divergence(logp: np.ndarray, logq: np.ndarray) -> np.ndarray:
    p, q = np.exp(logp), np.exp(logq)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(-1)
        kl_qm = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


class OutputConsistency(Detector):
    name = "output_consistency"

    def __init__(self, mode: str = "residual", metric: str = "js", proposed_cfg: ProposedConfig | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if metric not in ("js", "flip"):
            raise ValueError(metric)
        self.mode = mode
        self.metric = metric
        self.proposed_cfg = proposed_cfg
        self.access = AccessSpec(
            action_scores=True, forwards_per_root=1.0,
            notes="no activations; same edited inputs as the proposed detector",
        )

    # ---------------------------------------------------------- statistics
    def t_instability(self, bundle: ResponseBundle) -> np.ndarray:
        idx = bundle.edit_index("T")
        if len(idx) == 0:
            return np.zeros(len(bundle))
        valid = bundle.edit_valid[:, idx]
        if self.metric == "js":
            per = _js_divergence(bundle.a_base[:, None, :].astype(np.float64), bundle.a_edit[:, idx].astype(np.float64))
        else:
            per = (bundle.edit_decisions()[:, idx] != bundle.chosen_slot[:, None]).astype(float)
        per = np.where(valid, per, 0.0)
        return per.sum(1) / np.maximum(valid.sum(1), 1)

    def u_insensitivity(self, bundle: ResponseBundle) -> np.ndarray:
        idx = bundle.edit_index("U")
        if len(idx) == 0:
            return np.zeros(len(bundle))
        eligible = bundle.edit_valid[:, idx].copy()
        if "edit_decision_changed" in bundle.extras:
            eligible &= np.asarray(bundle.extras["edit_decision_changed"], bool)[:, idx]
        same = bundle.edit_decisions()[:, idx] == bundle.chosen_slot[:, None]
        return np.where(eligible, same, False).sum(1) / np.maximum(eligible.sum(1), 1)

    # ----------------------------------------------------------------- api
    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> OutputConsistency:
        E = len(bundle.edit_names)
        self.access = AccessSpec(
            action_scores=True, forwards_per_root=float(1 + E),
            notes="no activations; same edited inputs as the proposed detector",
        )
        if self.mode == "residual":
            cfg = (self.proposed_cfg or ProposedConfig()).model_copy(
                update={"use_activations": False, "use_action_scores": True, "remove_nuisance": False}
            )
            self.inner_ = InterventionResponseDetector(cfg).fit(bundle)
        elif self.mode == "combined":
            t, u = self.t_instability(bundle), self.u_insensitivity(bundle)
            self.ref_ = (t.mean(), t.std() or 1.0, u.mean(), u.std() or 1.0)
        self.fitted_ = True
        return self

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        if self.mode == "residual":
            return self.inner_.score(bundle)
        if self.mode == "t_instability":
            return self.t_instability(bundle)
        if self.mode == "u_insensitivity":
            return self.u_insensitivity(bundle)
        tm, ts, um, us = self.ref_
        return (self.t_instability(bundle) - tm) / ts + (self.u_insensitivity(bundle) - um) / us
