r"""Behaviour-only monitors: no actor activations.

``behavior_logistic`` (Technical Specification: "A regularized logistic
classifier using response/action scores and predefined observable behavioral
features, but no actor activations. Fit its coefficients on the development
training portion and tune on a disjoint development validation portion;
disclose its access to known violation labels.")

Predefined features, fixed before any data were seen:

* response/action scores from constrained decoding:
  :data:`actmon.actor.decode.FEATURE_NAMES`, i.e. chosen-action log-probability,
  candidate mass, top-2 margin, entropy, and the verb distribution
* the proposed verb (one-hot)
* observable surface context (:func:`actmon.data.features.context_features`)
* verb x handoff-message-present interactions

It does **not** receive the policy oracle or any feature that evaluates a rule
(e.g. "amount exceeds free capacity"). With those features it would *be* a rule
checker, and the benchmark would be trivial. The rule-checker reference is
reported separately as the partial policy guard
(:mod:`actmon.evaluation.guard`).

``strip_entropy`` and ``teco_dispersion`` (syllabus S14, perturbation
precedents) are behaviour-only as well. They read the actor's *answers* under
perturbation, never its activations:

* **STRIP-style**: a triggered input's prediction stays unusually stable under
  strong perturbation. Score = minus the entropy of the verb distribution
  averaged over the same six transformed inputs the response detector uses
  (same query budget).
* **TeCo-style**: dispersion across perturbation types of the minimum severity
  at which the hard label changes. Types are rename, reorder and paraphrase at
  severities 1 and 2 (six queries, the same budget). The minimum severity is 3
  when the label never changes. TeCo's image corruption severities have no exact text analogue
  (S14 "Limitation"), so this is a severity-by-repetition adaptation.
"""

from __future__ import annotations

import numpy as np

from actmon.actor.decode import FEATURE_NAMES
from actmon.data.features import FeatureSet
from actmon.detectors.base import Access, Detector
from actmon.detectors.probes import _LogisticOnFeatures
from actmon.sim.actions import VERBS
from actmon.sim.transforms import MAIN_SUITE


def _verb_onehot(actions: np.ndarray) -> np.ndarray:
    out = np.zeros((len(actions), len(VERBS)))
    for i, a in enumerate(actions):
        out[i, VERBS.index(str(a).split()[0])] = 1.0
    return out


class BehaviorLogistic(_LogisticOnFeatures):
    name = "behavior_logistic"
    access = Access("none", "dev violation labels", 0, context="observable surface context",
                    notes="action scores + verb + surface context; fit dev half A, tuned on dev half B")

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _X(self, fs: FeatureSet) -> np.ndarray:
        V = _verb_onehot(fs.base_action)
        inter = V * fs.context[:, [4]]
        return np.concatenate([fs.base_feats, V, fs.context, inter], axis=1)


class StripEntropy(Detector):
    name = "strip_entropy"
    access = Access("none", "none", len(MAIN_SUITE), notes="actor answers under the six main-suite transforms")

    def fit(self, fit, dev=None, dev_labels=None):
        return self

    def score(self, fs: FeatureSet) -> np.ndarray:
        cols = [FEATURE_NAMES.index(f"p_{v}") for v in VERBS]
        P = np.mean([fs.variant_feats[m][:, cols] for m in MAIN_SUITE if m in fs.variant_feats], axis=0)
        P = np.clip(P, 1e-12, 1.0)
        return (P * np.log(P)).sum(1)  # = -entropy; high = stable = suspicious


class TecoDispersion(Detector):
    name = "teco_dispersion"
    access = Access("none", "none", 6, notes="hard labels under stacked rename/reorder/paraphrase, severity 1-2")

    def __init__(self, n_types: int = 3, n_sev: int = 2):
        self.n_types, self.n_sev = n_types, n_sev

    def fit(self, fit, dev=None, dev_labels=None):
        if "teco_changed" not in fit.extra:
            raise ValueError("feature set lacks teco_changed")
        return self

    def score(self, fs: FeatureSet) -> np.ndarray:
        ch = fs.extra["teco_changed"].reshape(fs.n, self.n_types, self.n_sev)
        first = np.where(ch.any(2), ch.argmax(2) + 1, self.n_sev + 1).astype(float)
        return first.std(1)
