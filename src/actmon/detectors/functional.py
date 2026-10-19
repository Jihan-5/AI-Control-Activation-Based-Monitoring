r"""FA-style detector: compare a root's loss-response trace with trusted references (syllabus S15).

Traces come from :func:`actmon.actor.functional.functional_traces` (``K`` fixed
parameter perturbations). Trusted references are the normal-fitting roots,
aggregated by proposed verb (class-conditioned aggregation). The score is
:math:`1 - \mathrm{corr}(t(x), \bar t_{\text{verb}(x)})`, where the correlation
is taken across the ``K`` perturbations. Verbs with fewer than 10 fitting roots
fall back to the pooled reference.

An **approximation** of functional-attribution MAD. The access row records that it needs
weight access and ``K`` extra scored passes per root.
"""

from __future__ import annotations

import numpy as np

from actmon.data.features import FeatureSet
from actmon.detectors.base import Access, Detector


class FunctionalAttribution(Detector):
    name = "fa_style"

    def __init__(self, min_class: int = 10):
        self.min_class = min_class
        self.access = Access("none", "none", 0, weights=True, notes="K parameter-perturbed scored passes per root")

    def fit(self, fit: FeatureSet, dev=None, dev_labels=None):
        if "fa_trace" not in fit.extra:
            raise ValueError("feature set lacks fa_trace")
        T = fit.extra["fa_trace"]
        self.access = Access("none", "none", int(T.shape[1]), weights=True,
                             notes="K parameter-perturbed scored passes per root")
        verbs = np.asarray([str(a).split()[0] for a in fit.base_action])
        self.pooled = T.mean(0)
        self.refs = {v: T[verbs == v].mean(0) for v in set(verbs) if (verbs == v).sum() >= self.min_class}
        return self

    def score(self, fs: FeatureSet) -> np.ndarray:
        T = fs.extra["fa_trace"]
        verbs = [str(a).split()[0] for a in fs.base_action]
        R = np.stack([self.refs.get(v, self.pooled) for v in verbs])
        Tc = T - T.mean(1, keepdims=True)
        Rc = R - R.mean(1, keepdims=True)
        corr = (Tc * Rc).sum(1) / (np.linalg.norm(Tc, axis=1) * np.linalg.norm(Rc, axis=1) + 1e-12)
        return 1.0 - corr
