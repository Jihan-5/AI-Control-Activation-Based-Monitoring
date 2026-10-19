r"""Activation-distance baselines (syllabus S09, adapted to an LLM actor).

All three use the same sites and the same 32-dimensional training-only
projection per site. The per-site projections are concatenated (3 x 32 = 96
dimensions), and one Ledoit-Wolf covariance is fit on the concatenation
(decision A-D5). Thresholds are set elsewhere, on the calibration split.

``mahalanobis_pooled``
    One mean and one covariance from the normal-fitting split.

``mahalanobis_task``
    *Task-conditioned*: conditions on legitimately observable task context. Lee
    et al. fit one mean per class with a tied covariance. Here the "class" is
    the observable context class (ticket resource type, whether a handoff
    message is present, number of pools) that the monitor reads from the
    prompt's surface. The score is the distance to the root's *own* class mean,
    not to the closest class, because the class is observed rather than
    predicted. Sparse classes shrink towards the pooled mean:
    :math:`\tilde\mu_c = (n_c \hat\mu_c + k \hat\mu)/(n_c + k)`, with ``k = 10``.
    Image-gradient input preprocessing has no analogue for discrete tokens and
    is not used.

``mahalanobis_env``
    The invariance comparator (syllabus S13). Template families act as observed
    environments. Each family's normal mean is removed before the covariance is
    fit, and at test time the root's family mean is subtracted (the pooled mean
    for a family never seen in fitting, e.g. held-out families). It removes
    environment-specific *location*, the simplest form of PCIR's goal. It is not
    an MMD-regularised representation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from actmon.data.features import FeatureSet
from actmon.detectors.base import Access, Detector
from actmon.detectors.projection import ShrinkageGaussian, SitePCA


class _ProjectedBase(Detector):
    def __init__(self, dim: int = 32, sites: list[int] | None = None):
        self.dim = dim
        self.sites = sites

    def _mats(self, fs: FeatureSet):
        sites = self.sites or fs.sites
        return {s: fs.base[s] for s in sites}

    def _fit_proj(self, fit: FeatureSet):
        self.proj = SitePCA(self.dim).fit(self._mats(fit))
        return self.proj.concat(self._mats(fit))

    def _z(self, fs: FeatureSet):
        return self.proj.concat(self._mats(fs))


class PooledMahalanobis(_ProjectedBase):
    name = "mahalanobis_pooled"
    access = Access("sites", "none", 0, context="none", notes="normal fitting split only")

    def fit(self, fit, dev=None, dev_labels=None):
        self.g = ShrinkageGaussian().fit(self._fit_proj(fit))
        return self

    def score(self, fs):
        return self.g.score(self._z(fs))


class _ConditionalMahalanobis(_ProjectedBase):
    k_shrink: float = 10.0

    def _keys(self, fs: FeatureSet) -> np.ndarray:
        raise NotImplementedError

    def fit(self, fit, dev=None, dev_labels=None):
        Z = self._fit_proj(fit)
        keys = self._keys(fit)
        self.mu = Z.mean(0)
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        self.means = {}
        for k, idx in groups.items():
            n = len(idx)
            self.means[k] = (n * Z[idx].mean(0) + self.k_shrink * self.mu) / (n + self.k_shrink)
        R = Z - np.stack([self.means[k] for k in keys])
        self.g = ShrinkageGaussian().fit(R, assume_centered=True)
        return self

    def score(self, fs):
        Z = self._z(fs)
        M = np.stack([self.means.get(k, self.mu) for k in self._keys(fs)])
        return self.g.score(Z, M)


class TaskMahalanobis(_ConditionalMahalanobis):
    name = "mahalanobis_task"
    access = Access("sites", "none", 0, context="observable surface context",
                    notes="class = resource type x handoff message present x pool count")

    def _keys(self, fs):
        return fs.context_class


class EnvCentredMahalanobis(_ConditionalMahalanobis):
    name = "mahalanobis_env"
    access = Access("sites", "none", 0, context="template family",
                    notes="invariance comparator: family-mean removal")

    def _keys(self, fs):
        return fs.family
