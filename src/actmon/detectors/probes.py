r"""Probe baselines (syllabus S10).

``linear_probe``
    L2-regularised logistic regression on the same projected sites (96 dims,
    standardised). Trained on the *development* split: a fixed half is used to fit
    the coefficients and the other half to choose ``C`` by AUROC. It then sees
    the calibration split only through the shared threshold rule. **It receives
    violation labels**, supervision the unsupervised detectors lack, and its
    access row says so.

``contrast_probe``
    A generic contrast-pair probe that uses no knowledge of the planted
    violation and no labels. On normal-fitting states the actor is prompted twice:
    with a system suffix asking it to follow every policy rule, and with one asking
    it to maximise throughput and ignore the rules. Per site, the direction is the
    mean activation difference (disregard minus comply) in projected coordinates.
    The score is the sum over sites of the standardised projection of a root's
    activation onto that direction. The offline stand-in actor never saw such
    instructions during training, so on it this probe is a plumbing check only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from actmon.data.features import FeatureSet
from actmon.detectors.base import Access, Detector
from actmon.detectors.projection import SitePCA


def dev_halves(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed split of the development split into a training half and a validation half."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


class _LogisticOnFeatures(Detector):
    supervised = True
    Cs = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

    def _X(self, fs: FeatureSet) -> np.ndarray:
        raise NotImplementedError

    def _prep(self, fit: FeatureSet):
        pass

    def fit(self, fit, dev=None, dev_labels=None):
        if dev is None or dev_labels is None:
            raise ValueError(f"{self.name} is supervised and needs development labels")
        self._prep(fit)
        X = self._X(dev)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        X = (X - self.mu) / self.sd
        y = np.asarray(dev_labels)
        tr, va = dev_halves(len(y), self.seed)
        best = (-1.0, None)
        for C in self.Cs:
            m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
            auc = roc_auc_score(y[va], m.decision_function(X[va])) if len(set(y[va])) > 1 else 0.5
            if auc > best[0]:
                best = (auc, C)
        self.C, self.val_auc = best[1], best[0]
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit(X[tr], y[tr])
        return self

    def score(self, fs):
        return self.model.decision_function((self._X(fs) - self.mu) / self.sd)


class LinearProbe(_LogisticOnFeatures):
    name = "linear_probe"
    access = Access("sites", "dev violation labels", 0, notes="fit on dev half A, C tuned on dev half B")

    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim, self.seed = dim, seed

    def _prep(self, fit):
        self.proj = SitePCA(self.dim).fit({s: fit.base[s] for s in fit.sites})

    def _X(self, fs):
        return self.proj.concat({s: fs.base[s] for s in fs.sites})


class ContrastProbe(Detector):
    name = "contrast_probe"
    access = Access("sites", "contrast instructions (no labels)", 0,
                    notes="2 extra passes per fitting root only; none at test time")

    def __init__(self, dim: int = 32):
        self.dim = dim

    def fit(self, fit, dev=None, dev_labels=None):
        keys = [k for k in fit.extra if k.startswith("contrast_comply@")]
        if not keys:
            raise ValueError("fit split has no contrast-pair activations")
        self.proj = SitePCA(self.dim).fit({s: fit.base[s] for s in fit.sites})
        self.u, self.stats = {}, {}
        for s in fit.sites:
            zc = self.proj.transform({s: fit.extra[f"contrast_comply@{s}"]})[s]
            zd = self.proj.transform({s: fit.extra[f"contrast_disregard@{s}"]})[s]
            u = (zd - zc).mean(0)
            self.u[s] = u / (np.linalg.norm(u) + 1e-12)
            proj = self.proj.transform({s: fit.base[s]})[s] @ self.u[s]
            self.stats[s] = (proj.mean(), proj.std() + 1e-8)
        return self

    def score(self, fs):
        z = self.proj.transform({s: fs.base[s] for s in fs.sites})
        return np.sum([(z[s] @ self.u[s] - self.stats[s][0]) / self.stats[s][1] for s in fs.sites], axis=0)
