"""PCIR-style invariant encoder (syllabus S13; Carvalho et al., 2023, adapted).

Partial Conditional Invariant Regularization matches normal representations
across observed environments with an MMD penalty. Adaptation used here:

* Input: standardized, training-only-PCA-reduced h_base (layers concatenated)
  of normal-fit roots.
* Model: MLP autoencoder (encoder d -> hidden -> latent, mirrored decoder).
* Penalty: RBF-kernel MMD^2 (median-heuristic bandwidth) between the latent
  codes of each pair of observable environments (``env_label``, i.e. the
  scenario family visible in the prompt), computed *within* each observable
  ``context_group`` - partial conditional invariance: representations should
  not depend on the environment given the legitimate task configuration.
* Score: reconstruction error + latent Mahalanobis, with fixed unit weights
  after normal-fit standardisation of each term.

This tests whether standard invariance learning already suffices. It is an
adaptation (different data modality, our environment definition, our score),
not a replication. Matching normal representations across training
environments does not constrain unseen anomalous inputs (the PCIR paper's own
stated limitation).
"""

from __future__ import annotations

import numpy as np

from irmon.features.bundle import ResponseBundle
from irmon.features.projection import LayerProjection

from .base import AccessSpec, Detector
from .linalg import Standardizer, Whitener


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch optional
        raise ImportError("PCIRInvariantEncoder needs torch: pip install 'irmon[model]'") from exc
    return torch


class PCIRInvariantEncoder(Detector):
    name = "pcir"
    access = AccessSpec(
        activations=True, forwards_per_root=1.0,
        notes="single base forward; uses observable environment labels; normal-fit only; adaptation",
    )

    def __init__(
        self,
        pca_dim: int = 32,
        hidden: int = 64,
        latent: int = 16,
        mmd_weight: float = 1.0,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 128,
        min_env_rows: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.hidden = hidden
        self.latent = latent
        self.mmd_weight = mmd_weight
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_env_rows = min_env_rows
        self.seed = seed

    def _inputs(self, bundle: ResponseBundle) -> np.ndarray:
        F = self.proj_.transform(bundle.h_base.astype(np.float64))
        return self.scaler_.transform(F.reshape(len(F), -1))

    @staticmethod
    def _mmd2(torch, a, b):
        x = torch.cat([a, b], 0)
        d2 = torch.cdist(x, x) ** 2
        bw = torch.median(d2[d2 > 0]).detach() if (d2 > 0).any() else torch.tensor(1.0, dtype=x.dtype)
        k = torch.exp(-d2 / bw.clamp_min(1e-6))
        n = len(a)
        return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()

    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> PCIRInvariantEncoder:
        torch = _require_torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.proj_ = LayerProjection("pca", self.pca_dim, self.seed).fit(bundle.h_base)
        F = self.proj_.transform(bundle.h_base.astype(np.float64))
        self.scaler_ = Standardizer().fit(F.reshape(len(F), -1))
        X = self._inputs(bundle)
        d = X.shape[1]
        self.enc_ = torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, self.latent)
        ).double()
        self.dec_ = torch.nn.Sequential(
            torch.nn.Linear(self.latent, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, d)
        ).double()
        params = list(self.enc_.parameters()) + list(self.dec_.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr)
        Xt = torch.as_tensor(X)
        envs = np.asarray(bundle.env_label).astype(str)
        groups = np.asarray(bundle.context_group).astype(str)
        n = len(X)
        self.loss_history_: list[float] = []
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                z = self.enc_(Xt[idx])
                rec = ((self.dec_(z) - Xt[idx]) ** 2).mean()
                pen = torch.zeros((), dtype=torch.float64)
                n_terms = 0
                b_env, b_grp = envs[idx], groups[idx]
                for g in np.unique(b_grp):
                    in_g = b_grp == g
                    env_ids = [e for e in np.unique(b_env[in_g]) if (in_g & (b_env == e)).sum() >= self.min_env_rows]
                    for i in range(len(env_ids)):
                        for j in range(i + 1, len(env_ids)):
                            ia = torch.as_tensor(np.flatnonzero(in_g & (b_env == env_ids[i])))
                            ib = torch.as_tensor(np.flatnonzero(in_g & (b_env == env_ids[j])))
                            pen = pen + self._mmd2(torch, z[ia], z[ib])
                            n_terms += 1
                loss = rec + self.mmd_weight * (pen / max(n_terms, 1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.loss_history_.append(float(loss.detach()))
        self.enc_.eval()
        self.dec_.eval()
        rec, Z = self._terms(X)
        self.latent_whitener_ = Whitener().fit(Z, "ledoit_wolf")
        lat = self.latent_whitener_.score(Z)
        self.term_ref_ = (rec.mean(), rec.std() or 1.0, lat.mean(), lat.std() or 1.0)
        self.fitted_ = True
        return self

    def _terms(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = _require_torch()
        with torch.no_grad():
            Xt = torch.as_tensor(X)
            Z = self.enc_(Xt)
            rec = ((self.dec_(Z) - Xt) ** 2).mean(dim=1)
        return rec.numpy(), Z.numpy()

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        rec, Z = self._terms(self._inputs(bundle))
        rm, rs, lm, ls = self.term_ref_
        return (rec - rm) / rs + (self.latent_whitener_.score(Z) - lm) / ls
