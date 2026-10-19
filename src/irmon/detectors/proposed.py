"""The proposed intervention-response detector (plan pp.6-7).

Pipeline (every statistic fit on normal-fit roots only):

1. Signature: s(x) = [P_l (I - B_l B_l^T)(h(edit(x)) - h(x)) for T and U edits,
   candidate-set log-prob changes]  (``irmon.features.signature``).
2. Training-only scaling of signature columns and of context covariates. Scaling
   matters because shrinkage pulls the covariance toward a scaled identity,
   which is only sensible on comparable column scales.
3. Normal-response predictor: ridge regression of the scaled signature on
   legitimately observable context (optionally a 1x64 MLP, used only if it
   improves development generalisation; excess capacity may explain away the
   failure signal). Ridge is fit per *validity group* (columns from the same
   edit share a validity mask), using only rows where that edit is valid.
4. Residual r(x) = s(x) - f(context(x)). Invalid-edit entries are imputed as 0
   after prediction, i.e. "no evidence" for that block (assumption A-DET-02).
5. Covariance of residuals from K-fold CROSS-FITTED predictions. In-sample
   residuals are optimistically small (the predictor has seen those rows), so a
   covariance fit on them under-estimates normal residual spread and inflates
   every new score. Cross-fitting gives residuals with the same out-of-sample
   distribution the detector sees at calibration/test time.
6. score(x) = r^T Σ^{-1} r with a shrinkage Σ inverted through an eigenvalue-
   floored eigendecomposition. Aggregation ``concat`` uses one Mahalanobis over
   all columns; ``max`` takes the maximum of per-layer block Mahalanobis
   scores, each standardised by its cross-fitted normal-fit median/IQR. The
   combined score is calibrated once, downstream.
"""

from __future__ import annotations

import numpy as np

from irmon.config import ProposedConfig
from irmon.features.bundle import ResponseBundle
from irmon.features.signature import SignatureBuilder

from .base import AccessSpec, Detector
from .linalg import Standardizer, Whitener, impute_nan, nanmean0, robust_standardize_fit


# ---------------------------------------------------------------- predictor
class NormalResponsePredictor:
    """Predict the (scaled) normal signature from scaled context.

    ``groups``: list of column-index arrays whose rows share a validity mask.
    ``kind``: ``ridge`` | ``mlp`` | ``none`` (training mean only).
    """

    def __init__(
        self,
        kind: str = "ridge",
        alpha: float = 1.0,
        hidden: int = 64,
        lr: float = 1e-3,
        seed: int = 0,
        groups: list[np.ndarray] | None = None,
        max_epochs: int = 400,
        patience: int = 20,
        feature_groups: list[np.ndarray] | None = None,
    ):
        if kind not in ("ridge", "mlp", "none"):
            raise ValueError(f"unknown predictor {kind!r}")
        self.kind = kind
        self.alpha = alpha
        self.hidden = hidden
        self.lr = lr
        self.seed = seed
        self.groups = groups
        self.max_epochs = max_epochs
        self.patience = patience
        # Optional per-group input columns (ridge only): the block of edit e is
        # predicted from base context + edit e's context change, not from every
        # edit's change (fewer nuisance inputs per regression).
        self.feature_groups = feature_groups

    def fit(self, X: np.ndarray, Y: np.ndarray) -> NormalResponsePredictor:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        p = Y.shape[1]
        groups = self.groups if self.groups is not None else [np.arange(p)]
        self.p_ = p
        self.col_mean_ = nanmean0(Y) if len(Y) else np.zeros(p)
        if self.kind == "none" or X.shape[1] == 0:
            self.fitted_kind_ = "none"
            return self
        if self.kind == "ridge":
            self.fitted_kind_ = "ridge"
            self.coef_ = np.zeros((X.shape[1], p))
            self.intercept_ = self.col_mean_.copy()
            feats = self.feature_groups or [np.arange(X.shape[1])] * len(groups)
            for cols, fcols in zip(groups, feats):
                rows = np.isfinite(Y[:, cols]).all(axis=1)
                if rows.sum() < 2:
                    continue
                Xg, Yg = X[rows][:, fcols], Y[rows][:, cols]
                xm, ym = Xg.mean(axis=0), Yg.mean(axis=0)
                Xc, Yc = Xg - xm, Yg - ym
                A = Xc.T @ Xc + self.alpha * np.eye(len(fcols))
                W = np.linalg.solve(A, Xc.T @ Yc)
                self.coef_[np.ix_(fcols, cols)] = W
                self.intercept_[cols] = ym - xm @ W
            return self
        self.fitted_kind_ = "mlp"
        self._fit_mlp(X, Y)
        return self

    # Small masked-loss MLP: weights are kept as numpy arrays so the detector
    # stays picklable (CalibrationLock hashes the pickled detector).
    def _fit_mlp(self, X: np.ndarray, Y: np.ndarray) -> None:
        mask = np.isfinite(Y)
        Y0 = np.where(mask, Y, 0.0)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        val, tr = perm[:n_val], perm[n_val:]
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is optional
            from sklearn.neural_network import MLPRegressor

            reg = MLPRegressor(
                hidden_layer_sizes=(self.hidden,), learning_rate_init=self.lr, early_stopping=True,
                random_state=self.seed, max_iter=self.max_epochs,
            ).fit(X, np.where(mask, Y, self.col_mean_))
            self.w1_, self.b1_ = reg.coefs_[0], reg.intercepts_[0]
            self.w2_, self.b2_ = reg.coefs_[1], reg.intercepts_[1]
            return
        torch.manual_seed(self.seed)
        d_in, d_out = X.shape[1], Y.shape[1]
        net = torch.nn.Sequential(
            torch.nn.Linear(d_in, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, d_out)
        ).double()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr)
        Xt, Yt, Mt = (torch.as_tensor(a) for a in (X, Y0, mask.astype(np.float64)))

        def loss_on(idx):
            out = net(Xt[idx])
            m = Mt[idx]
            return ((out - Yt[idx]) ** 2 * m).sum() / m.sum().clamp_min(1.0)

        best, best_state, bad = np.inf, None, 0
        tr_t, val_t = torch.as_tensor(tr), torch.as_tensor(val)
        for _ in range(self.max_epochs):
            net.train()
            opt.zero_grad()
            loss = loss_on(tr_t)
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_on(val_t))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = [p.detach().clone() for p in net.parameters()]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        with torch.no_grad():
            for p, b in zip(net.parameters(), best_state):
                p.copy_(b)
        self.w1_ = net[0].weight.detach().numpy().T.copy()
        self.b1_ = net[0].bias.detach().numpy().copy()
        self.w2_ = net[2].weight.detach().numpy().T.copy()
        self.b2_ = net[2].bias.detach().numpy().copy()
        self.val_loss_ = best

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.fitted_kind_ == "none":
            return np.broadcast_to(self.col_mean_, (len(X), self.p_)).copy()
        if self.fitted_kind_ == "ridge":
            return X @ self.coef_ + self.intercept_
        h = np.maximum(X @ self.w1_ + self.b1_, 0.0)
        return h @ self.w2_ + self.b2_


# ----------------------------------------------------------------- detector
class InterventionResponseDetector(Detector):
    name = "proposed"

    def __init__(self, config: ProposedConfig | None = None, **overrides):
        super().__init__()
        cfg = config or ProposedConfig()
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        self.config = cfg
        self.access = AccessSpec(
            activations=cfg.use_activations,
            action_scores=cfg.use_action_scores,
            forwards_per_root=1.0,  # updated at fit time to 1 + number of edits used
            notes="normal-fit only; extra forwards for every edited input plus candidate scoring",
        )

    def describe(self) -> dict:
        d = super().describe()
        d["params"] = self.config.model_dump()
        return d

    # ------------------------------------------------------------- helpers
    def _builder(self) -> SignatureBuilder:
        c = self.config
        return SignatureBuilder(
            width=c.width,
            projection=c.projection,
            projection_source=c.projection_source,
            remove_nuisance=c.remove_nuisance,
            nuisance_rank=c.nuisance_rank,
            use_t=c.use_t,
            use_u=c.use_u,
            use_activations=c.use_activations,
            use_action_scores=c.use_action_scores,
            include_base_snapshot=c.include_base_snapshot,
            seed=c.seed,
            action_features=c.action_features,
        )

    def _edit_groups(self) -> dict[str, list[np.ndarray]]:
        by_edit: dict[str, list[np.ndarray]] = {}
        for b in self.builder_.blocks_:
            by_edit.setdefault(b.edit or "__snapshot__", []).append(b.cols)
        return by_edit

    def _validity_groups(self) -> list[np.ndarray]:
        return [np.concatenate(v) for v in self._edit_groups().values()]

    def _feature_groups(self) -> list[np.ndarray] | None:
        """Per validity group: base-context columns + that edit's delta columns."""
        if not self.use_edit_context_:
            return None
        C = self.n_ctx_
        out = []
        for edit in self._edit_groups():
            cols = list(range(C))
            if edit in self.edit_names_:
                e = self.edit_names_.index(edit)
                cols += list(range(C + e * C, C + (e + 1) * C))
            out.append(np.array(cols, dtype=int))
        return out

    def _new_predictor(self) -> NormalResponsePredictor:
        c = self.config
        kind = c.predictor if c.use_context else "none"
        return NormalResponsePredictor(
            kind=kind, alpha=c.ridge_alpha, hidden=c.mlp_hidden, lr=c.mlp_lr, seed=c.seed,
            groups=self._validity_groups(), feature_groups=self._feature_groups() if kind == "ridge" else None,
        )

    @staticmethod
    def _edit_delta(bundle: ResponseBundle) -> np.ndarray:
        ce = bundle.extras["context_edit"].astype(np.float64)
        return (ce - bundle.context[:, None, :].astype(np.float64)).reshape(len(bundle), -1)

    def _context(self, bundle: ResponseBundle) -> np.ndarray:
        if not self.config.use_context:
            return np.zeros((len(bundle), 0))
        X = self.ctx_scaler_.transform(bundle.context)
        if self.use_edit_context_:
            X = np.hstack([X, self.delta_scaler_.transform(self._edit_delta(bundle))])
        return X

    def _folds(self, n: int) -> list[np.ndarray]:
        k = int(min(max(2, self.config.cross_fit_folds), n // 2))
        rng = np.random.default_rng(self.config.seed)
        return np.array_split(rng.permutation(n), k)

    def _whiten_fit(self, R: np.ndarray) -> Whitener:
        c = self.config
        return Whitener().fit(R, method=c.covariance, shrinkage=c.fixed_shrinkage)

    # ----------------------------------------------------------------- fit
    def fit(self, bundle: ResponseBundle, labels: np.ndarray | None = None) -> InterventionResponseDetector:
        # Unsupervised: labels are ignored by design (normal-fit roots only).
        c = self.config
        self.builder_ = self._builder().fit(bundle)
        S = self.builder_.transform(bundle)
        self.sig_scaler_ = Standardizer().fit(S)
        Z = self.sig_scaler_.transform(S)
        self.ctx_scaler_ = Standardizer().fit(bundle.context)
        self.n_ctx_ = bundle.context.shape[1]
        self.edit_names_ = list(bundle.edit_names)
        self.use_edit_context_ = bool(c.use_context and c.edit_context and "context_edit" in bundle.extras)
        if c.use_context and c.edit_context and not self.use_edit_context_:
            import warnings

            warnings.warn("edit_context requested but bundle has no 'context_edit' extra; using base context only",
                          stacklevel=2)
        if self.use_edit_context_:
            self.delta_scaler_ = Standardizer().fit(self._edit_delta(bundle))
        X = self._context(bundle)
        n = len(bundle)

        # Cross-fitted residuals (see module docstring, step 5).
        R_cf = np.full_like(Z, np.nan)
        folds = self._folds(n)
        for held in folds:
            train = np.setdiff1d(np.arange(n), held)
            pred = self._new_predictor().fit(X[train], Z[train])
            R_cf[held] = Z[held] - pred.predict(X[held])
        R_cf = impute_nan(R_cf, 0.0)

        self.predictor_ = self._new_predictor().fit(X, Z)
        R_in = impute_nan(Z - self.predictor_.predict(X), 0.0)
        self.residual_trace_insample_ = float(np.var(R_in, axis=0).sum())
        self.residual_trace_crossfit_ = float(np.var(R_cf, axis=0).sum())

        if c.aggregation == "concat":
            self.whitener_ = self._whiten_fit(R_cf)
        elif c.aggregation == "max":
            self.groups_ = self.builder_.groups("layer")
            self.block_whiteners_ = {g: self._whiten_fit(R_cf[:, cols]) for g, cols in self.groups_.items()}
            # Normal-fit reference scale per block from cross-fitted block scores.
            self.block_ref_ = {}
            for g, cols in self.groups_.items():
                ref = np.empty(n)
                for held in folds:
                    train = np.setdiff1d(np.arange(n), held)
                    w = self._whiten_fit(R_cf[train][:, cols])
                    ref[held] = w.score(R_cf[held][:, cols])
                self.block_ref_[g] = robust_standardize_fit(ref)
        else:
            raise ValueError(f"unknown aggregation {c.aggregation!r}")

        n_edits = len(self.builder_.edit_idx_)
        self.access = AccessSpec(
            activations=c.use_activations,
            action_scores=c.use_action_scores,
            forwards_per_root=float(1 + n_edits),
            notes=f"normal-fit only; {n_edits} edited inputs per root plus candidate scoring",
        )
        self.n_features_ = int(Z.shape[1])
        self.fitted_ = True
        return self

    # --------------------------------------------------------------- score
    def residuals(self, bundle: ResponseBundle) -> np.ndarray:
        self._check_fitted()
        Z = self.sig_scaler_.transform(self.builder_.transform(bundle))
        R = Z - self.predictor_.predict(self._context(bundle))
        return impute_nan(R, 0.0)

    def score(self, bundle: ResponseBundle) -> np.ndarray:
        R = self.residuals(bundle)
        if self.config.aggregation == "concat":
            return self.whitener_.score(R)
        block = []
        for g, cols in self.groups_.items():
            med, scale = self.block_ref_[g]
            block.append((self.block_whiteners_[g].score(R[:, cols]) - med) / scale)
        return np.max(np.stack(block, axis=1), axis=1)

    def block_scores(self, bundle: ResponseBundle) -> dict[str, np.ndarray]:
        """Per-kind Mahalanobis of residual blocks (diagnostics / failure slices)."""
        R = self.residuals(bundle)
        out = {}
        for kind, cols in self.builder_.groups("kind").items():
            out[kind] = (R[:, cols] ** 2).sum(axis=1)
        return out
