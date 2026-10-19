"""Intervention-response signatures (plan p.6).

    response(layer, edit, x) = P_l( (I - B_l B_l^T) (h(layer, edit(x)) - h(layer, x)) )
    signature(x)             = [T responses, U responses, action-score changes]
                               (+ optional projected base snapshot)

``P_l`` is a training-only projection (PCA by default, random as a baseline)
and ``B_l`` the nuisance subspace estimated from T responses on normal data.
Both are fit on normal-fit roots only.

Column layout (fixed at fit time and recorded in ``blocks_``):
    for each enabled edit (bundle order, T before U as stored), for each layer:
        k projected response columns                              kind T / U
    for each enabled edit:
        K candidate-set log-prob changes (``raw``) and/or 3 slot-permutation-
        invariant summaries (``summary``): change in log-prob of the action
        chosen on the root, Jensen-Shannon divergence between the two
        candidate-set distributions, and a decision-flip indicator      kind A
    optionally, for each layer: k projected base-snapshot columns  kind S

Invalid edits yield NaN columns; the detector decides how to impute them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irmon.features.bundle import ResponseBundle

from .projection import LayerProjection, NuisanceSubspace


@dataclass(frozen=True)
class SignatureBlock:
    kind: str  # "T" | "U" | "A" (action-score change) | "S" (base snapshot)
    layer: int | None  # model layer index for activation blocks
    edit: str | None  # edit-slot name for response blocks
    start: int
    stop: int

    @property
    def cols(self) -> np.ndarray:
        return np.arange(self.start, self.stop)


class SignatureBuilder:
    def __init__(
        self,
        width: int = 32,
        projection: str = "pca",
        projection_source: str = "responses",
        remove_nuisance: bool = True,
        nuisance_rank: int = 8,
        use_t: bool = True,
        use_u: bool = True,
        use_activations: bool = True,
        use_action_scores: bool = True,
        include_base_snapshot: bool = False,
        seed: int = 0,
        action_features: str = "both",
    ):
        if action_features not in ("raw", "summary", "both"):
            raise ValueError(f"unknown action_features {action_features!r}")
        self.action_features = action_features
        if projection_source not in ("responses", "activations"):
            raise ValueError(f"unknown projection_source {projection_source!r}")
        self.width = width
        self.projection = projection
        self.projection_source = projection_source
        self.remove_nuisance = remove_nuisance
        self.nuisance_rank = nuisance_rank
        self.use_t = use_t
        self.use_u = use_u
        self.use_activations = use_activations
        self.use_action_scores = use_action_scores
        self.include_base_snapshot = include_base_snapshot
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def _edit_idx(self, bundle: ResponseBundle) -> np.ndarray:
        kinds = []
        if self.use_t:
            kinds.append("T")
        if self.use_u:
            kinds.append("U")
        return np.array([i for i, k in enumerate(bundle.edit_kinds) if k in kinds], dtype=int)

    def fit(self, bundle: ResponseBundle) -> SignatureBuilder:
        self.edit_names_ = bundle.edit_names
        self.layers_ = bundle.layers
        self.edit_idx_ = self._edit_idx(bundle)
        need_acts = self.use_activations and (len(self.edit_idx_) > 0 or self.include_base_snapshot)
        need_actions = self.use_action_scores and len(self.edit_idx_) > 0
        if not (need_acts or need_actions):
            raise ValueError("signature would be empty: enable T/U responses, activations or action scores")

        self.nuisance_: NuisanceSubspace | None = None
        if self.remove_nuisance and need_acts:
            t_idx = bundle.edit_index("T")
            if len(t_idx) == 0:
                raise ValueError("remove_nuisance requires T edits in the bundle")
            self.nuisance_ = NuisanceSubspace(self.nuisance_rank).fit(bundle.responses("T"))

        self.projection_: LayerProjection | None = None
        self.snapshot_projection_: LayerProjection | None = None
        if need_acts and len(self.edit_idx_) > 0:
            if self.projection_source == "responses":
                src = self._clean(bundle.responses()[:, self.edit_idx_])
            else:
                src = self._clean(bundle.h_base.astype(np.float64))
            self.projection_ = LayerProjection(self.projection, self.width, self.seed).fit(src)
        if need_acts and self.include_base_snapshot:
            self.snapshot_mean_ = bundle.h_base.astype(np.float64).mean(axis=0)
            snap = self._clean(bundle.h_base.astype(np.float64) - self.snapshot_mean_)
            self.snapshot_projection_ = LayerProjection(self.projection, self.width, self.seed + 1).fit(snap)

        self._need_acts, self._need_actions = need_acts, need_actions
        self.blocks_ = self._layout(bundle)
        self.n_features_ = self.blocks_[-1].stop
        return self

    def _clean(self, X: np.ndarray) -> np.ndarray:
        return self.nuisance_.remove(X) if self.nuisance_ is not None else X

    def _layout(self, bundle: ResponseBundle) -> list[SignatureBlock]:
        blocks: list[SignatureBlock] = []
        pos = 0
        if self.projection_ is not None:
            k = self.projection_.out_width
            for e in self.edit_idx_:
                for layer in bundle.layers:
                    blocks.append(SignatureBlock(bundle.edit_kinds[e], layer, bundle.edit_names[e], pos, pos + k))
                    pos += k
        if self._need_actions:
            K = self._n_action_cols(len(bundle.slot_names))
            for e in self.edit_idx_:
                blocks.append(SignatureBlock("A", None, bundle.edit_names[e], pos, pos + K))
                pos += K
        if self.snapshot_projection_ is not None:
            k = self.snapshot_projection_.out_width
            for layer in bundle.layers:
                blocks.append(SignatureBlock("S", layer, None, pos, pos + k))
                pos += k
        return blocks

    # ------------------------------------------------------------ transform
    def transform(self, bundle: ResponseBundle) -> np.ndarray:
        if bundle.edit_names != self.edit_names_ or bundle.layers != self.layers_:
            raise ValueError("bundle edit slots / layers differ from those seen at fit time")
        n = len(bundle)
        parts: list[np.ndarray] = []
        if self.projection_ is not None:
            resp = self._clean(bundle.responses()[:, self.edit_idx_])  # (N, E', L, D)
            proj = self.projection_.transform(resp)  # (N, E', L, k)
            parts.append(proj.reshape(n, -1))
        if self._need_actions:
            parts.append(self._action_block(bundle))
        if self.snapshot_projection_ is not None:
            snap = self._clean(bundle.h_base.astype(np.float64) - self.snapshot_mean_)
            parts.append(self.snapshot_projection_.transform(snap).reshape(n, -1))
        S = np.concatenate(parts, axis=1)
        assert S.shape[1] == self.n_features_
        return S

    def _n_action_cols(self, K: int) -> int:
        return {"raw": K, "summary": 3, "both": K + 3}[self.action_features]

    def _action_block(self, bundle: ResponseBundle) -> np.ndarray:
        """Per edit: raw slot-aligned log-prob changes and/or invariant summaries.

        Raw changes are slot-specific: for a normal actor the *direction* of a
        U-induced change is a permutation that depends non-linearly on the rule
        logic, which a linear predictor cannot track. The summaries are
        invariant to which slots swap and capture *whether and how much* the
        decision moved, which is what the U contract predicts.
        """
        n = len(bundle)
        idx = self.edit_idx_
        out = []
        raw = bundle.action_responses()[:, idx].astype(np.float64)  # (N, E', K), NaN if invalid
        for j, e in enumerate(idx):
            cols = []
            if self.action_features in ("raw", "both"):
                cols.append(raw[:, j])
            if self.action_features in ("summary", "both"):
                a0 = bundle.a_base.astype(np.float64)
                a1 = bundle.a_edit[:, e].astype(np.float64)
                p0, p1 = np.exp(a0), np.exp(a1)
                m = 0.5 * (p0 + p1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    js = 0.5 * np.nansum(p0 * (a0 - np.log(m)), axis=1) + 0.5 * np.nansum(p1 * (a1 - np.log(m)), axis=1)
                ch = bundle.chosen_slot
                d_chosen = a1[np.arange(n), ch] - a0[np.arange(n), ch]
                flip = (a1.argmax(1) != ch).astype(np.float64)
                summ = np.stack([d_chosen, js, flip], axis=1)
                summ[~bundle.edit_valid[:, e]] = np.nan
                cols.append(summ)
            out.append(np.concatenate(cols, axis=1))
        return np.concatenate(out, axis=1)

    def fit_transform(self, bundle: ResponseBundle) -> np.ndarray:
        return self.fit(bundle).transform(bundle)

    # --------------------------------------------------------------- groups
    def groups(self, by: str = "layer") -> dict[str, np.ndarray]:
        """Column groups for block-wise aggregation.

        ``layer``: one group per model layer (all response + snapshot blocks of
        that layer) plus one ``actions`` group for action-score changes.
        ``kind``: one group per block kind (T, U, A, S).
        """
        out: dict[str, list[np.ndarray]] = {}
        for b in self.blocks_:
            if by == "layer":
                key = "actions" if b.kind == "A" else f"layer{b.layer}"
            elif by == "kind":
                key = b.kind
            else:
                raise ValueError(by)
            out.setdefault(key, []).append(b.cols)
        return {k: np.concatenate(v) for k, v in out.items()}
