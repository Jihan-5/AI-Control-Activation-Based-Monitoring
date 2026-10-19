"""ResponseBundle: the per-root feature store every detector consumes.

One row per *root*. Edited inputs never become rows of their own; their
activations and action scores live inside the root's row (so every transformed
variant necessarily shares the root's split).

Shapes (N roots, L layers, D model width, E edit slots, K candidate slots,
C context covariates):

    h_base      (N, L, D)     residual stream at the final non-padding prompt token
    h_edit      (N, E, L, D)  same, for each edited input
    a_base      (N, K)        candidate-set log-probabilities (log-softmax over the
                              bounded candidate set, NOT the unrestricted policy)
    a_edit      (N, E, K)
    edit_valid  (N, E)        False where an edit failed validation (masked)
    context     (N, C)        legitimately observable task covariates
    chosen_slot (N,)          actor's proposed slot index (observable output)

Labels are *not* stored here. They live in a separate label table keyed by
root_id (plan p.4: keep label files separate from activation files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

_ARRAY_FIELDS = (
    "root_ids",
    "h_base",
    "h_edit",
    "a_base",
    "a_edit",
    "edit_valid",
    "context",
    "context_group",
    "env_label",
    "chosen_slot",
)
_META_FIELDS = ("layers", "slot_names", "edit_names", "edit_kinds", "context_names", "cost", "meta")


@dataclass
class ResponseBundle:
    root_ids: np.ndarray
    layers: tuple[int, ...]
    slot_names: tuple[str, ...]
    edit_names: tuple[str, ...]
    edit_kinds: tuple[str, ...]
    h_base: np.ndarray
    h_edit: np.ndarray
    a_base: np.ndarray
    a_edit: np.ndarray
    edit_valid: np.ndarray
    context: np.ndarray
    context_names: tuple[str, ...]
    context_group: np.ndarray
    env_label: np.ndarray
    chosen_slot: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    # Forward-equivalent cost per root used to build each feature family, e.g.
    # {"base": 1, "edits": 6, "candidates": 7, "loss_trace": 16}.
    cost: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ invariants
    def __post_init__(self) -> None:
        n = len(self.root_ids)
        L, E, K = len(self.layers), len(self.edit_names), len(self.slot_names)
        if self.h_base.shape[:2] != (n, L):
            raise ValueError(f"h_base shape {self.h_base.shape} != ({n}, {L}, D)")
        if self.h_edit.shape[:3] != (n, E, L):
            raise ValueError(f"h_edit shape {self.h_edit.shape} != ({n}, {E}, {L}, D)")
        if self.h_edit.shape[3] != self.h_base.shape[2]:
            raise ValueError("h_edit and h_base disagree on model width")
        if self.a_base.shape != (n, K) or self.a_edit.shape != (n, E, K):
            raise ValueError("action-score shapes inconsistent")
        if self.edit_valid.shape != (n, E):
            raise ValueError("edit_valid shape inconsistent")
        if self.context.shape != (n, len(self.context_names)):
            raise ValueError("context shape inconsistent")
        if len(self.edit_kinds) != E or not set(self.edit_kinds) <= {"T", "U"}:
            raise ValueError("edit_kinds must be 'T'/'U' per edit slot")
        for k, v in self.extras.items():
            if len(v) != n:
                raise ValueError(f"extra {k!r} has {len(v)} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.root_ids)

    @property
    def d_model(self) -> int:
        return int(self.h_base.shape[2])

    def edit_index(self, kind: str | None = None) -> np.ndarray:
        if kind is None:
            return np.arange(len(self.edit_names))
        return np.array([i for i, k in enumerate(self.edit_kinds) if k == kind], dtype=int)

    # ------------------------------------------------------------- features
    def responses(self, kind: str | None = None) -> np.ndarray:
        """Raw activation responses h(edit(x)) - h(x), shape (N, E', L, D).
        Invalid edits are NaN."""
        idx = self.edit_index(kind)
        r = self.h_edit[:, idx].astype(np.float32) - self.h_base[:, None].astype(np.float32)
        r[~self.edit_valid[:, idx]] = np.nan
        return r

    def action_responses(self, kind: str | None = None) -> np.ndarray:
        """Candidate-set log-prob changes, shape (N, E', K). Invalid edits NaN."""
        idx = self.edit_index(kind)
        r = self.a_edit[:, idx].astype(np.float32) - self.a_base[:, None].astype(np.float32)
        r[~self.edit_valid[:, idx]] = np.nan
        return r

    def edit_decisions(self) -> np.ndarray:
        """Actor's argmax slot under each edit, shape (N, E)."""
        return np.argmax(self.a_edit, axis=2)

    # ------------------------------------------------------------- slicing
    def subset(self, idx) -> ResponseBundle:
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in _ARRAY_FIELDS:
                kw[f.name] = v[idx]
            elif f.name == "extras":
                kw[f.name] = {k: a[idx] for k, a in v.items()}
            else:
                kw[f.name] = v
        return ResponseBundle(**kw)

    def select_roots(self, root_ids) -> ResponseBundle:
        pos = {r: i for i, r in enumerate(self.root_ids.tolist())}
        missing = [r for r in root_ids if r not in pos]
        if missing:
            raise KeyError(f"{len(missing)} root ids not in bundle, e.g. {missing[:3]}")
        return self.subset(np.array([pos[r] for r in root_ids], dtype=int))

    @staticmethod
    def concat(bundles: list[ResponseBundle]) -> ResponseBundle:
        b0 = bundles[0]
        for b in bundles[1:]:
            for m in ("layers", "slot_names", "edit_names", "edit_kinds", "context_names"):
                if getattr(b, m) != getattr(b0, m):
                    raise ValueError(f"cannot concat bundles with different {m}")
        kw = {f: np.concatenate([getattr(b, f) for b in bundles]) for f in _ARRAY_FIELDS}
        extras = {k: np.concatenate([b.extras[k] for b in bundles]) for k in b0.extras if all(k in b.extras for b in bundles)}
        return ResponseBundle(
            layers=b0.layers, slot_names=b0.slot_names, edit_names=b0.edit_names,
            edit_kinds=b0.edit_kinds, context_names=b0.context_names, extras=extras,
            cost=dict(b0.cost), meta=dict(b0.meta), **kw,
        )

    # -------------------------------------------------------------- storage
    def save(self, path: str | Path, store_dtype: str = "float16") -> None:
        """Save to ``path/`` as npz arrays + json metadata.

        Activations are stored at ``store_dtype`` (default float16) to bound
        disk use: N x (1+E) x L x D x 2 bytes. Precision must stay consistent
        within any comparison (plan p.3); it is recorded in meta.json.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for f in _ARRAY_FIELDS:
            v = getattr(self, f)
            if f in ("h_base", "h_edit"):
                v = v.astype(store_dtype)
            elif v.dtype.kind in "US":
                v = v.astype(str)
            arrays[f] = v
        for k, v in self.extras.items():
            arrays[f"extra__{k}"] = v
        np.savez(path / "arrays.npz", **arrays)
        meta = {f: getattr(self, f) for f in _META_FIELDS}
        meta["store_dtype"] = store_dtype
        meta["layers"] = list(self.layers)
        (path / "meta.json").write_text(json.dumps(meta, indent=1, default=list))

    @classmethod
    def load(cls, path: str | Path) -> ResponseBundle:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        with np.load(path / "arrays.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        extras = {k[len("extra__"):]: arrays.pop(k) for k in list(arrays) if k.startswith("extra__")}
        for f in ("h_base", "h_edit"):
            arrays[f] = arrays[f].astype(np.float32)
        return cls(
            layers=tuple(meta["layers"]),
            slot_names=tuple(meta["slot_names"]),
            edit_names=tuple(meta["edit_names"]),
            edit_kinds=tuple(meta["edit_kinds"]),
            context_names=tuple(meta["context_names"]),
            cost=dict(meta.get("cost", {})),
            meta=dict(meta.get("meta", {})),
            extras=extras,
            **arrays,
        )
