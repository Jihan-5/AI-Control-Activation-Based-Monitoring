r"""What a monitor may see (:class:`FeatureSet`) and what only evaluation may see (:class:`RootLabels`).

A :class:`FeatureSet` holds, for every root of one split:

* ``base[site]``: residual activations at the decision token, shape ``(n, d)``
* ``variants[name][site]``: the same under each main-suite transformation
* ``base_feats`` and ``variant_feats[name]``: response/action scores from
  constrained decoding (:data:`actmon.actor.decode.FEATURE_NAMES`)
* ``base_action`` and ``variant_action[name]``: the proposed action strings. A
  deployed monitor sees the proposal it is asked to approve, so these count as
  observable
* ``context``: legitimately observable task context, parsed from the prompt's
  *surface* without the policy (see :func:`context_features`)
* ``context_class``: a discrete context key for task-conditioned baselines
* ``extra``: optional arrays for specific baselines (TeCo-style severity
  variants, functional-attribution traces), keyed by name

It contains **no** oracle output: no validity, violations, planner action or
mechanism. Those live in :class:`RootLabels`, which only
:mod:`actmon.evaluation` and the explicitly supervised baselines receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from actmon.actor.decode import FEATURE_NAMES
from actmon.sim.sampler import RESOURCES
from actmon.sim.state import Scenario

CONTEXT_NAMES = tuple(f"res_{r}" for r in RESOURCES) + ("handoff_msg", "n_pools", "n_distractors", "n_logs")


def context_features(s: Scenario) -> np.ndarray:
    """Observable task descriptors that need no policy evaluation.

    Resource type of the ticket, whether an operator handoff message is present
    (its authentication result is *not* used), and record counts. These say what
    kind of task this is, not whether any action is permitted.
    """
    v = np.zeros(len(CONTEXT_NAMES), dtype=np.float32)
    v[RESOURCES.index(s.focal.resource)] = 1.0
    v[4] = float(s.handoff.present)
    v[5] = len(s.pools)
    v[6] = len(s.distractors)
    v[7] = len(s.logs)
    return v


def context_class(s: Scenario) -> str:
    return f"{s.focal.resource}|h{int(s.handoff.present)}|p{len(s.pools)}"


@dataclass
class FeatureSet:
    split: str
    root_ids: np.ndarray
    family: np.ndarray
    sites: list[int]
    base: dict[int, np.ndarray]
    base_feats: np.ndarray
    base_action: np.ndarray
    context: np.ndarray
    context_class: np.ndarray
    variants: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    variant_feats: dict[str, np.ndarray] = field(default_factory=dict)
    variant_action: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.root_ids.shape[0])

    def subset(self, idx) -> "FeatureSet":
        idx = np.asarray(idx)
        return FeatureSet(
            split=self.split, root_ids=self.root_ids[idx], family=self.family[idx], sites=list(self.sites),
            base={s: v[idx] for s, v in self.base.items()}, base_feats=self.base_feats[idx],
            base_action=self.base_action[idx], context=self.context[idx], context_class=self.context_class[idx],
            variants={k: {s: v[idx] for s, v in d.items()} for k, d in self.variants.items()},
            variant_feats={k: v[idx] for k, v in self.variant_feats.items()},
            variant_action={k: v[idx] for k, v in self.variant_action.items()},
            extra={k: v[idx] for k, v in self.extra.items()},
        )

    @staticmethod
    def concat(sets: list["FeatureSet"], split: str) -> "FeatureSet":
        f0 = sets[0]
        cat = np.concatenate
        return FeatureSet(
            split=split, root_ids=cat([f.root_ids for f in sets]), family=cat([f.family for f in sets]),
            sites=list(f0.sites), base={s: cat([f.base[s] for f in sets]) for s in f0.sites},
            base_feats=cat([f.base_feats for f in sets]), base_action=cat([f.base_action for f in sets]),
            context=cat([f.context for f in sets]), context_class=cat([f.context_class for f in sets]),
            variants={k: {s: cat([f.variants[k][s] for f in sets]) for s in f0.sites} for k in f0.variants},
            variant_feats={k: cat([f.variant_feats[k] for f in sets]) for k in f0.variant_feats},
            variant_action={k: cat([f.variant_action[k] for f in sets]) for k in f0.variant_action},
            extra={k: cat([f.extra[k] for f in sets]) for k in f0.extra if all(k in f.extra for f in sets)},
        )

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        arrs = {"root_ids": self.root_ids.astype(str), "family": self.family.astype(str),
                "sites": np.asarray(self.sites), "base_feats": self.base_feats,
                "base_action": self.base_action.astype(str), "context": self.context,
                "context_class": self.context_class.astype(str)}
        for s, v in self.base.items():
            arrs[f"base@{s}"] = v
        for k, d in self.variants.items():
            for s, v in d.items():
                arrs[f"var@{k}@{s}"] = v
        for k, v in self.variant_feats.items():
            arrs[f"vfeat@{k}"] = v
        for k, v in self.variant_action.items():
            arrs[f"vact@{k}"] = v.astype(str)
        for k, v in self.extra.items():
            arrs[f"extra@{k}"] = v
        np.savez_compressed(path, split=np.asarray(self.split), **arrs)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        z = np.load(path, allow_pickle=False)
        sites = [int(s) for s in z["sites"]]
        fs = cls(split=str(z["split"]), root_ids=z["root_ids"], family=z["family"], sites=sites,
                 base={s: z[f"base@{s}"] for s in sites}, base_feats=z["base_feats"], base_action=z["base_action"],
                 context=z["context"], context_class=z["context_class"])
        for key in z.files:
            if key.startswith("var@"):
                _, k, s = key.split("@")
                fs.variants.setdefault(k, {})[int(s)] = z[key]
            elif key.startswith("vfeat@"):
                fs.variant_feats[key.split("@", 1)[1]] = z[key]
            elif key.startswith("vact@"):
                fs.variant_action[key.split("@", 1)[1]] = z[key]
            elif key.startswith("extra@"):
                fs.extra[key.split("@", 1)[1]] = z[key]
        return fs


@dataclass
class RootLabels:
    """Oracle-side table for one split. Index-aligned with the split's FeatureSet."""

    table: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        """1 = violating root, 0 = normal root."""
        return (~self.table["valid"].to_numpy(dtype=bool)).astype(int)

    def save(self, path: str | Path) -> None:
        self.table.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "RootLabels":
        return cls(pd.read_csv(path, keep_default_na=False))


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)
