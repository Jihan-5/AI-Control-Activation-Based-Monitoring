"""Split assignment, immutable manifests, and leakage guards.

Rules enforced here (plan pp.3-4, 7, 9):
* All transformed variants of a root stay in the root's split (edits are never
  rows of their own; they live inside the root's record).
* Clusters (outcome-matched pairs) are never split across partitions.
* Family tiers are partitioned: dev_unfamiliar families only in ``dev``,
  test_unfamiliar families only in ``test``.
* Held-out violation mechanisms never appear in ``dev``.
* The final test set is never available to model, layer, projection or
  threshold selection: ``SplitGuard`` raises if a stage touches a split it is
  not entitled to.
* The calibration split opens only after the entire scoring function is fitted
  and frozen (``CalibrationLock``).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import HELD_OUT_MECHANISMS

SPLITS: tuple[str, ...] = ("normal_fit", "dev", "normal_cal", "test", "audit", "utility")

# Which splits each pipeline stage may read. Anything else raises.
STAGE_PERMISSIONS: dict[str, frozenset[str]] = {
    "fit": frozenset({"normal_fit"}),
    "select": frozenset({"normal_fit", "dev"}),  # hyperparameters, layers, projection
    "supervised_fit": frozenset({"dev"}),  # probes / behavioral logistic (labelled)
    "calibrate": frozenset({"normal_cal"}),
    "test": frozenset({"test"}),
    "audit": frozenset({"audit"}),
    "utility": frozenset({"utility"}),
}


class LeakageError(RuntimeError):
    pass


@dataclass
class SplitQuota:
    normal: int = 0
    violation: int = 0
    tiers: tuple[str, ...] = ("familiar",)
    # Optional fraction of *normal* roots per tier (must sum to 1 if given).
    normal_tier_mix: dict[str, float] | None = None
    violation_tier_mix: dict[str, float] | None = None
    allowed_mechanisms: tuple[str, ...] | None = None  # None = any


@dataclass
class SplitPlan:
    quotas: dict[str, SplitQuota] = field(default_factory=dict)

    @classmethod
    def pilot(cls) -> SplitPlan:
        """Pilot counts from the research plan (p.4). Planning values only."""
        return cls(
            {
                "normal_fit": SplitQuota(normal=1000, tiers=("familiar",)),
                "dev": SplitQuota(
                    normal=500, violation=500, tiers=("familiar", "dev_unfamiliar"),
                    normal_tier_mix={"familiar": 0.6, "dev_unfamiliar": 0.4},
                    allowed_mechanisms=("ordinary", "prompt_conflict"),
                ),
                "normal_cal": SplitQuota(normal=2000, tiers=("familiar",)),
                "test": SplitQuota(
                    normal=2000, violation=1000, tiers=("familiar", "test_unfamiliar"),
                    normal_tier_mix={"familiar": 0.5, "test_unfamiliar": 0.5},
                    violation_tier_mix={"familiar": 0.5, "test_unfamiliar": 0.5},
                ),
            }
        )

    @classmethod
    def expanded(cls) -> SplitPlan:
        """Expanded sizes from the technical specification (p.2)."""
        p = cls.pilot()
        p.quotas["normal_cal"] = SplitQuota(normal=5000, tiers=("familiar",))
        # TS p.2 proposed 500 test violations; the power analysis
        # (scripts/plan_compute.py) shows 500 gives only 50-80% power for a
        # 5-point paired gain at 15-30% discordance, so the expanded plan keeps
        # the pilot's 1,000 (decision log 2026-10-19).
        p.quotas["test"] = SplitQuota(
            normal=5000, violation=1000, tiers=("familiar", "test_unfamiliar"),
            normal_tier_mix={"familiar": 0.5, "test_unfamiliar": 0.5},
            violation_tier_mix={"familiar": 0.5, "test_unfamiliar": 0.5},
        )
        p.quotas["audit"] = SplitQuota(normal=900, violation=100, tiers=("familiar", "test_unfamiliar"))
        p.quotas["utility"] = SplitQuota(normal=500, tiers=("familiar", "test_unfamiliar"))
        return p

    def scaled(self, factor: float, minimum: int = 4) -> SplitPlan:
        """Shrink every quota (smoke tests). Ratios and constraints preserved."""
        q = {}
        for name, s in self.quotas.items():
            q[name] = SplitQuota(
                normal=max(minimum, int(s.normal * factor)) if s.normal else 0,
                violation=max(minimum, int(s.violation * factor)) if s.violation else 0,
                tiers=s.tiers,
                normal_tier_mix=s.normal_tier_mix,
                violation_tier_mix=s.violation_tier_mix,
                allowed_mechanisms=s.allowed_mechanisms,
            )
        return SplitPlan(q)


def _take(rng: np.random.Generator, df: pd.DataFrame, n: int, used: set[str], what: str) -> list[str]:
    """Sample ``n`` clusters' roots not yet used. Raises when the pool is short."""
    avail = df[~df["cluster_id"].isin(used)]
    clusters = avail["cluster_id"].unique()
    if len(clusters) < n:
        raise ValueError(
            f"screened pool too small for {what}: need {n} clusters, have {len(clusters)}. "
            "Generate a larger screening pool (plan: increase counts rather than relax constraints)."
        )
    chosen = rng.choice(clusters, size=n, replace=False)
    used.update(chosen.tolist())
    return chosen.tolist()


def _alloc_by_mix(total: int, mix: dict[str, float] | None, tiers: tuple[str, ...]) -> dict[str, int]:
    if not mix:
        return {"*": total}
    keys = [t for t in tiers if t in mix]
    raw = np.array([mix[k] for k in keys], dtype=float)
    raw = raw / raw.sum()
    counts = np.floor(raw * total).astype(int)
    counts[: total - counts.sum()] += 1
    return dict(zip(keys, counts.tolist()))


def assign_splits(pool: pd.DataFrame, plan: SplitPlan, seed: int) -> pd.DataFrame:
    """Case-control sampling of the screened pool into splits.

    ``pool`` needs columns: root_id, cluster_id, tier, is_violation, mechanism,
    slice. Only ``slice == 'main'`` rows are assigned by quota; other slices
    (outcome-matched pairs) are attached to ``test`` whole.

    Returns ``pool`` with a ``split`` column (``unused`` for unsampled roots).
    The full screened pool is kept so the realised prevalence can be logged;
    the chosen violation fraction is *not* a deployment prevalence.
    """
    rng = np.random.default_rng(seed)
    out = pool.copy()
    out["split"] = "unused"
    main = out[out["slice"] == "main"]
    used: set[str] = set()
    # Assign test and unfamiliar-heavy splits first so scarce tiers are not
    # consumed by less constrained splits.
    order = sorted(plan.quotas, key=lambda s: {"test": 0, "dev": 1, "audit": 2}.get(s, 3))
    for split in order:
        q = plan.quotas[split]
        for label, n_total, mix in (
            (False, q.normal, q.normal_tier_mix),
            (True, q.violation, q.violation_tier_mix),
        ):
            if n_total == 0:
                continue
            cand = main[(main["is_violation"] == label) & (main["tier"].isin(q.tiers))]
            if label and q.allowed_mechanisms is not None:
                cand = cand[cand["mechanism"].isin(q.allowed_mechanisms)]
            if split == "dev" and label:
                assert not cand["mechanism"].isin(HELD_OUT_MECHANISMS).any()
            for tier, n in _alloc_by_mix(n_total, mix, q.tiers).items():
                sub = cand if tier == "*" else cand[cand["tier"] == tier]
                chosen = _take(rng, sub, n, used, f"{split}/{'violation' if label else 'normal'}/{tier}")
                out.loc[out["cluster_id"].isin(chosen), "split"] = split
    extra = out["slice"] != "main"
    out.loc[extra, "split"] = "test"
    check_split_invariants(out)
    return out


def check_split_invariants(df: pd.DataFrame) -> None:
    """Hard assertions on an assigned label table."""
    assigned = df[df["split"] != "unused"]
    per_cluster = assigned.groupby("cluster_id")["split"].nunique()
    if (per_cluster > 1).any():
        raise LeakageError(f"clusters split across partitions: {per_cluster[per_cluster > 1].index[:5].tolist()}")
    if assigned["root_id"].duplicated().any():
        raise LeakageError("duplicate root ids")
    bad = assigned[(assigned["tier"] == "test_unfamiliar") & (assigned["split"] != "test")]
    if len(bad):
        raise LeakageError(f"test_unfamiliar roots outside test: {len(bad)}")
    bad = assigned[(assigned["tier"] == "dev_unfamiliar") & (assigned["split"] != "dev")]
    if len(bad):
        raise LeakageError(f"dev_unfamiliar roots outside dev: {len(bad)}")
    bad = assigned[(assigned["split"] == "dev") & assigned["mechanism"].isin(HELD_OUT_MECHANISMS)]
    if len(bad):
        raise LeakageError(f"held-out mechanisms in dev: {len(bad)}")
    for s in ("normal_fit", "normal_cal"):
        if assigned[(assigned["split"] == s)]["is_violation"].astype(bool).any():
            raise LeakageError(f"violations in normal-only split {s}")


# ------------------------------------------------------------------ manifest
def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def write_manifest(df: pd.DataFrame, path: str | Path, meta: dict | None = None) -> dict:
    """Write an immutable split manifest. Refuses to overwrite a different one."""
    path = Path(path)
    splits = {
        s: sorted(df.loc[df["split"] == s, "root_id"].tolist())
        for s in sorted(df["split"].unique())
        if s != "unused"
    }
    manifest = {
        "created_unix": int(time.time()),
        "meta": meta or {},
        "splits": {s: {"n": len(ids), "sha256": _sha(ids), "root_ids": ids} for s, ids in splits.items()},
    }
    manifest["digest"] = hashlib.sha256(
        json.dumps({s: v["sha256"] for s, v in manifest["splits"].items()}, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("digest") != manifest["digest"]:
            raise LeakageError(
                f"refusing to overwrite immutable manifest {path} with different split contents; "
                "record the change in docs/DECISION_LOG.md and write a new manifest version"
            )
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text())
    for s, v in m["splits"].items():
        if _sha(v["root_ids"]) != v["sha256"]:
            raise LeakageError(f"manifest hash mismatch for split {s}")
    return m


# -------------------------------------------------------------------- guards
class SplitGuard:
    """Raises if a pipeline stage reads roots outside its permitted splits."""

    def __init__(self, root_to_split: dict[str, str]):
        self.root_to_split = dict(root_to_split)
        self.log: list[tuple[str, str, int]] = []

    @classmethod
    def from_table(cls, df: pd.DataFrame) -> SplitGuard:
        return cls(dict(zip(df["root_id"], df["split"])))

    def require(self, root_ids, stage: str) -> None:
        allowed = STAGE_PERMISSIONS[stage.split(":")[0]]
        bad = {self.root_to_split.get(r, "unknown") for r in root_ids} - allowed
        if bad:
            raise LeakageError(f"stage {stage!r} touched splits {sorted(bad)}; allowed {sorted(allowed)}")
        self.log.append((stage, ",".join(sorted(allowed)), len(root_ids)))


class CalibrationLock:
    """Calibration opens only after the full scoring function is frozen.

    ``freeze`` hashes the pickled detector. ``open_calibration`` fails unless a
    freeze happened; ``verify`` fails if the detector changed after freezing
    (e.g. someone refits on calibration data).
    """

    def __init__(self):
        self._digest: str | None = None
        self.opened = False

    @staticmethod
    def _hash(obj) -> str:
        return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()

    def freeze(self, detector) -> str:
        self._digest = self._hash(detector)
        return self._digest

    def open_calibration(self) -> None:
        if self._digest is None:
            raise LeakageError("calibration split opened before the scoring function was frozen")
        self.opened = True

    def verify(self, detector) -> None:
        if self._digest is None or self._hash(detector) != self._digest:
            raise LeakageError("detector changed after freeze; calibration is invalid")
