"""Split design, manifest immutability and leakage-guard tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irmon.data import (
    CalibrationLock,
    LeakageError,
    SplitGuard,
    SplitPlan,
    assign_splits,
    load_manifest,
    write_manifest,
)


def _pool(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tiers = rng.choice(["familiar", "dev_unfamiliar", "test_unfamiliar"], size=n, p=[0.6, 0.15, 0.25])
    is_v = rng.random(n) < 0.3
    mech = np.where(is_v, rng.choice(["ordinary", "prompt_conflict", "adapter_conflict", "cue_conditional"], size=n), "none")
    ids = [f"fs-{i:06x}" for i in range(n)]
    return pd.DataFrame({
        "root_id": ids, "cluster_id": ids, "tier": tiers, "is_violation": is_v,
        "mechanism": mech, "slice": "main",
    })


def test_scaled_pilot_assignment_respects_every_invariant():
    plan = SplitPlan.pilot().scaled(0.1)
    out = assign_splits(_pool(), plan, seed=1)
    counts = out[out.split != "unused"].groupby(["split", "is_violation"]).size()
    assert counts[("normal_fit", False)] == 100
    assert counts[("dev", True)] == 50
    assert counts[("test", False)] == 200
    test_norm = out[(out.split == "test") & ~out.is_violation]
    assert set(test_norm.tier) == {"familiar", "test_unfamiliar"}
    assert not out[(out.split == "dev")].mechanism.isin(["adapter_conflict", "cue_conditional"]).any()
    assert not out[out.split.isin(["normal_fit", "normal_cal"])].is_violation.any()


def test_clusters_are_never_split():
    pool = _pool(3000, seed=2)
    pool.loc[1::2, "cluster_id"] = pool["cluster_id"].shift(1)[1::2]
    pool.loc[pool.index % 2 == 1, "is_violation"] = pool["is_violation"].shift(1)[pool.index % 2 == 1]
    pool.loc[pool.index % 2 == 1, "tier"] = pool["tier"].shift(1)[pool.index % 2 == 1]
    pool.loc[pool.index % 2 == 1, "mechanism"] = pool["mechanism"].shift(1)[pool.index % 2 == 1]
    out = assign_splits(pool, SplitPlan.pilot().scaled(0.05), seed=3)
    assigned = out[out.split != "unused"]
    assert (assigned.groupby("cluster_id").split.nunique() == 1).all()


def test_pool_too_small_raises_instead_of_relaxing():
    with pytest.raises(ValueError, match="too small"):
        assign_splits(_pool(200), SplitPlan.pilot(), seed=0)


def test_manifest_is_immutable(tmp_path):
    out = assign_splits(_pool(), SplitPlan.pilot().scaled(0.1), seed=1)
    path = tmp_path / "manifest.json"
    m = write_manifest(out, path, meta={"seed": 1})
    assert load_manifest(path)["digest"] == m["digest"]
    write_manifest(out, path)  # identical content is fine
    out2 = assign_splits(_pool(), SplitPlan.pilot().scaled(0.1), seed=2)
    with pytest.raises(LeakageError):
        write_manifest(out2, path)


def test_split_guard_blocks_test_access_during_fitting():
    out = assign_splits(_pool(), SplitPlan.pilot().scaled(0.1), seed=1)
    guard = SplitGuard.from_table(out)
    fit_ids = out.loc[out.split == "normal_fit", "root_id"].tolist()
    test_ids = out.loc[out.split == "test", "root_id"].tolist()
    guard.require(fit_ids, "fit:proposed")
    with pytest.raises(LeakageError):
        guard.require(fit_ids + test_ids[:1], "fit:proposed")
    with pytest.raises(LeakageError):
        guard.require(test_ids, "select:layers")
    guard.require(test_ids, "test:final")


def test_calibration_lock_enforces_order_and_freezing():
    lock = CalibrationLock()
    det = {"w": [1, 2, 3]}
    with pytest.raises(LeakageError):
        lock.open_calibration()
    lock.freeze(det)
    lock.open_calibration()
    lock.verify(det)
    det["w"].append(4)
    with pytest.raises(LeakageError):
        lock.verify(det)
