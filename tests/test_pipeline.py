"""End-to-end pipeline test on the synthetic simulator (pipeline validation only).

Runs every stage at a tiny scale and checks that artifacts exist and that the
leakage machinery holds. The numbers produced are NOT evidence about language
models and are not asserted on.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from irmon.config import load_config
from irmon.data import LeakageError
from irmon.pipeline import stages as S


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = load_config("configs/smoke_synthetic.yaml", {
        "run": {"name": "pytest_smoke", "out_dir": str(out)},
        "env": {"pool_size": 3000, "n_outcome_matched_pairs": 10},
        "splits": {"scale": 0.08},
        "evaluation": {"n_bootstrap": 200},
        "selection": {"width": [8], "ridge_alpha": [1.0], "aggregation": ["concat"], "nuisance_rank": [0, 4]},
        "detectors": {"baselines": ["mahalanobis_pooled", "mahalanobis_conditional", "output_consistency",
                                    "behavior_logistic", "oracle", "metadata_shortcut", "probe_supervised"]},
        "attack": {"enabled": True, "n_roots": 3, "black_box_budget": 6,
                   "detectors": ["proposed", "mahalanobis_conditional"]},
    })
    c = S.RunContext(cfg)
    S.run_all(c)
    return c


def test_artifacts_exist(ctx):
    for rel in ["config.resolved.yaml", "env/validity.json", "env/review_sample.jsonl", "labels/pool.csv",
                "splits/manifest.json", "splits/labels.csv", "bundles/normal_fit/arrays.npz",
                "selection/selected.json", "detectors/proposed.pkl", "calibration/thresholds.json",
                "results/operating_points.csv", "results/per_group_fpr.csv", "results/paired.csv",
                "results/ablations.csv", "results/control.csv", "results/utility.csv",
                "results/root_predictions.csv", "results/report.md", "attacks/report.json",
                "manifest/run_manifest.json", "decision_log.jsonl"]:
        assert (ctx.run_dir / rel).exists(), rel


def test_synthetic_outputs_carry_evidence_banner(ctx):
    text = (ctx.run_dir / "results" / "operating_points.md").read_text()
    assert "SYNTHETIC" in text
    assert "not evidence" in (ctx.run_dir / "results" / "report.md").read_text()


def test_primary_comparison_recorded_once(ctx):
    entries = [json.loads(x) for x in (ctx.run_dir / "decision_log.jsonl").read_text().splitlines()]
    prim = [e for e in entries if e["kind"] == "primary"]
    assert prim and prim[0]["content"]["baseline"] == "mahalanobis_conditional"


def test_oracle_is_the_ceiling(ctx):
    op = pd.read_csv(ctx.run_dir / "results" / "operating_points.csv", comment="#")
    row = op[(op.method == "oracle") & (op.nominal_fpr == 0.01)].iloc[0]
    assert row.tpr == 1.0 and row.achieved_fpr == 0.0


def test_detector_tampering_after_calibration_is_caught(ctx):
    p = ctx.run_dir / "detectors" / "oracle.pkl"
    original = p.read_bytes()
    try:
        p.write_bytes(original + b"tampered")
        with pytest.raises(LeakageError):
            S._verified_detectors(ctx)
    finally:
        p.write_bytes(original)


def test_test_split_unreachable_from_fit_stage(ctx):
    with pytest.raises(LeakageError):
        S._bundle(ctx, "test", "fit:illegal")
