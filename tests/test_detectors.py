"""Detectors, calibration, metrics, information boundary and attack surrogates."""

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from actmon.data.features import FeatureSet
from actmon.detectors import REGISTRY, build
from actmon.evaluation.audit import audit_table, utility_table
from actmon.evaluation.calibrate import flags, threshold_for_fpr
from actmon.evaluation.guard import guard_table
from actmon.evaluation.metrics import ambiguous_action_strings, detection_table, paired_tests
from actmon.synthetic import SyntheticConfig, SyntheticWorld

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "actmon"


def _imports(path):
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_detectors_never_import_the_oracle_or_evaluation():
    for f in (SRC / "detectors").glob("*.py"):
        mods = _imports(f)
        assert not any(m.startswith("actmon.sim.oracle") or m.startswith("actmon.evaluation") or
                       m.startswith("actmon.data.collect") for m in mods), f.name


def test_featureset_carries_no_labels():
    fields = set(FeatureSet.__dataclass_fields__)
    for bad in ("valid", "violations", "label", "labels", "y", "mechanism", "planner"):
        assert bad not in fields


def test_conformal_threshold_controls_fpr():
    rng = np.random.default_rng(0)
    fprs = []
    for _ in range(200):
        cal, test = rng.normal(size=999), rng.normal(size=5000)
        fprs.append(flags(test, threshold_for_fpr(cal, 0.05)).mean())
    assert np.mean(fprs) <= 0.05 + 0.003
    assert threshold_for_fpr(np.arange(10.0), 0.01) == float("inf")  # too few points for 1 %


@pytest.fixture(scope="module")
def worlds():
    cfg = SyntheticConfig(d=24, q=4, n_classes=4)
    out = {}
    for regime in ("null", "context_shift", "response_only", "base_shift"):
        w = SyntheticWorld(cfg, np.random.default_rng(0))
        rng = np.random.default_rng({"null": 1, "context_shift": 2, "response_only": 3, "base_shift": 4}[regime])
        fit, _ = w.sample(rng, 800, 0, regime, "fit")
        dev, ydev = w.sample(rng, 300, 300, regime, "dev")
        cal, _ = w.sample(rng, 1500, 0, regime, "calib")
        test, y = w.sample(rng, 1500, 300, regime, "test")
        out[regime] = (fit, dev, ydev, cal, test, y)
    return out


def _tpr(name, data, alpha=0.05):
    fit, dev, ydev, cal, test, y = data
    det = build(name)
    det.fit(fit, dev, ydev) if det.supervised else det.fit(fit)
    f = flags(det.score(test), threshold_for_fpr(det.score(cal), alpha))
    return f[y == 1].mean(), f[y == 0].mean()


def test_known_mechanisms_are_detected_by_the_right_detector(worlds):
    assert _tpr("response", worlds["response_only"])[0] > _tpr("mahalanobis_pooled", worlds["response_only"])[0] + 0.3
    assert _tpr("mahalanobis_task", worlds["context_shift"])[0] > \
        _tpr("mahalanobis_pooled", worlds["context_shift"])[0] + 0.3
    assert _tpr("mahalanobis_pooled", worlds["base_shift"])[0] > 0.5
    for name in ("response", "mahalanobis_pooled", "mahalanobis_task", "linear_probe"):
        tpr, fpr = _tpr(name, worlds["null"])
        assert tpr < 0.12 and fpr < 0.08, name


def test_supervised_detectors_refuse_to_fit_without_labels(worlds):
    fit = worlds["null"][0]
    for name in ("linear_probe", "behavior_logistic"):
        with pytest.raises(ValueError):
            build(name).fit(fit)
    assert {n for n in REGISTRY if build(n).supervised} == {"linear_probe", "behavior_logistic"}


def test_attack_surrogates_equal_detector_scores(worlds):
    torch = pytest.importorskip("torch")
    from actmon.attack.obfuscation import surrogate

    fit, dev, ydev, cal, test, y = worlds["response_only"]
    sub = test.subset(np.arange(40))
    base = {s: torch.tensor(v, dtype=torch.float32) for s, v in sub.base.items()}
    var = {m: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for m, d in sub.variants.items()}
    ctx = torch.tensor(sub.context, dtype=torch.float32)
    for name in ("mahalanobis_pooled", "mahalanobis_task", "linear_probe", "response", "response_plus_base",
                 "response_tc_only", "response_no_context"):
        det = build(name)
        det.fit(fit, dev, ydev) if det.supervised else det.fit(fit)
        got = surrogate(det)(base, var, ctx, list(sub.context_class)).detach().numpy()
        want = det.score(sub)
        assert np.allclose(got, want, rtol=1e-3, atol=1e-2 * (1 + np.abs(want).max())), name


def test_detection_and_paired_tables():
    y = np.r_[np.zeros(1000, int), np.ones(100, int)]
    rng = np.random.default_rng(0)
    a = rng.normal(size=1100) + 2.0 * y
    b = rng.normal(size=1100) + 1.0 * y
    cal = {"a": rng.normal(size=5000), "b": rng.normal(size=5000)}
    thr = {k: {0.05: threshold_for_fpr(v, 0.05)} for k, v in cal.items()}
    tab = detection_table({"a": a, "b": b}, y, thr)
    ra = tab[tab.method == "a"].iloc[0]
    assert ra.detected == int((a[y == 1] > thr["a"][0.05]).sum())
    assert abs(ra["alarm_frac@0.1"] - (ra.tpr * 0.1 + ra.fpr * 0.9)) < 1e-12
    pt = paired_tests({"a": a, "b": b}, y, thr, "a", ["b"], clusters=np.arange(1100) % 10)
    r = pt.iloc[0]
    assert r.both + r.proposed_only + r.comparator_only + r.neither == 100
    assert r.tpr_diff == pytest.approx((r.proposed_only - r.comparator_only) / 100)
    assert r.mcnemar_p < 0.05 and r.n_clusters == 10


def test_audit_utility_and_guard():
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    s = np.r_[np.zeros(90), np.ones(10)]
    t = audit_table({"m": s}, y, ks=(10, 20))
    assert t.iloc[0].violations_found == 10 and t.iloc[1].precision == 0.5
    comp = np.r_[np.zeros(2, bool), np.ones(8, bool)]
    u = utility_table({"m": np.arange(10.0)}, comp, {"m": {0.05: 7.5}})  # flags roots 8 and 9
    assert u.iloc[0].completed == 8 and u.iloc[1].completed == 6 and u.iloc[1].blocked == 2
    assert u.iloc[1].utility_cost == pytest.approx(0.2)
    lab = pd.DataFrame({"valid": [True, False, False, False], "violations": ["", "reserve", "review",
                                                                             "handoff+auth"]})
    g = guard_table(lab).set_index("group")
    assert g.loc["all violating", "intercepted"] == 2 and g.loc["valid (false blocking)", "intercepted"] == 0


def test_ambiguous_action_strings():
    lab = pd.DataFrame({"action": ["allocate p1", "allocate p1", "review"], "valid": [True, False, True]})
    out = ambiguous_action_strings(lab)
    assert list(out.action) == ["allocate p1"]
