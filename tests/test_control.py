import numpy as np
import pytest

from irmon.evaluation import control as C
from irmon.evaluation.gates import DecisionLog, go_no_go


def test_offline_topk_exact_budget():
    s = np.arange(1000, dtype=float)
    f = C.offline_topk(s, 0.02)
    assert f.sum() == 20 and f[-20:].all()


def test_online_budget_never_exceeded_and_exhaustion_recorded():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    s = rng.normal(size=1000) + 2.0 * y
    out = C.run_online(s, y, threshold=0.5, budget=0.02, seed=3)
    summ = out["summary"]
    assert summ["audited"] <= int(0.02 * 1000)
    assert summ["budget_capacity"] == 20
    # Threshold 0.5 flags far more than 2% -> budget exhausts and later flags are missed.
    assert summ["exhausted_at"] is not None
    assert summ["missed_after_exhaustion"] > 0
    log = out["log"]
    assert (log["budget_left"] >= 0).all()
    after = log[log["t"] > summ["exhausted_at"]]
    assert not after["audited"].any() and after["flagged"].any()


def test_fpr_is_not_audit_budget():
    # Spec p.5: 310 violating + 250 clean flagged of 5,500 -> ~10.2% of roots.
    assert (310 + 250) / 5500 == pytest.approx(0.102, abs=0.001)
    assert C.offline_topk(np.arange(5500.0), 0.05).sum() == 275


def test_rule_escalation_and_utility():
    flags = C.rule_based_escalation(np.array([0, 1, 0, 2]), slot_names=["ALLOC_FAST", "ALLOC_STD", "DEFER"])
    assert flags.tolist() == [True, False, True, False]
    u = C.utility_under_monitoring(np.array([1, 1, 1, 0], bool), np.array([1, 0, 0, 0], bool))
    assert u["task_success_unmonitored"] == 0.75 and u["task_success_monitored"] == 0.5
    assert u["utility_cost"] == pytest.approx(0.25)


def test_combine_rank_max_and_threshold():
    a, b = np.array([0.0, 1.0, 2.0]), np.array([5.0, 0.0, 1.0])
    comb = C.combine_rank_max(a, b)
    assert comb.shape == (3,) and comb.max() <= 1
    thr = C.threshold_for_alarm_rate(np.arange(100.0), 0.05)
    assert np.mean(np.arange(100.0) > thr) == pytest.approx(0.05)


def test_go_no_go_and_decision_log(tmp_path):
    ok = go_no_go(primary_ci_low=0.01, primary_gain=0.07, worst_group_fpr=0.02, nominal_fpr=0.01)
    assert ok.decision == "go"
    bad = go_no_go(primary_ci_low=-0.01, primary_gain=0.03, worst_group_fpr=0.08, nominal_fpr=0.01)
    assert bad.decision == "no-go" and len(bad.reasons) == 3
    log = DecisionLog(tmp_path / "log.jsonl")
    log.append("primary", {"metric": "tpr@1%", "baseline": "mahalanobis_conditional"})
    log.append("primary", {"metric": "tpr@1%", "baseline": "mahalanobis_conditional"})
    with pytest.raises(ValueError):
        log.append("primary", {"metric": "auroc"})
    e = log.append("primary", {"metric": "auroc"}, exploratory=True)
    assert e["exploratory"] and len(log.entries()) == 3
