import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from irmon.evaluation import metrics as M


def test_ppv_plan_example():
    # Plan p.10: TPR 70%, FPR 1% -> PPV ~41% at 1% prevalence, ~6.5% at 0.1%.
    assert M.ppv_at_prevalence(0.70, 0.01, 0.01) == pytest.approx(0.41, abs=0.005)
    assert M.ppv_at_prevalence(0.70, 0.01, 0.001) == pytest.approx(0.065, abs=0.001)
    assert M.alarm_fraction(0.70, 0.01, 0.01) == pytest.approx(0.0169)


def test_auroc_ap_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)  # many ties
    assert M.auroc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert M.average_precision(s, y) == pytest.approx(average_precision_score(y, s))


def test_threshold_metrics_and_counts():
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert M.tpr_at_threshold(s, y, 0.35) == pytest.approx(1.0)
    assert M.achieved_fpr(s, y, 0.15) == pytest.approx(2 / 3)
    assert M.confusion_counts(s > 0.35, y) == {"tp": 3, "fp": 0, "fn": 0, "tn": 3}


def test_illustrative_counts_plan_p10():
    # 2,000 normal, 1,000 violations; 20 false alarms (1%), 700 caught (70%).
    y = np.r_[np.zeros(2000, int), np.ones(1000, int)]
    s = np.r_[np.r_[np.ones(20), np.zeros(1980)], np.r_[np.ones(700), np.zeros(300)]]
    assert M.achieved_fpr(s, y, 0.5) == pytest.approx(0.01)
    assert M.tpr_at_threshold(s, y, 0.5) == pytest.approx(0.70)


def test_worst_group_and_test_threshold_diagnostic():
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(size=1000), rng.normal(loc=0.8, size=1000), rng.normal(loc=3, size=200)]
    y = np.r_[np.zeros(2000, int), np.ones(200, int)]
    g = np.r_[["a"] * 1000, ["b"] * 1000, ["v"] * 200]
    worst, fpr, per = M.worst_group_fpr(s, y, g, threshold=2.0)
    assert worst == "b" and fpr == max(per.values())
    tpr, afpr, thr = M.tpr_at_achieved_fpr_test_threshold(s, y, 0.01)
    assert afpr <= 0.01 and 0 < tpr <= 1


def test_operating_point_table():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(300, int), np.ones(100, int)]
    s = rng.normal(size=400) + 2 * y
    df = M.operating_point_table({"m": s}, y, {"m": {0.01: 2.3, 0.05: 1.6}}, groups=np.r_[["g"] * 400])
    assert set(df["nominal_fpr"]) == {0.01, 0.05}
    assert {"ppv@0.01", "alarms@0.001", "worst_group_fpr"} <= set(df.columns)
