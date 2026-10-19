import math
import warnings

import numpy as np
import pytest

from irmon.calibration.conformal import (
    SplitConformal,
    coverage_distribution,
    fpr_drift_report,
    per_group_fpr,
)


def test_threshold_order_statistic():
    cal = np.arange(1, 100, dtype=float)  # n = 99
    c = SplitConformal().fit(cal)
    # k = ceil(100 * 0.99) = 99 -> 99th smallest = 99.0
    assert c.threshold(0.01) == 99.0
    # k = ceil(100 * 0.95) = 95
    assert c.threshold(0.05) == 95.0


def test_threshold_infinite_when_n_too_small():
    c = SplitConformal().fit(np.random.default_rng(0).normal(size=50))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert math.isinf(c.threshold(0.01))
        assert w


def test_pvalue_matches_threshold_rule():
    rng = np.random.default_rng(1)
    c = SplitConformal().fit(rng.normal(size=999))
    s = rng.normal(size=5000)
    for a in (0.01, 0.05):
        assert np.array_equal(c.flag(s, a), s > c.threshold(a))


def test_marginal_validity_on_exchangeable_data():
    rng = np.random.default_rng(2)
    achieved = []
    for _ in range(300):
        c = SplitConformal().fit(rng.normal(size=199))
        achieved.append(np.mean(c.flag(rng.normal(size=400), 0.05)))
    # Marginal guarantee: E[FPR] <= alpha (= exactly floor((n+1)a)/(n+1) = 0.05 here).
    assert abs(np.mean(achieved) - 0.05) < 0.006


def test_coverage_law_mean():
    law = coverage_distribution(999, 0.01)
    assert law.a == 990 and law.b == 10
    assert abs(law.mean_fpr - 0.01) < 1e-12
    q = law.fpr_quantiles((0.05, 0.95))
    assert q[0.05] < 0.01 < q[0.95]
    with pytest.raises(ValueError):
        coverage_distribution(50, 0.01)


def test_group_fpr_and_drift():
    rng = np.random.default_rng(3)
    c = SplitConformal().fit(rng.normal(size=1999))
    s = np.r_[rng.normal(size=2000), rng.normal(loc=1.0, size=2000)]  # second group shifted
    g = np.r_[["familiar"] * 2000, ["unfamiliar"] * 2000]
    df = per_group_fpr(s, g, threshold=c.threshold(0.01))
    fam = df.set_index("group").loc["familiar"]
    unf = df.set_index("group").loc["unfamiliar"]
    assert fam.ci_low <= 0.01 <= fam.ci_high + 0.01
    assert unf.fpr > 0.03
    drift = fpr_drift_report(c, s, g, alphas=(0.01,))
    assert bool(drift.set_index("group").loc["unfamiliar", "exceeds_no_shift_band"])
