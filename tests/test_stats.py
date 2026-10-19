import numpy as np
import pytest

from irmon.evaluation import stats as S


def test_mcnemar_spec_examples():
    # Spec p.4: proposed-only 60, baseline-only 25 -> p ~ 0.000187.
    assert S.mcnemar_exact(250, 60, 25, 165).p_value == pytest.approx(0.000187, rel=0.01)
    # Spec p.10 (Override Lab illustrative tables).
    assert S.mcnemar_exact(80, 70, 20, 330).p_value == pytest.approx(1.14e-7, rel=0.01)
    assert S.mcnemar_exact(85, 40, 15, 360).p_value == pytest.approx(0.00102, rel=0.01)
    assert S.mcnemar_exact(156, 8, 24, 12).p_value == pytest.approx(0.00700, rel=0.01)
    assert S.mcnemar_exact(10, 0, 0, 5).p_value == 1.0


def test_mcnemar_from_flags_counts():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 1, 0, 0], bool)
    r = S.mcnemar_from_flags(a, b)
    assert (r.both, r.a_only, r.b_only, r.neither) == (1, 2, 1, 1)


def test_holm():
    df = S.holm_bonferroni([0.01, 0.04, 0.03, 0.2])
    assert df["p_holm"].tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert df["reject"].tolist() == [True, False, False, False]


def test_intervals_cover_point():
    for k, n in ((0, 20), (5, 20), (20, 20)):
        for f in (S.wilson_interval, S.clopper_pearson):
            lo, hi = f(k, n)
            assert lo <= k / n <= hi


def test_paired_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(0)
    covered = 0
    for rep in range(40):
        n = 400
        y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
        a = (rng.random(n) < np.where(y == 1, 0.6, 0.02))
        b = (rng.random(n) < np.where(y == 1, 0.5, 0.02))
        res = S.paired_cluster_bootstrap(S.tpr_difference, a, b, y, n_boot=400, seed=rep)
        covered += res.ci_low <= 0.1 <= res.ci_high
    assert covered >= 34  # ~95% nominal


def test_cluster_bootstrap_widens_with_correlated_clusters():
    rng = np.random.default_rng(1)
    n_clusters, per = 40, 10
    clusters = np.repeat(np.arange(n_clusters), per)
    effect = np.repeat(rng.normal(scale=0.5, size=n_clusters), per)
    a = effect + rng.normal(scale=0.1, size=clusters.size)
    b = np.zeros_like(a)
    y = np.ones_like(clusters)
    stat = lambda x, z, yy: float(np.mean(x - z))  # noqa: E731
    naive = S.paired_cluster_bootstrap(stat, a, b, y, None, n_boot=500)
    clus = S.paired_cluster_bootstrap(stat, a, b, y, clusters, n_boot=500)
    assert clus.se > 2 * naive.se
    miller = S.clustered_paired_difference(a, b, clusters)
    assert miller.n_clusters == n_clusters
    assert miller.se == pytest.approx(clus.se, rel=0.35)


def test_power_and_sample_size():
    p_small = S.power_mcnemar(100, 0.1, 0.05)
    p_big = S.power_mcnemar(1000, 0.1, 0.05)
    assert 0 < p_small < p_big <= 1
    assert S.power_mcnemar(500, 0.1, 0.05) == pytest.approx(S.power_mcnemar(500, 0.1, 0.05, exact=False), abs=0.05)
    n = S.required_n_for_tpr_gain(0.05, 0.15, target_power=0.8)
    assert S.power_mcnemar(n, 0.1, 0.05, exact=False) >= 0.8
    assert S.power_mcnemar(n - 1, 0.1, 0.05, exact=False) < 0.8


def test_independent_z_from_spec():
    # Spec p.4: ~2.25 for 55% vs 62% with 500 per rate (NOT a paired test).
    assert S.two_proportion_z(0.62, 0.55, 500, 500) == pytest.approx(2.25, abs=0.02)


def test_seed_spread():
    df = S.seed_spread({11: {"tpr": 0.5}, 23: {"tpr": 0.6}, 47: {"tpr": 0.7}})
    assert df.loc["tpr", "mean"] == pytest.approx(0.6)
    assert df.loc["tpr", "n_seeds"] == 3
