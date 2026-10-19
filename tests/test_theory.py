import numpy as np
import pytest

from irmon.theory import restricted_model as T


def test_overlap_construction_and_projection_identity():
    rng = np.random.default_rng(0)
    B = T.orthonormal_basis(20, 3, rng)
    for rho in (0.0, 0.3, 0.8, 1.0):
        d = T.failure_direction(B, rho, 2.0, rng)
        assert T.overlap(B, d) == pytest.approx(rho, abs=1e-9)
        # Elementary identity: ||P delta||^2 = ||delta||^2 (1 - rho^2).
        assert T.noncentrality_projected(B, d, 1.0) == pytest.approx(4.0 * (1 - rho**2), abs=1e-9)


def test_analytical_matches_simulation_known_basis():
    rows = []
    for seed in range(6):
        rows += T.simulate(d=16, r=3, n_fit=400, rho=0.3, delta_norm=4.0, tau=2.0, sigma=1.0,
                           n_cal=3000, n_test=3000, alpha=0.05, seed=seed)
    sim = np.mean([r["tpr"] for r in rows if r["method"] == "projection_oracle"])
    ana = T.analytical_power(16, 3, 0.3, 4.0, 2.0, 1.0, alpha=0.05)["power_projected"]
    assert sim == pytest.approx(ana, abs=0.04)


def test_full_overlap_kills_projected_power():
    p = T.analytical_power(32, 4, 1.0, 6.0, 3.0, 1.0, alpha=0.01)
    assert p["power_projected"] == pytest.approx(0.01)
    assert p["power_mahalanobis_known"] > p["power_projected"]


def test_subspace_error_decreases_with_n():
    def err(n):
        return np.mean([
            [r["sin_theta"] for r in T.simulate(d=24, r=3, n_fit=n, n_cal=200, n_test=200, seed=s)
             if r["method"] == "projection_estimated"][0]
            for s in range(4)
        ])
    assert err(400) < err(40)


def test_projection_robust_to_nuisance_shift():
    fpr_m = T.shifted_fpr_mahalanobis_known(32, 4, 3.0, 1.0, kappa=2.0, n_mc=50000)
    assert fpr_m > 0.05  # known-covariance Mahalanobis loses FPR control under shift
    rows = T.simulate(d=32, r=4, n_fit=300, kappa=2.0, n_cal=2000, n_test=4000, seed=1)
    proj = [r for r in rows if r["method"] == "projection_oracle"][0]
    assert proj["fpr"] < 0.025


def test_davis_kahan_helper():
    assert T.davis_kahan_bound(0.1, 1.0) == pytest.approx(0.2)
    assert T.davis_kahan_bound(5.0, 1.0) == 1.0
