"""Restricted theoretical model: nuisance subspace versus failure shift (plan p.8).

Model (after the normal-response predictor has removed mu(context)):

    r = B z + delta * 1[failure] + eps,
    B in R^{d x r} orthonormal,  z ~ N(0, tau^2 I_r),  eps ~ N(0, sigma^2 I_d).

T (nuisance) responses are draws of B z + eps, so span(B) is estimable from
normal data. Projecting with P = I - B B^T removes every nuisance direction,
including any *benign shift* that inflates the nuisance scale at deployment
(tau -> kappa * tau) - but it also erases the part of the failure shift that
lies inside span(B).

What is and is not claimed
--------------------------
* The projection identity ||P delta||^2 = ||delta||^2 (1 - rho^2), with
  rho = ||B^T delta|| / ||delta||, is elementary linear algebra. It is NOT a
  novel theorem and must not be sold as the paper's contribution.
* With the covariance known, Mahalanobis already down-weights span(B):
  Sigma^{-1} = (I - BB^T)/sigma^2 + BB^T/(tau^2 + sigma^2). Projection only pays
  off when the nuisance scale at test differs from fitting (benign shift) or
  when Sigma must be estimated from few samples.
* The research target is the *finite-sample* characterisation: how well B is
  estimated from n T-responses (principal-angle error), how that error
  interacts with the overlap rho, and the resulting detection power at a fixed
  calibrated FPR under shift. ``simulate`` measures exactly that; the
  Davis-Kahan helper is a standard bound quoted for reference.

Everything in this module is simulation of the stated model; none of it is a
language-model result.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ geometry
def orthonormal_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return q[:, :r]


def failure_direction(B: np.ndarray, rho: float, norm: float, rng: np.random.Generator) -> np.ndarray:
    """delta with ||delta|| = norm and ||B^T delta|| / ||delta|| = rho."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("overlap must lie in [0, 1]")
    d, r = B.shape
    inside = B @ rng.normal(size=r)
    outside = rng.normal(size=d)
    outside -= B @ (B.T @ outside)
    inside /= np.linalg.norm(inside)
    outside /= np.linalg.norm(outside)
    return norm * (rho * inside + math.sqrt(max(0.0, 1.0 - rho**2)) * outside)


def overlap(B: np.ndarray, delta: np.ndarray) -> float:
    """rho = ||B^T delta|| / ||delta|| for orthonormal B."""
    return float(np.linalg.norm(B.T @ delta) / np.linalg.norm(delta))


def principal_angle_sin(B: np.ndarray, B_hat: np.ndarray) -> float:
    """Largest principal angle sine, ||sin Theta(B, B_hat)||_op."""
    s = np.linalg.svd(B.T @ B_hat, compute_uv=False)
    cos_min = float(np.clip(s.min(), 0.0, 1.0)) if s.size else 0.0
    return math.sqrt(max(0.0, 1.0 - cos_min**2))


def davis_kahan_bound(perturbation_opnorm: float, eigengap: float) -> float:
    """Standard Davis-Kahan sin-theta bound (Yu, Wang & Samworth 2015 variant):
    ||sin Theta||_op <= min(1, 2 ||Sigma_hat - Sigma||_op / gap). Quoted, not novel."""
    if eigengap <= 0:
        return 1.0
    return float(min(1.0, 2.0 * perturbation_opnorm / eigengap))


# ---------------------------------------------------------------- analytics
def noncentrality_projected(B: np.ndarray, delta: np.ndarray, sigma: float) -> float:
    """lambda = ||(I - BB^T) delta||^2 / sigma^2  (df = d - r)."""
    pd_ = delta - B @ (B.T @ delta)
    return float(pd_ @ pd_ / sigma**2)


def noncentrality_mahalanobis(B: np.ndarray, delta: np.ndarray, tau: float, sigma: float) -> float:
    """lambda' = delta^T (tau^2 BB^T + sigma^2 I)^{-1} delta  (df = d)."""
    inside = B.T @ delta
    outside_sq = float(delta @ delta - inside @ inside)
    return outside_sq / sigma**2 + float(inside @ inside) / (tau**2 + sigma**2)


def chi2_power(df: int, lam: float, alpha: float) -> float:
    """P(chi2_df(lam) > q_{1-alpha}(chi2_df))."""
    q = stats.chi2.ppf(1 - alpha, df)
    return float(stats.ncx2.sf(q, df, lam)) if lam > 0 else float(alpha)


def analytical_power(d: int, r: int, rho: float, delta_norm: float, tau: float, sigma: float,
                     alpha: float = 0.01) -> dict:
    """No-shift power of the oracle-projected and known-covariance detectors.

    Depends on B and delta only through rho, so it is exact for any basis.
    """
    lam_p = (delta_norm**2) * (1 - rho**2) / sigma**2
    lam_m = (delta_norm**2) * (1 - rho**2) / sigma**2 + (delta_norm**2) * rho**2 / (tau**2 + sigma**2)
    return {
        "power_projected": chi2_power(d - r, lam_p, alpha),
        "power_mahalanobis_known": chi2_power(d, lam_m, alpha),
        "lambda_projected": lam_p,
        "lambda_mahalanobis": lam_m,
    }


def shifted_fpr_mahalanobis_known(d: int, r: int, tau: float, sigma: float, kappa: float,
                                  alpha: float = 0.01, n_mc: int = 200000, seed: int = 0) -> float:
    """FPR of the known-covariance Mahalanobis test when the nuisance scale at
    test is kappa * tau (benign shift). Score = c * chi2_r + chi2_{d-r} with
    c = (kappa^2 tau^2 + sigma^2) / (tau^2 + sigma^2). The projected detector's
    FPR is exactly alpha for every kappa."""
    rng = np.random.default_rng(seed)
    c = (kappa**2 * tau**2 + sigma**2) / (tau**2 + sigma**2)
    s = c * rng.chisquare(r, n_mc) + rng.chisquare(d - r, n_mc)
    return float(np.mean(s > stats.chi2.ppf(1 - alpha, d)))


# ---------------------------------------------------------------- simulation
def _ledoit_wolf_inv(X: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf(assume_centered=True).fit(X)
    return np.linalg.inv(lw.covariance_)


def simulate(
    d: int = 32,
    r: int = 4,
    n_fit: int = 200,
    rho: float = 0.25,
    delta_norm: float = 4.0,
    tau: float = 3.0,
    sigma: float = 1.0,
    kappa: float = 1.0,
    n_cal: int = 2000,
    n_test: int = 2000,
    alpha: float = 0.01,
    seed: int = 0,
) -> list[dict]:
    """One replicate. Returns one row per detector.

    Fitting data: ``n_fit`` normal residuals (for covariance) and ``n_fit``
    T-responses (for the nuisance subspace), all at nuisance scale tau.
    Calibration: ``n_cal`` normals at scale tau; conformal threshold at alpha.
    Test: ``n_test`` normals and failures at scale kappa * tau (kappa > 1 is a
    benign shift the fitting data never showed).
    """
    rng = np.random.default_rng(seed)
    B = orthonormal_basis(d, r, rng)
    delta = failure_direction(B, rho, delta_norm, rng)

    def draw(n, scale):
        return (B @ rng.normal(scale=scale, size=(r, n))).T + rng.normal(scale=sigma, size=(n, d))

    fit_normal = draw(n_fit, tau)
    t_resp = draw(n_fit, tau)
    cal = draw(n_cal, tau)
    test_norm = draw(n_test, kappa * tau)
    test_fail = draw(n_test, kappa * tau) + delta

    # Estimated nuisance subspace: top-r eigenvectors of T-response second moment.
    _, vecs = np.linalg.eigh(t_resp.T @ t_resp / n_fit)
    B_hat = vecs[:, ::-1][:, :r]
    sin_theta = principal_angle_sin(B, B_hat)
    sigma_hat2 = float(np.mean(np.sum((fit_normal - fit_normal @ B_hat @ B_hat.T) ** 2, 1)) / (d - r))

    sig_inv_true = (np.eye(d) - B @ B.T) / sigma**2 + B @ B.T / (tau**2 + sigma**2)
    sig_inv_est = _ledoit_wolf_inv(fit_normal)
    P, P_hat = np.eye(d) - B @ B.T, np.eye(d) - B_hat @ B_hat.T

    scorers = {
        "mahalanobis_known": lambda X: np.einsum("nd,de,ne->n", X, sig_inv_true, X),
        "mahalanobis_estimated": lambda X: np.einsum("nd,de,ne->n", X, sig_inv_est, X),
        "projection_oracle": lambda X: np.sum((X @ P) ** 2, 1) / sigma**2,
        "projection_estimated": lambda X: np.sum((X @ P_hat) ** 2, 1) / sigma_hat2,
    }
    rows = []
    for name, f in scorers.items():
        cal_s = np.sort(f(cal))
        k = math.ceil(round((n_cal + 1) * (1 - alpha), 9))
        thr = cal_s[k - 1] if k <= n_cal else np.inf
        rows.append({
            "method": name, "d": d, "r": r, "n_fit": n_fit, "rho": rho, "delta_norm": delta_norm,
            "tau": tau, "sigma": sigma, "kappa": kappa, "alpha": alpha, "seed": seed,
            "tpr": float(np.mean(f(test_fail) > thr)),
            "fpr": float(np.mean(f(test_norm) > thr)),
            "sin_theta": sin_theta if name == "projection_estimated" else (0.0 if name == "projection_oracle" else np.nan),
        })
    return rows


def run_grid(grid: dict[str, Iterable], n_reps: int = 5, seed: int = 0, **fixed) -> pd.DataFrame:
    """Cartesian grid over ``grid`` values (keys are ``simulate`` arguments),
    ``n_reps`` replicates each. Adds analytical no-shift power columns."""
    keys = list(grid)
    rows: list[dict] = []
    for combo_i, combo in enumerate(itertools.product(*[list(grid[k]) for k in keys])):
        params = {**fixed, **dict(zip(keys, combo))}
        for rep in range(n_reps):
            rows.extend(simulate(**params, seed=seed + 1000 * combo_i + rep))
    df = pd.DataFrame(rows)
    ana = df.apply(
        lambda row: pd.Series(analytical_power(int(row.d), int(row.r), row.rho, row.delta_norm,
                                               row.tau, row.sigma, row.alpha)),
        axis=1,
    )
    return pd.concat([df, ana[["power_projected", "power_mahalanobis_known"]]], axis=1)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        df.groupby(by + ["method"])
        .agg(tpr=("tpr", "mean"), tpr_sd=("tpr", "std"), fpr=("fpr", "mean"),
             sin_theta=("sin_theta", "mean"), n_reps=("seed", "count"),
             power_projected=("power_projected", "first"),
             power_mahalanobis_known=("power_mahalanobis_known", "first"))
        .reset_index()
    )


TITLE_NOTE = "Restricted theoretical model - synthetic simulation, not a language-model result"


def plot_theory(df_overlap: pd.DataFrame, df_n: pd.DataFrame, out_overlap: str, out_n: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarise(df_overlap, ["rho", "kappa"])
    kappas = sorted(s["kappa"].unique())
    styles = {
        "mahalanobis_known": ("#4C72B0", "-"), "mahalanobis_estimated": ("#4C72B0", "--"),
        "projection_oracle": ("#DD8452", "-"), "projection_estimated": ("#DD8452", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    k0, k1 = kappas[0], kappas[-1]
    # (a) power vs overlap without shift, with the analytic curve.
    ax = axes[0]
    sk = s[s["kappa"] == k0]
    for m, (col, ls) in styles.items():
        sm = sk[sk["method"] == m].sort_values("rho")
        ax.plot(sm["rho"], sm["tpr"], color=col, ls=ls, marker="o", ms=3, label=m)
    ana = sk[sk["method"] == "projection_oracle"].sort_values("rho")
    ax.plot(ana["rho"], ana["power_projected"], color="k", lw=0.8, ls=":", label="analytic, projected")
    ax.set_title(f"(a) TPR, nuisance scale at test = {k0:g} x fit (no shift)", fontsize=9)
    # (b) achieved FPR at the calibrated 1% threshold, per nuisance scale.
    ax = axes[1]
    width = 0.8 / len(styles)
    for j, (m, (col, _)) in enumerate(styles.items()):
        vals = [s[(s["kappa"] == k) & (s["method"] == m)]["fpr"].mean() for k in kappas]
        ax.bar([i + j * width for i in range(len(kappas))], vals, width, color=col,
               alpha=1.0 if "known" in m or "oracle" in m else 0.55, label=m)
    nominal = float(df_overlap["alpha"].iloc[0])
    ax.axhline(nominal, color="k", lw=0.8, ls=":", label=f"nominal {nominal:g}")
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(kappas))])
    ax.set_xticklabels([f"test scale {k:g}x" for k in kappas])
    ax.set_ylabel("achieved FPR (calibrated at 1%)")
    ax.set_title("(b) false alarms under benign nuisance shift", fontsize=9)
    # (c) power under shift - must be read together with (b).
    ax = axes[2]
    sk = s[s["kappa"] == k1]
    for m, (col, ls) in styles.items():
        sm = sk[sk["method"] == m].sort_values("rho")
        ax.plot(sm["rho"], sm["tpr"], color=col, ls=ls, marker="o", ms=3, label=f"{m} (FPR {sm['fpr'].mean():.3f})")
    ax.set_title(f"(c) TPR at test scale {k1:g}x - uncontrolled FPR inflates TPR (see b)", fontsize=9)
    for ax in (axes[0], axes[2]):
        ax.set_xlabel(r"overlap $\rho = \|B^\top\delta\| / \|\delta\|$")
        ax.set_ylabel("TPR at calibrated 1% threshold")
        ax.set_ylim(-0.02, 1.02)
    for ax, loc in zip(axes, ("upper right", "upper left", "upper right")):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc=loc)
    fig.suptitle(TITLE_NOTE, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_overlap, dpi=150)
    plt.close(fig)

    sn = summarise(df_n, ["n_fit"])
    fig, ax1 = plt.subplots(figsize=(5.6, 4.2))
    est = sn[sn["method"] == "projection_estimated"].sort_values("n_fit")
    ax1.plot(est["n_fit"], est["sin_theta"], marker="o", color="#55A868", label=r"$\|\sin\Theta(B,\hat B)\|$")
    ax1.set_xscale("log")
    ax1.set_xlabel("number of T-responses used to estimate the nuisance subspace")
    ax1.set_ylabel("largest principal-angle sine", color="#55A868")
    ax2 = ax1.twinx()
    for m, (col, ls) in styles.items():
        sm = sn[sn["method"] == m].sort_values("n_fit")
        ax2.plot(sm["n_fit"], sm["tpr"], color=col, ls=ls, marker=".", label=f"TPR {m}")
    ax2.set_ylabel("TPR at calibrated 1% threshold")
    ax2.set_ylim(-0.02, 1.02)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax1.grid(alpha=0.3)
    fig.suptitle(TITLE_NOTE, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_n, dpi=150)
    plt.close(fig)
