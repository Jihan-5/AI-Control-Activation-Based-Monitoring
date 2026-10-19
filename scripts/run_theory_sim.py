#!/usr/bin/env python
"""Run the restricted-theoretical-model simulation (plan p.8).

Writes
  docs/figures/theory_power_vs_overlap.png
  docs/figures/theory_subspace_error_vs_n.png
  reports/theory/theory_grid.csv

These are simulations of the stated Gaussian model, not language-model results.

Usage: python scripts/run_theory_sim.py [--reps 5] [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from irmon.theory.restricted_model import plot_theory, run_grid, shifted_fpr_mahalanobis_known


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="tiny grid for CI")
    ap.add_argument("--fig-dir", default="docs/figures")
    ap.add_argument("--out-dir", default="reports/theory")
    args = ap.parse_args()
    t0 = time.time()
    fig_dir, out_dir = Path(args.fig_dir), Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = 2 if args.quick else args.reps
    rhos = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    ns = [25, 200] if args.quick else [16, 32, 64, 128, 256, 512, 1024]
    fixed = dict(d=32, r=4, delta_norm=4.0, tau=3.0, sigma=1.0, n_cal=2000, n_test=2000, alpha=0.01)

    df_overlap = run_grid({"rho": rhos, "kappa": [1.0, 2.0]}, n_reps=reps, seed=1, n_fit=256, **fixed)
    df_n = run_grid({"n_fit": ns}, n_reps=reps, seed=2, rho=0.25, kappa=1.0, **fixed)
    df_overlap["sweep"], df_n["sweep"] = "overlap", "n_fit"
    pd.concat([df_overlap, df_n]).to_csv(out_dir / "theory_grid.csv", index=False)
    plot_theory(df_overlap, df_n, str(fig_dir / "theory_power_vs_overlap.png"),
                str(fig_dir / "theory_subspace_error_vs_n.png"))
    for kappa in (1.0, 2.0):
        fpr = shifted_fpr_mahalanobis_known(32, 4, 3.0, 1.0, kappa)
        print(f"known-covariance Mahalanobis FPR at kappa={kappa:g}: {fpr:.4f} (nominal 0.01)")
    print(f"wrote {out_dir / 'theory_grid.csv'} and figures in {fig_dir} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
