"""A0 - detector gate on planted activations with known mechanisms (runs before any actor data).

For each regime in :data:`actmon.synthetic.REGIMES` and ``reps`` independent
worlds, fit the activation detectors exactly as A2 does (normal fitting
split, development labels only for the probe, thresholds on a normal
calibration split), then score a test split. The gate passes when:

* ``null``: every detector's TPR and FPR are within Monte Carlo error of the
  target (calibration works, nothing leaks labels)
* ``base_shift``: pooled Mahalanobis detects the base anomaly. The response
  detector without its base term stays far below it. It is not exactly at alpha,
  because a displaced base also displaces the *predicted* response (the
  context model extrapolates), which is a legitimate residual
* ``context_shift``: task-conditioned Mahalanobis beats pooled Mahalanobis
* ``response_only``: the response detector beats every base-activation detector,
  and those stay near alpha
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actmon.utils.stats import wilson_interval
from actmon.utils.seeding import SeedBank

from actmon.detectors import build
from actmon.evaluation.calibrate import flags, thresholds
from actmon.experiments.common import log_to, save_table, write_json
from actmon.synthetic import REGIMES, SyntheticConfig, SyntheticWorld

DETECTORS = ("response", "response_plus_base", "mahalanobis_pooled", "mahalanobis_task", "mahalanobis_env",
             "linear_probe")


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    g = cfg.get("gate", {})
    reps = int(cfg.get("reps", g.get("reps", 5)))
    sizes = g.get("sizes", {"fit": 1000, "dev": [500, 500], "calib": 2000, "test": [2000, 300]})
    alpha = float(g.get("alpha", 0.05))
    scfg = SyntheticConfig(**g.get("synthetic", {}))
    bank = SeedBank(int(cfg["seed"])).child("a0")
    rows = []
    for regime in REGIMES:
        for r in range(reps):
            world = SyntheticWorld(scfg, bank.child("world", r).rng())
            rb = bank.child(regime, r)
            fit, _ = world.sample(rb.child("fit").rng(), sizes["fit"], 0, regime, "fit")
            dev, ydev = world.sample(rb.child("dev").rng(), *sizes["dev"], regime, "dev")
            cal, _ = world.sample(rb.child("calib").rng(), sizes["calib"], 0, regime, "calib")
            test, y = world.sample(rb.child("test").rng(), *sizes["test"], regime, "test")
            for name in DETECTORS:
                det = build(name)
                det.fit(fit, dev, ydev) if det.supervised else det.fit(fit)
                t = thresholds(det.score(cal), [alpha])[alpha]
                f = flags(det.score(test), t)
                rows.append({"regime": regime, "rep": r, "detector": name, "tpr": f[y == 1].mean(),
                             "fpr": f[y == 0].mean(), "n_viol": int((y == 1).sum()), "n_norm": int((y == 0).sum())})
        log(f"[A0] {regime} done")
    df = pd.DataFrame(rows)
    save_table(df, out, "gate_rows")
    summ = df.groupby(["regime", "detector"]).agg(tpr=("tpr", "mean"), tpr_sd=("tpr", "std"), fpr=("fpr", "mean"),
                                                   n_viol=("n_viol", "sum"), n_norm=("n_norm", "sum")).reset_index()
    save_table(summ, out, "gate_summary")
    T = summ.set_index(["regime", "detector"])
    n_v = int(summ["n_viol"].max())
    lo, hi = wilson_interval(int(round(alpha * n_v)), n_v, alpha=0.001)
    checks = {
        "null: all TPR within 99.9% band of alpha": bool(all(lo <= T.loc[("null", d), "tpr"] <= hi for d in DETECTORS)),
        "null: all FPR <= alpha + 0.01": bool(all(T.loc[("null", d), "fpr"] <= alpha + 0.01 for d in DETECTORS)),
        "base_shift: pooled TPR > 0.5": bool(T.loc[("base_shift", "mahalanobis_pooled"), "tpr"] > 0.5),
        "base_shift: response (no base term) far below pooled (< pooled - 0.5)": bool(
            T.loc[("base_shift", "response"), "tpr"] < T.loc[("base_shift", "mahalanobis_pooled"), "tpr"] - 0.5),
        "context_shift: task-conditioned > pooled + 0.2": bool(
            T.loc[("context_shift", "mahalanobis_task"), "tpr"] > T.loc[("context_shift", "mahalanobis_pooled"), "tpr"] + 0.2),
        "response_only: response > every base detector + 0.3": bool(all(
            T.loc[("response_only", "response"), "tpr"] > T.loc[("response_only", d), "tpr"] + 0.3
            for d in ("mahalanobis_pooled", "mahalanobis_task", "mahalanobis_env", "linear_probe"))),
    }
    checks["gate_passed"] = all(checks.values())
    write_json(checks, out / "gate_checks.json")
    log("[A0] summary:\n" + summ.to_string(index=False))
    log(f"[A0] checks: {checks}")
    return checks
