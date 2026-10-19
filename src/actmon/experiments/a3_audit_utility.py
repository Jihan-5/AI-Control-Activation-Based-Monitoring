"""A3 - fixed audit budget, benign utility under monitoring, and the partial-guard reference row.

* Audit: rank the separate 1,000-root audit cohort (900 normal + 100 violating)
  and inspect the top 10, 20 and 50 roots. Report violations found, recall and
  precision, plus paired McNemar tests on the violating roots for the proposed
  detector against each comparator.
* Utility: on the separate 500-root benign set, compare completion of the
  authorized task with no monitor against a monitored protocol that blocks
  proposals flagged at the 1/2/5 % FPR thresholds.
* Guard: replay the partial reserve+handoff guard on the recorded test
  proposals (interception by rule, false blocking).
* Alarm budget implied by the main test's FPR operating points:
  ``(TPR x n_violating + FPR x n_normal) / n``. This shows that a 5 % FPR is
  not a 5 % audit budget.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from actmon.detectors import PROPOSED
from actmon.evaluation.audit import audit_paired, audit_table, utility_table
from actmon.evaluation.calibrate import flags
from actmon.evaluation.guard import guard_table
from actmon.experiments.common import calibrate, fit_detectors, log_to, save_table, score_all, setup, write_json


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    m2, m1, meta, data = setup(cfg, log)
    ev = cfg["evaluation"]
    alphas = [float(a) for a in ev["fpr"]]
    ks = [int(k) for k in ev["audit_k"]]
    dets, access = fit_detectors(cfg["detectors"]["main"], data, cfg["detectors"].get("params"), log)
    save_table(access, out, "access")
    thr = calibrate(dets, data["calib"][0], alphas)

    afs, alab = data["audit"]
    a_scores, _ = score_all(dets, afs)
    aud = audit_table(a_scores, alab.y, ks)
    save_table(aud, out, "audit")
    comps = [c for c in cfg["detectors"]["comparators"] if c in a_scores]
    save_table(audit_paired(a_scores, alab.y, PROPOSED, comps, ks), out, "audit_paired", floatfmt=".4g")

    ufs, ulab = data["utility"]
    u_scores, _ = score_all(dets, ufs)
    completes = ulab.table["completes"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    util = utility_table(u_scores, completes, thr)
    save_table(util, out, "utility")

    guard = guard_table(data["test"][1].table)
    save_table(guard, out, "guard_test")

    tfs, tlab = data["test"]
    t_scores, _ = score_all(dets, tfs)
    rows = []
    for name, s in t_scores.items():
        for a in alphas:
            f = flags(s, thr[name][a])
            rows.append({"method": name, "target_fpr": a, "flagged": int(f.sum()), "n": int(len(f)),
                         "alarm_fraction": float(f.mean())})
    alarm = pd.DataFrame(rows)
    save_table(alarm, out, "alarm_fraction_test")

    summary = {
        "audit_recall": aud.pivot(index="method", columns="k", values="recall").round(4).to_dict(),
        "utility_completion": {f"{m}@{a:g}" if a == a else m: round(float(c), 4)
                               for m, a, c in util[["method", "target_fpr", "completion"]].itertuples(index=False)},
        "guard_interception_all": float(guard.iloc[0]["rate"]),
    }
    write_json({str(k): v for k, v in summary.items()}, out / "summary.json")
    log(f"[A3] audit recall@50: " + ", ".join(f"{m}={v:.2f}" for m, v in summary["audit_recall"].get(50, {}).items()))
    return summary
