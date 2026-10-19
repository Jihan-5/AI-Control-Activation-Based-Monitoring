"""A2 - main detection comparison on the locked test split (and, separately, held-out families).

Steps (``docs/EXPERIMENT_PLAN.md`` §5):

1. Fit every detector on the 1,000-root normal-fitting split. Supervised
   baselines (linear probe, behaviour logistic) also get the development split
   and its labels, and nothing else does.
2. Set thresholds on the 5,000-root normal calibration split for 1 %, 2 % and
   5 % FPR.
3. Score the locked test split (5,000 normal + 500 violating). Report TPR,
   achieved FPR, Wilson intervals, AUROC/AP, and precision and alarm fraction
   at stated prevalences.
4. Paired comparisons of the proposed response detector with every comparator
   on the same violating roots: exact McNemar, the i.i.d. and family-clustered
   paired TPR difference, the false-alarm difference, and Holm over the
   predeclared family.
5. Group results by mechanism, broken rule and cue presence. FPR on normal
   roots with and without the cue.
6. The held-out-family partition is scored with the *same* thresholds and
   reported separately. Exchangeability with calibration does not hold there,
   so its achieved FPR is itself a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actmon.detectors import PROPOSED
from actmon.evaluation.metrics import ambiguous_action_strings, detection_table, group_table, paired_tests
from actmon.experiments.common import calibrate, fit_detectors, log_to, save_table, score_all, setup, write_json


def screened_prevalence(work: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(work) / "data").glob("pool_*.csv.gz")):
        df = pd.read_csv(p)
        split = p.name[len("pool_"):-len(".csv.gz")]
        rows.append({"split": split, "screened": len(df), "screened_violation_rate": 1 - df["valid"].mean(),
                     "selected": int(df["selected"].sum()),
                     "selected_violation_rate": 1 - df.loc[df.selected, "valid"].mean()})
    return pd.DataFrame(rows)


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    m2, m1, meta, data = setup(cfg, log)
    ev = cfg["evaluation"]
    alphas = [float(a) for a in ev["fpr"]]
    names = cfg["detectors"]["main"]
    dets, access = fit_detectors(names, data, cfg["detectors"].get("params"), log)
    save_table(access, out, "access")
    thr = calibrate(dets, data["calib"][0], alphas)
    write_json(thr, out / "thresholds.json")

    summary = {}
    for split in ("test", "test_heldout"):
        if split not in data:
            continue
        fs, lab = data[split]
        y = lab.y
        scores, secs = score_all(dets, fs)
        np.savez_compressed(out / f"scores_{split}.npz", y=y, root_ids=fs.root_ids, **scores)
        det_tab = detection_table(scores, y, thr, ev.get("prevalences", (0.1, 0.01, 0.001)))
        save_table(det_tab, out, f"detection_{split}")
        comps = [c for c in cfg["detectors"]["comparators"] if c in scores]
        pt = paired_tests(scores, y, thr, PROPOSED, comps, clusters=fs.family)
        save_table(pt, out, f"paired_{split}", floatfmt=".4g")
        grp = pd.concat([group_table(scores, lab.table, thr, a) for a in alphas], ignore_index=True)
        save_table(grp, out, f"groups_{split}")
        summary[split] = {
            "n_normal": int((y == 0).sum()), "n_violating": int((y == 1).sum()),
            "n_families": int(len(set(fs.family))),
            "tpr": det_tab.pivot(index="method", columns="target_fpr", values="tpr").round(4).to_dict(),
            "fpr": det_tab.pivot(index="method", columns="target_fpr", values="fpr").round(4).to_dict(),
            "auroc": det_tab.groupby("method")["auroc"].first().round(4).to_dict(),
            "score_ms_per_1000": secs,
        }
        log(f"[A2] {split}: TPR at 5% FPR " + ", ".join(
            f"{m}={v:.3f}" for m, v in summary[split]["tpr"].get(0.05, {}).items()))
    amb = ambiguous_action_strings(data["test"][1].table)
    save_table(amb, out, "ambiguous_action_strings")
    prev = screened_prevalence(Path(cfg["work_dir"]))
    save_table(prev, out, "screened_prevalence")
    mech = data["test"][1].table.query("valid == False")["mechanism"].value_counts().rename_axis(
        "mechanism").reset_index(name="n")
    save_table(mech, out, "test_violation_mechanisms")
    summary["ambiguous_action_strings"] = int(len(amb))
    summary["hook_check"] = meta.get("hook_check")
    write_json(summary, out / "summary.json")
    _figure(out, thr, data, dets)
    return summary


def _figure(out: Path, thr, data, dets):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:  # pragma: no cover
        return
    z = np.load(out / "scores_test.npz")
    y = z["y"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in dets:
        f, t, _ = roc_curve(y, z[name])
        lw = 2.4 if name == PROPOSED else 1.2
        ax[0].plot(f, t, lw=lw, label=name)
        ax[1].plot(f, t, lw=lw, label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC, locked test split")
    ax[1].set(xlim=(0, 0.06), xlabel="false-positive rate", title="Low-FPR region (1-5 % targets)")
    for a in (0.01, 0.02, 0.05):
        ax[1].axvline(a, color="0.7", lw=0.8, ls="--")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=150)
    plt.close(fig)
