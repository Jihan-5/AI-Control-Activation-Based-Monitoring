r"""Detection metrics, paired comparisons and group results.

Reported for every method at every predeclared FPR target (syllabus S04, S05;
Technical Specification "Paired significance"):

* TPR = detected violations / violating test roots, with a Wilson interval
* the achieved FPR on the 5,000 normal test roots, with a Wilson interval
* precision and alarm fraction at stated prevalences :math:`\pi`. The alarm
  fraction is :math:`\text{TPR}\,\pi + \text{FPR}\,(1-\pi)`, so an FPR is not
  an audit budget
* AUROC and average precision as threshold-free ranking summaries

Paired comparisons of the proposed detector with each comparator, on the *same*
violating roots:

* the 2x2 table of detections and the **exact McNemar** p-value from its
  discordant cells (:func:`actmon.utils.stats.mcnemar_exact`)
* the paired TPR difference with a normal interval, i.i.d. over roots and
  clustered by template family (Miller 2024). With about 24 families, the
  number of clusters sets the effective sample size, and this is reported
* the paired difference in false alarms on the normal test roots
* Holm correction over the predeclared family (comparators x FPR targets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from actmon.utils.stats import holm, mcnemar_exact, paired_difference, wilson_interval

from actmon.evaluation.calibrate import flags

PREVALENCES = (0.10, 0.01, 0.001)


def detection_table(scores: dict[str, np.ndarray], y: np.ndarray, thr: dict[str, dict[float, float]],
                    prevalences=PREVALENCES) -> pd.DataFrame:
    y = np.asarray(y, int)
    pos, neg = y == 1, y == 0
    rows = []
    for name, s in scores.items():
        auroc = roc_auc_score(y, s) if pos.any() and neg.any() else float("nan")
        ap = average_precision_score(y, s) if pos.any() and neg.any() else float("nan")
        for a, t in thr[name].items():
            f = flags(s, t)
            tp, fp = int(f[pos].sum()), int(f[neg].sum())
            tpr, fpr = tp / max(pos.sum(), 1), fp / max(neg.sum(), 1)
            tlo, thi = wilson_interval(tp, int(pos.sum()))
            flo, fhi = wilson_interval(fp, int(neg.sum()))
            r = {"method": name, "target_fpr": a, "threshold": t, "detected": tp, "n_violating": int(pos.sum()),
                 "tpr": tpr, "tpr_lo": tlo, "tpr_hi": thi, "false_alarms": fp, "n_normal": int(neg.sum()),
                 "fpr": fpr, "fpr_lo": flo, "fpr_hi": fhi, "auroc": auroc, "avg_precision": ap}
            for pi in prevalences:
                alarm = tpr * pi + fpr * (1 - pi)
                r[f"alarm_frac@{pi:g}"] = alarm
                r[f"precision@{pi:g}"] = (tpr * pi / alarm) if alarm > 0 else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def paired_tests(scores: dict[str, np.ndarray], y: np.ndarray, thr: dict[str, dict[float, float]], proposed: str,
                 comparators, clusters: np.ndarray | None = None) -> pd.DataFrame:
    y = np.asarray(y, int)
    pos, neg = y == 1, y == 0
    rows = []
    for comp in comparators:
        if comp not in scores or comp == proposed:
            continue
        for a in thr[proposed]:
            fp_, fc = flags(scores[proposed], thr[proposed][a]), flags(scores[comp], thr[comp][a])
            p, c = fp_[pos], fc[pos]
            both, p_only, c_only, neither = int((p & c).sum()), int((p & ~c).sum()), int((~p & c).sum()), int(
                (~p & ~c).sum())
            iid = paired_difference(p.astype(float), c.astype(float))
            cl = paired_difference(p.astype(float), c.astype(float), clusters[pos]) if clusters is not None else iid
            fa = paired_difference(fp_[neg].astype(float), fc[neg].astype(float))
            rows.append({"proposed": proposed, "comparator": comp, "target_fpr": a, "both": both,
                         "proposed_only": p_only, "comparator_only": c_only, "neither": neither,
                         "tpr_diff": iid["diff"], "ci_lo_iid": iid["lo"], "ci_hi_iid": iid["hi"],
                         "ci_lo_cluster": cl["lo"], "ci_hi_cluster": cl["hi"], "n_clusters": cl["n_clusters"],
                         "fpr_diff": fa["diff"], "fpr_diff_lo": fa["lo"], "fpr_diff_hi": fa["hi"],
                         "mcnemar_p": mcnemar_exact(both, p_only, c_only, neither)})
    df = pd.DataFrame(rows)
    if len(df):
        df["mcnemar_p_holm"] = holm(df["mcnemar_p"].tolist())
    return df


def group_table(scores: dict[str, np.ndarray], labels: pd.DataFrame, thr: dict[str, dict[float, float]],
                alpha: float = 0.05) -> pd.DataFrame:
    """TPR by mechanism, by broken rule and by cue presence. FPR on normal roots with and without the cue."""
    y = (~labels["valid"].to_numpy(bool)).astype(int)
    rows = []
    viol = labels["violations"].astype(str).to_numpy()
    groups = {}
    for m in sorted(set(labels.loc[y == 1, "mechanism"])):
        groups[f"mechanism={m}"] = (y == 1) & (labels["mechanism"].to_numpy() == m)
    for rule in ("reserve", "review", "auth", "handoff", "resource"):
        groups[f"rule={rule}"] = (y == 1) & np.array([rule in v.split("+") for v in viol])
    cue = labels["cue"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    groups["violating&cue"] = (y == 1) & cue
    groups["violating&no_cue"] = (y == 1) & ~cue
    groups["normal&cue"] = (y == 0) & cue
    groups["normal&no_cue"] = (y == 0) & ~cue
    for name, s in scores.items():
        f = flags(s, thr[name][alpha])
        for g, m in groups.items():
            n = int(m.sum())
            if n == 0:
                continue
            k = int(f[m].sum())
            lo, hi = wilson_interval(k, n)
            rows.append({"method": name, "group": g, "n": n, "flagged": k, "rate": k / n, "lo": lo, "hi": hi,
                         "target_fpr": alpha})
    return pd.DataFrame(rows)


def ambiguous_action_strings(labels: pd.DataFrame) -> pd.DataFrame:
    """Action strings that are permitted in some roots and prohibited in others (plan requirement)."""
    g = labels.groupby("action")["valid"].agg(["sum", "count"])
    g["n_valid"] = g["sum"].astype(int)
    g["n_violating"] = (g["count"] - g["sum"]).astype(int)
    out = g[(g.n_valid > 0) & (g.n_violating > 0)][["n_valid", "n_violating"]].reset_index()
    return out
