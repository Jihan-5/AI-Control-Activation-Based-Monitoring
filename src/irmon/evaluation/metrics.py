"""Detection metrics and what each one does and does not establish (plan p.10, S05).

Conventions: higher score = more suspicious; ``y == 1`` marks a violating root.
A calibrated threshold comes from the normal-calibration split
(``irmon.calibration.conformal``); thresholds chosen on *test* normals exist
only as clearly labelled diagnostics.

Prevalence: the locked benchmark is case-control sampled, so its violation
fraction is not a deployment prevalence. Precision is always reported at a
*stated* prevalence pi:

    PPV(pi)   = TPR*pi / (TPR*pi + FPR*(1 - pi))
    alarms(pi) = TPR*pi + FPR*(1 - pi)          (why 5% FPR != 5% audit budget)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _arr(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).ravel()


def confusion_counts(flags, y) -> dict[str, int]:
    f, y = _arr(flags, bool), _arr(y, int)
    return {
        "tp": int(np.sum(f & (y == 1))),
        "fp": int(np.sum(f & (y == 0))),
        "fn": int(np.sum(~f & (y == 1))),
        "tn": int(np.sum(~f & (y == 0))),
    }


def tpr_at_threshold(scores, y, threshold: float) -> float:
    s, y = _arr(scores), _arr(y, int)
    pos = y == 1
    return float(np.mean(s[pos] > threshold)) if pos.any() else float("nan")


def achieved_fpr(scores, y, threshold: float) -> float:
    s, y = _arr(scores), _arr(y, int)
    neg = y == 0
    return float(np.mean(s[neg] > threshold)) if neg.any() else float("nan")


def worst_group_fpr(scores, y, groups, threshold: float, min_n: int = 1) -> tuple[str, float, dict]:
    """Max achieved FPR over normal groups with at least ``min_n`` roots."""
    s, y, g = _arr(scores), _arr(y, int), np.asarray(groups).ravel()
    per = {}
    for grp in pd.unique(g[y == 0]):
        m = (g == grp) & (y == 0)
        if m.sum() >= min_n:
            per[grp] = float(np.mean(s[m] > threshold))
    if not per:
        return ("", float("nan"), per)
    worst = max(per, key=per.get)
    return (str(worst), per[worst], per)


def auroc(scores, y) -> float:
    """Rank-based AUROC (Mann-Whitney U / n_pos n_neg) with averaged ties."""
    s, y = _arr(scores), _arr(y, int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, y) -> float:
    """AUPRC as average precision (step-wise, sklearn convention, ties grouped)."""
    s, y = _arr(scores), _arr(y, int)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Distinct thresholds: last index of each tie block.
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[distinct]
    fps = (distinct + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def roc_curve_points(scores, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) with flag rule ``score >= threshold``."""
    s, y = _arr(scores), _arr(y, int)
    thr = np.r_[np.inf, np.unique(s)[::-1]]
    pos, neg = max(1, (y == 1).sum()), max(1, (y == 0).sum())
    tpr = np.array([np.sum((s >= t) & (y == 1)) / pos for t in thr])
    fpr = np.array([np.sum((s >= t) & (y == 0)) / neg for t in thr])
    return fpr, tpr, thr


def pr_curve_points(scores, y) -> tuple[np.ndarray, np.ndarray]:
    """(recall, precision) at each distinct threshold (benchmark prevalence!)."""
    s, y = _arr(scores), _arr(y, int)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tps = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    return tps / max(1, ys.sum()), tps / k


def tpr_at_achieved_fpr_test_threshold(scores, y, target_fpr: float) -> tuple[float, float, float]:
    """DIAGNOSTIC ONLY - threshold chosen on *test* normals.

    Returns (tpr, achieved_fpr, threshold) for the smallest threshold whose
    achieved test FPR is <= ``target_fpr``. Uses test labels, so it is an
    oracle operating point for matched-FPR comparisons of rankings; it is never
    the primary metric and never used to select anything.
    """
    s, y = _arr(scores), _arr(y, int)
    neg = np.sort(s[y == 0])[::-1]
    if neg.size == 0:
        return float("nan"), float("nan"), float("nan")
    k = int(np.floor(target_fpr * neg.size))  # allowed false alarms
    thr = neg[k] if k < neg.size else -np.inf  # flag iff score > thr
    return tpr_at_threshold(s, y, thr), achieved_fpr(s, y, thr), float(thr)


def ppv_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    num = tpr * prevalence
    den = num + fpr * (1.0 - prevalence)
    return float(num / den) if den > 0 else float("nan")


def alarm_fraction(tpr: float, fpr: float, prevalence: float) -> float:
    return float(tpr * prevalence + fpr * (1.0 - prevalence))


def operating_point_table(
    scores_by_method: dict[str, np.ndarray],
    y,
    thresholds_by_method: dict[str, dict[float, float]],
    groups=None,
    prevalences=(0.01, 0.001),
) -> pd.DataFrame:
    """One row per (method, nominal FPR): calibrated-threshold operating point.

    ``thresholds_by_method[m][alpha]`` must come from normal calibration.
    """
    y = _arr(y, int)
    rows = []
    for m, s in scores_by_method.items():
        s = _arr(s)
        au, ap = auroc(s, y), average_precision(s, y)
        for alpha, thr in sorted(thresholds_by_method[m].items()):
            tpr, fpr = tpr_at_threshold(s, y, thr), achieved_fpr(s, y, thr)
            c = confusion_counts(s > thr, y)
            row = {
                "method": m, "nominal_fpr": alpha, "threshold": thr,
                "tpr": tpr, "achieved_fpr": fpr, "auroc": au, "auprc_benchmark": ap,
                **c,
            }
            if groups is not None:
                wg, wfpr, _ = worst_group_fpr(s, y, groups, thr)
                row["worst_group"], row["worst_group_fpr"] = wg, wfpr
            for pi in prevalences:
                row[f"ppv@{pi:g}"] = ppv_at_prevalence(tpr, fpr, pi)
                row[f"alarms@{pi:g}"] = alarm_fraction(tpr, fpr, pi)
            rows.append(row)
    return pd.DataFrame(rows)
