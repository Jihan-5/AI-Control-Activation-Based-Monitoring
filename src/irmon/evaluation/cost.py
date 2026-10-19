"""Inference-cost accounting and cost-matched comparisons (plan p.8 "Cost fairness").

The proposed method spends extra forwards (one per edit, plus candidate
scoring). A weakly provisioned baseline is not a fair comparison, so every
detector declares ``AccessSpec.forwards_per_root`` and results are plotted as
detection versus forward-equivalent cost.

Forward-equivalent: one full forward pass of the monitored actor over one
prompt. Candidate scoring over K short continuations is counted from the
bundle's own ``cost`` record when present (backends may fuse it with the
prompt forward through KV reuse).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def detector_cost(access, bundle_cost: dict | None = None) -> float:
    """Forward-equivalents per monitored root for a detector."""
    return float(getattr(access, "forwards_per_root", 1.0))


def bundle_cost_total(bundle_cost: dict, families: Iterable[str] | None = None) -> float:
    """Sum the per-root cost of the feature families a detector reads."""
    keys = list(families) if families is not None else list(bundle_cost)
    return float(sum(bundle_cost.get(k, 0.0) for k in keys))


def access_cost_table(detectors: dict) -> pd.DataFrame:
    """One row per detector: its declared access and cost."""
    rows = []
    for name, det in detectors.items():
        row = {"method": name, **det.access.row()}
        rows.append(row)
    return pd.DataFrame(rows)


def cost_matched_curve(
    variants: dict[str, tuple[float, Callable[[], dict]]],
) -> pd.DataFrame:
    """Evaluate detector variants at different query budgets.

    ``variants`` maps a label (e.g. ``"proposed[T2+U1]"``, ``"strip[M=8]"``) to
    ``(forwards_per_root, evaluate_fn)`` where ``evaluate_fn()`` returns a dict
    of metrics (e.g. ``{"tpr@1%": .., "auroc": ..}``) computed on the dev or
    locked split. The caller builds variants with matched budgets: subsets of
    edit slots for the proposed method, matching numbers of perturbations for
    STRIP/TeCo/output-only, and matching sample counts for functional
    attribution.
    """
    rows = []
    for label, (cost, fn) in variants.items():
        metrics = fn()
        method = label.split("[")[0]
        rows.append({"variant": label, "method": method, "forwards_per_root": float(cost), **metrics})
    return pd.DataFrame(rows).sort_values(["method", "forwards_per_root"]).reset_index(drop=True)


def pareto_front(df: pd.DataFrame, cost_col: str = "forwards_per_root", metric_col: str = "tpr") -> pd.DataFrame:
    """Variants not dominated (lower-or-equal cost and higher metric)."""
    d = df.sort_values([cost_col, metric_col], ascending=[True, False])
    best = -np.inf
    keep = []
    for idx, row in d.iterrows():
        if row[metric_col] > best:
            keep.append(idx)
            best = row[metric_col]
    return d.loc[keep]
