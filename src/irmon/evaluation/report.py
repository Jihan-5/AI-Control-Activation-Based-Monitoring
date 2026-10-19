"""Result tables and figures.

Every artefact written from a synthetic-backend run carries
``irmon.EVIDENCE_STATUS`` in its header so pipeline-validation output can never
be mistaken for a measured result. Tables are written as CSV (machine) and
Markdown (human); figures use the Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from irmon import EVIDENCE_STATUS

from . import metrics as M
from . import stats as S

SYNTHETIC_BANNER = (
    "SYNTHETIC BACKEND - pipeline validation only, not evidence about language models. "
    + EVIDENCE_STATUS
)


def _banner(synthetic: bool) -> str:
    return SYNTHETIC_BANNER if synthetic else "Measured on the configured backend; see run manifest for provenance."


def _to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Dependency-free Markdown table (no ``tabulate`` requirement)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else format(float(v), floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, path: str | Path, title: str, synthetic: bool, note: str = "") -> None:
    """Write ``path.csv`` and ``path.md`` with the evidence banner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    banner = _banner(synthetic)
    with open(path.with_suffix(".csv"), "w") as fh:
        fh.write(f"# {banner}\n")
        df.to_csv(fh, index=False)
    md = [f"### {title}", "", f"> {banner}", ""]
    if note:
        md += [note, ""]
    md.append(_to_markdown(df))
    path.with_suffix(".md").write_text("\n".join(md) + "\n")


# ------------------------------------------------------------------ tables
def operating_points(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    thresholds: dict[str, dict[float, float]],
    groups=None,
    prevalences=(0.01, 0.001),
) -> pd.DataFrame:
    return M.operating_point_table(scores, y, thresholds, groups=groups, prevalences=prevalences)


def per_group_fpr_table(scores: dict[str, np.ndarray], y, groups, thresholds: dict[str, dict[float, float]]) -> pd.DataFrame:
    y = np.asarray(y, int)
    g = np.asarray(groups)
    rows = []
    for m, s in scores.items():
        s = np.asarray(s, float)
        for alpha, thr in sorted(thresholds[m].items()):
            for grp in sorted(pd.unique(g[y == 0]).tolist(), key=str):
                mask = (g == grp) & (y == 0)
                k, n = int(np.sum(s[mask] > thr)), int(mask.sum())
                lo, hi = S.clopper_pearson(k, n)
                rows.append({"method": m, "nominal_fpr": alpha, "group": grp, "n": n,
                             "false_alarms": k, "achieved_fpr": k / n if n else np.nan,
                             "ci_low": lo, "ci_high": hi, "ratio_to_nominal": (k / n) / alpha if n else np.nan})
    return pd.DataFrame(rows)


def paired_comparison_table(
    flags: dict[str, np.ndarray],
    y,
    clusters,
    primary: tuple[str, str],
    secondary: list[tuple[str, str]],
    n_boot: int = 10000,
    seed: int = 0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired TPR differences at calibrated thresholds.

    ``flags[m]`` are boolean flags at the predeclared nominal FPR. The primary
    comparison is tested at the nominal level; secondaries get Holm (or
    Bonferroni) adjusted McNemar p-values. CIs: paired cluster bootstrap.
    """
    y = np.asarray(y, int)
    pos = y == 1
    rows = []
    for kind, pairs in (("primary", [primary]), ("secondary", secondary)):
        for a, b in pairs:
            boot = S.paired_cluster_bootstrap(S.tpr_difference, flags[a], flags[b], y, clusters, n_boot=n_boot, seed=seed)
            mc = S.mcnemar_from_flags(flags[a][pos], flags[b][pos])
            fa = S.paired_cluster_bootstrap(S.fpr_difference, flags[a], flags[b], y, clusters, n_boot=max(1000, n_boot // 5), seed=seed + 1)
            rows.append({
                "kind": kind, "method": a, "baseline": b,
                "tpr_method": float(flags[a][pos].mean()), "tpr_baseline": float(flags[b][pos].mean()),
                "tpr_diff": boot.estimate, "ci_low": boot.ci_low, "ci_high": boot.ci_high,
                "fpr_diff": fa.estimate, "fpr_diff_ci_low": fa.ci_low, "fpr_diff_ci_high": fa.ci_high,
                "method_only": mc.a_only, "baseline_only": mc.b_only, "mcnemar_p": mc.p_value,
                "n_clusters": boot.n_clusters,
            })
    df = pd.DataFrame(rows)
    sec = df["kind"] == "secondary"
    if sec.any() and correction != "none":
        adj = (S.holm_bonferroni if correction == "holm" else S.bonferroni)(df.loc[sec, "mcnemar_p"].to_numpy())
        df.loc[sec, "p_adjusted"] = adj.iloc[:, 1].to_numpy()
    df.loc[~sec, "p_adjusted"] = df.loc[~sec, "mcnemar_p"]
    return df


def ablation_table(results: dict[str, dict]) -> pd.DataFrame:
    """``results[ablation_name] = {metric: value, ...}`` -> table with deltas
    relative to the ``full`` row."""
    df = pd.DataFrame(results).T
    df.index.name = "ablation"
    if "full" in df.index:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[f"delta_{c}"] = df[c] - df.loc["full", c]
    return df.reset_index()


def access_table(detectors: dict) -> pd.DataFrame:
    return pd.DataFrame([{"method": n, **d.access.row()} for n, d in detectors.items()])


def control_table(summaries: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(summaries)


# ----------------------------------------------------------------- figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_roc_pr(scores: dict[str, np.ndarray], y, path: str | Path, synthetic: bool) -> None:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, s in scores.items():
        fpr, tpr, _ = M.roc_curve_points(s, y)
        a1.plot(fpr, tpr, lw=1.2, label=f"{m} (AUROC {M.auroc(s, y):.3f})")
        rec, prec = M.pr_curve_points(s, y)
        a2.plot(rec, prec, lw=1.2, label=f"{m} (AP {M.average_precision(s, y):.3f})")
    a1.plot([0, 1], [0, 1], "k:", lw=0.7)
    a1.set_xscale("symlog", linthresh=0.01)
    a1.axvline(0.01, color="grey", lw=0.6, ls="--")
    a1.axvline(0.05, color="grey", lw=0.6, ls="--")
    a1.set_xlabel("false-positive rate (symlog; dashed: 1%, 5%)")
    a1.set_ylabel("true-positive rate")
    a2.set_xlabel("recall")
    a2.set_ylabel("precision at BENCHMARK prevalence (not deployment)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(_banner(synthetic), fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_fpr(df: pd.DataFrame, path: str | Path, synthetic: bool, nominal: float = 0.01) -> None:
    """Achieved vs nominal FPR per held-out normal group (from per_group_fpr_table)."""
    plt = _plt()
    d = df[np.isclose(df["nominal_fpr"], nominal)]
    methods = list(pd.unique(d["method"]))
    groups = list(pd.unique(d["group"]))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups) * max(1, len(methods)) / 2), 4.2))
    width = 0.8 / max(1, len(methods))
    for j, m in enumerate(methods):
        dm = d[d["method"] == m].set_index("group").reindex(groups)
        x = np.arange(len(groups)) + j * width
        err = np.vstack([dm["achieved_fpr"] - dm["ci_low"], dm["ci_high"] - dm["achieved_fpr"]])
        ax.bar(x, dm["achieved_fpr"], width, yerr=err, capsize=2, label=m)
    ax.axhline(nominal, color="k", ls=":", lw=0.8, label=f"nominal {nominal:g}")
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("achieved FPR (Clopper-Pearson 95%)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(_banner(synthetic), fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_curve(df: pd.DataFrame, path: str | Path, synthetic: bool, metric: str = "tpr") -> None:
    """Detection vs forward-equivalent cost (from cost.cost_matched_curve)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, dm in df.groupby("method"):
        dm = dm.sort_values("forwards_per_root")
        ax.plot(dm["forwards_per_root"], dm[metric], marker="o", label=m)
    ax.set_xlabel("forward-equivalents per monitored root")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(_banner(synthetic), fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
