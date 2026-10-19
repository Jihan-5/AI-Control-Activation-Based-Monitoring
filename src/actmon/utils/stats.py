"""(Shared utility, originally written for the companion polysemanticity study.)

Evaluation statistics (syllabus S04 Miller; P1 Wasserman; X8 Dietterich).

* Wilson score interval for a proportion (coverage and its Monte Carlo
  uncertainty, plan p.8: "Report empirical coverage with its Monte Carlo
  uncertainty").
* Cluster bootstrap and clustered standard error for means and paired
  differences — resample whole clusters (datasets, documents, template
  families), never individual items inside a cluster.
* Exact McNemar test on paired binary outcomes.
* Holm step-down correction for predeclared families of comparisons.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def clustered_mean_se(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    """Mean and cluster-robust SE (Miller 2024, eq. for clustered SE)."""
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(clusters, return_inverse=True)
    G = inv.max() + 1
    mean = values.mean()
    n = values.size
    s = np.zeros(G)
    np.add.at(s, inv, values - mean)
    var = (G / max(G - 1, 1)) * np.sum(s**2) / n**2
    return float(mean), float(math.sqrt(var)), int(G)


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                         stat=np.mean, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile CI for ``stat(values)`` resampling whole clusters."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    est = float(stat(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([members[g] for g in pick])
        boots[b] = stat(values[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def paired_difference(a: np.ndarray, b: np.ndarray, clusters: np.ndarray | None = None, alpha: float = 0.05) -> dict:
    """Paired mean difference ``a - b`` with clustered normal CI (Miller-style)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    clusters = np.arange(diff.size) if clusters is None else clusters
    mean, se, G = clustered_mean_se(diff, clusters)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"diff": mean, "se": se, "lo": mean - z * se, "hi": mean + z * se, "n_clusters": G, "n": int(diff.size)}


def mcnemar_exact(both: int, a_only: int, b_only: int, neither: int) -> float:
    """Two-sided exact McNemar p-value from the discordant cells."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))


def holm(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
