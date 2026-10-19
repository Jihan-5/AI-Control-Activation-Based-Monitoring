"""Paired and clustered uncertainty for monitor comparisons (plan p.9; S04, P1, X8).

Independence model
------------------
The resampling unit is the *root* (or a cluster of roots: an outcome-matched
pair, or a task family when families are sampled). Edits of one root, tokens of
one prompt, and layers of one forward are never independent observations;
they live inside one root's score. Two monitors scored on the same roots are
*paired*, so a comparison must use the paired difference, not two marginal
intervals (spec p.4: overlapping Wilson intervals are not a paired test).

With few clusters (e.g. a handful of task families) the number of clusters,
not roots, is the effective sample size and normal-approximation intervals
misbehave; report the cluster count beside every interval. Training-seed
variation is a separate variance source and is reported via ``seed_spread``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------ proportions
def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------- McNemar
@dataclass(frozen=True)
class McNemarResult:
    both: int
    a_only: int
    b_only: int
    neither: int
    p_value: float
    statistic: int  # min(a_only, b_only)

    @property
    def n(self) -> int:
        return self.both + self.a_only + self.b_only + self.neither

    @property
    def difference(self) -> float:
        """Marginal rate difference (A - B) implied by the table."""
        return (self.a_only - self.b_only) / self.n if self.n else float("nan")


def mcnemar_exact(both: int, a_only: int, b_only: int, neither: int) -> McNemarResult:
    """Exact two-sided McNemar test: binomial test of a_only vs b_only at p=1/2.

    Only discordant cells carry information about the paired difference; the
    same marginal rates can correspond to very different p-values.
    """
    nd = a_only + b_only
    if nd == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(a_only, nd, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(both, a_only, b_only, neither, min(1.0, p), min(a_only, b_only))


def mcnemar_from_flags(flags_a, flags_b) -> McNemarResult:
    a, b = np.asarray(flags_a, bool), np.asarray(flags_b, bool)
    return mcnemar_exact(int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)), int(np.sum(~a & ~b)))


# --------------------------------------------------------- multiple tests
def holm_bonferroni(pvals, alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down adjusted p-values (strong FWER control).

    Applied to the predeclared *secondary* comparisons only; the single
    primary comparison is tested at the nominal level.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return pd.DataFrame({"p": p, "p_holm": adj, "reject": adj <= alpha})


def bonferroni(pvals, alpha: float = 0.05) -> pd.DataFrame:
    p = np.asarray(pvals, dtype=float)
    adj = np.minimum(1.0, p * p.size)
    return pd.DataFrame({"p": p, "p_bonferroni": adj, "reject": adj <= alpha})


# --------------------------------------------------------------- bootstrap
@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_boot: int
    n_clusters: int
    level: float
    samples: np.ndarray

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "se": self.se, "n_boot": self.n_boot, "n_clusters": self.n_clusters, "level": self.level,
        }


def _cluster_index(clusters) -> tuple[np.ndarray, list[np.ndarray]]:
    c = np.asarray(clusters)
    uniq, inv = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return uniq, members


def paired_cluster_bootstrap(
    stat_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    a,
    b,
    y,
    clusters=None,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap ``stat_fn(a[idx], b[idx], y[idx])`` resampling whole clusters.

    ``a``/``b`` are per-root scores or flags of two monitors on the SAME roots;
    ``stat_fn`` usually returns a paired difference, e.g.
    ``lambda a, b, y: a[y == 1].mean() - b[y == 1].mean()`` for TPR difference
    at fixed calibrated thresholds (a, b are flags). Thresholds are held fixed:
    they come from the calibration split, which is not resampled here.

    Percentile interval; SE is the bootstrap standard deviation. Resamples
    with no positives/negatives produce NaN and are dropped (counted in
    ``n_boot``).
    """
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y)
    if clusters is None:
        clusters = np.arange(len(y))
    _, members = _cluster_index(clusters)
    k = len(members)
    rng = np.random.default_rng(seed)
    est = float(stat_fn(a, b, y))
    # Precompute concatenation via lengths for speed.
    lens = np.array([len(m) for m in members])
    flat = np.concatenate(members)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    samples = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([flat[starts[j]: starts[j] + lens[j]] for j in pick]) if lens.max() > 1 else flat[pick]
        with np.errstate(invalid="ignore", divide="ignore"):
            samples[i] = stat_fn(a[idx], b[idx], y[idx])
    good = samples[np.isfinite(samples)]
    alpha = 1 - level
    lo, hi = (np.quantile(good, [alpha / 2, 1 - alpha / 2]) if good.size else (np.nan, np.nan))
    return BootstrapResult(est, float(lo), float(hi), float(good.std(ddof=1)) if good.size > 1 else float("nan"),
                           n_boot, k, level, good)


def tpr_difference(flags_a, flags_b, y) -> float:
    """Paired TPR difference (A - B) at fixed thresholds."""
    pos = y == 1
    return float(np.mean(flags_a[pos]) - np.mean(flags_b[pos])) if pos.any() else float("nan")


def fpr_difference(flags_a, flags_b, y) -> float:
    neg = y == 0
    return float(np.mean(flags_a[neg]) - np.mean(flags_b[neg])) if neg.any() else float("nan")


@dataclass(frozen=True)
class ClusteredDifference:
    mean_difference: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    n_clusters: int


def clustered_paired_difference(x_a, x_b, clusters=None, level: float = 0.95) -> ClusteredDifference:
    """Miller (2024) clustered SE for the mean paired difference d_i = a_i - b_i.

    SE^2 = (1/n^2) * sum_c (sum_{i in c} (d_i - dbar))^2, with a small-sample
    factor C/(C-1). Normal CI. With few clusters this is optimistic; compare
    with ``paired_cluster_bootstrap`` and report the cluster count.
    """
    d = np.asarray(x_a, float) - np.asarray(x_b, float)
    n = d.size
    if clusters is None:
        clusters = np.arange(n)
    _, members = _cluster_index(clusters)
    C = len(members)
    dbar = d.mean()
    s = np.array([np.sum(d[m] - dbar) for m in members])
    var = (s**2).sum() / n**2 * (C / (C - 1) if C > 1 else float("nan"))
    se = math.sqrt(var) if var == var else float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    return ClusteredDifference(float(dbar), se, float(dbar - z * se), float(dbar + z * se), n, C)


# -------------------------------------------------------------------- power
def power_mcnemar(n: int, p10: float, p01: float, alpha: float = 0.05, exact: bool = True) -> float:
    """Power of the two-sided McNemar test with ``n`` paired positives.

    p10 = P(A detects, B misses), p01 = P(B detects, A misses) per violating
    root. Exact: sum over the number of discordant pairs D ~ Bin(n, p10+p01)
    of P(reject | D) with A-only ~ Bin(D, p10/(p10+p01)). Normal approximation
    (Connor 1987) when ``exact=False``.
    """
    pd_ = p10 + p01
    if pd_ <= 0:
        return 0.0
    if not exact:
        z_a = stats.norm.ppf(1 - alpha / 2)
        diff = abs(p10 - p01)
        num = z_a * math.sqrt(pd_) - math.sqrt(n) * diff
        den = math.sqrt(pd_ - diff**2)
        return float(1 - stats.norm.cdf(num / den) + stats.norm.cdf((-z_a * math.sqrt(pd_) - math.sqrt(n) * diff) / den))
    q = p10 / pd_
    d = np.arange(1, n + 1)
    w = stats.binom.pmf(d, n, pd_)
    # Exact two-sided binomial test at p=1/2 is symmetric: reject iff
    # k <= k_lo(d) or k >= d - k_lo(d), with k_lo the largest k such that
    # 2 * P(K <= k) <= alpha.
    k_lo = stats.binom.ppf(alpha / 2, d, 0.5)
    k_lo = np.where(stats.binom.cdf(k_lo, d, 0.5) <= alpha / 2, k_lo, k_lo - 1)
    lower = np.where(k_lo >= 0, stats.binom.cdf(k_lo, d, q), 0.0)
    upper = np.where(k_lo >= 0, stats.binom.sf(d - k_lo - 1, d, q), 0.0)
    overlap = k_lo >= d - k_lo  # only possible for tiny d with large alpha
    return float(np.sum(w * np.where(overlap, 1.0, lower + upper)))


def required_n_for_tpr_gain(
    gain: float = 0.05,
    discordance: float = 0.15,
    alpha: float = 0.05,
    target_power: float = 0.8,
    n_max: int = 20000,
    exact: bool = False,
) -> int:
    """Smallest number of violating roots giving ``target_power`` to detect a
    paired TPR ``gain`` when a fraction ``discordance`` of violations are
    caught by exactly one monitor (p10 + p01 = discordance, p10 - p01 = gain).

    Planning helper for freezing the locked-test size (S04 power section).
    """
    if gain >= discordance:
        raise ValueError("gain must be smaller than the total discordance")
    p10, p01 = (discordance + gain) / 2, (discordance - gain) / 2
    lo, hi = 10, n_max
    if power_mcnemar(hi, p10, p01, alpha, exact) < target_power:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if power_mcnemar(mid, p10, p01, alpha, exact) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def two_proportion_z(p1: float, p2: float, n1: int, n2: int) -> float:
    """Independent-proportions z (spec p.4 approximations). NOT a paired test."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se > 0 else float("nan")


# ---------------------------------------------------------------- seeds
def seed_spread(results_by_seed: dict) -> pd.DataFrame:
    """Per-metric mean, SD, min, max across training/selection seeds.

    ``results_by_seed`` maps seed -> {metric: value}. Report every seed, not
    only the best (plan p.5, spec p.8).
    """
    df = pd.DataFrame(results_by_seed).T
    df.index.name = "seed"
    summ = pd.DataFrame({"mean": df.mean(), "sd": df.std(ddof=1), "min": df.min(), "max": df.max(), "n_seeds": df.count()})
    return summ
