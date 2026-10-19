"""Split-conformal calibration of the final combined score (plan p.7, syllabus S06).

Protocol
--------
1. Fit the *entire* scoring function (projection, predictor, covariance, layer
   aggregation) on normal-fit data and freeze it (``CalibrationLock``).
2. Score the held-out normal-calibration split once.
3. Upper-tail conformal p-value of a new score s:

       p(s) = (1 + #{i : cal_i >= s}) / (n + 1)

   Flag when p(s) <= alpha. Equivalently flag when s > t_alpha where t_alpha is
   the k-th smallest calibration score with k = ceil((n + 1)(1 - alpha)).

Assumption and its limit
------------------------
The finite-sample guarantee P(flag | normal) <= alpha holds *marginally* when
the calibration scores and the new normal score are exchangeable. It does NOT
certify a shifted deployment distribution: unfamiliar-but-legitimate task
families, new surface formats, or another checkpoint all break
exchangeability. That is precisely why achieved FPR is re-measured on every
held-out normal group (``per_group_fpr`` / ``fpr_drift_report``) instead of
being assumed equal to the nominal level.

Calibrate the combined score, never each component independently followed by
an uncalibrated OR rule (that does not preserve the global error rate).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CoverageLaw:
    """Law of the *conditional* normal-acceptance rate given the calibration set.

    For continuous exchangeable scores, the probability that a fresh normal score
    is NOT flagged, conditional on the calibration sample, is distributed as
    Beta(n + 1 - l, l) with l = floor((n + 1) * alpha) (Angelopoulos & Bates,
    Sec. 3.2). The achieved FPR on a large normal test set is therefore
    1 - Beta(n + 1 - l, l) even with no distribution shift at all.
    """

    n: int
    alpha: float
    a: float
    b: float

    @property
    def mean_fpr(self) -> float:
        return float(self.b / (self.a + self.b))

    def fpr_quantiles(self, qs=(0.05, 0.5, 0.95)) -> dict[float, float]:
        """Quantiles of the conditional FPR (= 1 - coverage)."""
        return {float(q): float(1.0 - stats.beta.ppf(1.0 - q, self.a, self.b)) for q in qs}

    def prob_fpr_exceeds(self, level: float) -> float:
        """P(conditional FPR > level) over draws of the calibration set."""
        return float(stats.beta.cdf(1.0 - level, self.a, self.b))


def coverage_distribution(n: int, alpha: float) -> CoverageLaw:
    if n < 1:
        raise ValueError("need at least one calibration score")
    ell = math.floor((n + 1) * alpha)
    if ell < 1:
        raise ValueError(f"alpha={alpha} too small for n={n}: (n+1)*alpha < 1, threshold is +inf")
    return CoverageLaw(n=n, alpha=alpha, a=float(n + 1 - ell), b=float(ell))


class SplitConformal:
    """Upper-tail split-conformal calibration on normal-calibration scores."""

    def __init__(self) -> None:
        self.cal_: np.ndarray | None = None

    def fit(self, cal_scores) -> SplitConformal:
        cal = np.asarray(cal_scores, dtype=float).ravel()
        if cal.size == 0:
            raise ValueError("empty calibration set")
        if not np.all(np.isfinite(cal)):
            raise ValueError("calibration scores must be finite")
        self.cal_ = np.sort(cal)
        return self

    @property
    def n(self) -> int:
        self._check()
        return int(self.cal_.size)

    def _check(self) -> None:
        if self.cal_ is None:
            raise RuntimeError("SplitConformal: call fit() first")

    def p_values(self, scores) -> np.ndarray:
        """p(s) = (1 + #{cal >= s}) / (n + 1), vectorised."""
        self._check()
        s = np.asarray(scores, dtype=float)
        n_ge = self.n - np.searchsorted(self.cal_, s, side="left")
        return (1.0 + n_ge) / (self.n + 1.0)

    def threshold(self, alpha: float) -> float:
        """Score threshold t_alpha: flag iff score > t_alpha.

        t_alpha is the k-th smallest calibration score, k = ceil((n+1)(1-alpha)).
        If k > n, no finite threshold achieves the guarantee: returns +inf (never
        flag) and warns. Flagging ``score > t_alpha`` is identical to
        ``p(score) <= alpha`` when calibration scores have no ties.
        """
        self._check()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        k = math.ceil(round((self.n + 1) * (1.0 - alpha), 9))
        if k > self.n:
            warnings.warn(
                f"n={self.n} calibration scores cannot support alpha={alpha}; threshold=+inf",
                stacklevel=2,
            )
            return float("inf")
        return float(self.cal_[max(k, 1) - 1])

    def flag(self, scores, alpha: float) -> np.ndarray:
        """Boolean flags via conformal p-values (robust to score ties)."""
        return self.p_values(scores) <= alpha

    def coverage_law(self, alpha: float) -> CoverageLaw:
        return coverage_distribution(self.n, alpha)


# ---------------------------------------------------------------- intervals
def _cp(k: int, n: int, level: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def _wilson(k: int, n: int, level: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def per_group_fpr(
    scores,
    groups,
    threshold: float | None = None,
    flags=None,
    level: float = 0.95,
    method: str = "clopper_pearson",
) -> pd.DataFrame:
    """Achieved false-positive rate per held-out normal group.

    ``scores``/``groups`` must contain NORMAL roots only. Supply either a
    calibrated ``threshold`` (flag iff score > threshold) or precomputed
    ``flags``. Intervals treat roots within a group as independent draws;
    edits of the same root are not separate observations (they are inside one
    root's score).
    """
    groups = np.asarray(groups)
    if flags is None:
        if threshold is None:
            raise ValueError("pass threshold or flags")
        flags = np.asarray(scores, dtype=float) > threshold
    flags = np.asarray(flags, dtype=bool)
    ci = _cp if method == "clopper_pearson" else _wilson
    rows = []
    for g in sorted(pd.unique(groups).tolist(), key=str):
        m = groups == g
        k, n = int(flags[m].sum()), int(m.sum())
        lo, hi = ci(k, n, level)
        rows.append({"group": g, "n": n, "false_alarms": k, "fpr": k / n if n else float("nan"),
                     "ci_low": lo, "ci_high": hi})
    k, n = int(flags.sum()), int(flags.size)
    lo, hi = ci(k, n, level)
    rows.append({"group": "ALL", "n": n, "false_alarms": k, "fpr": k / n if n else float("nan"),
                 "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def fpr_drift_report(
    conformal: SplitConformal,
    normal_scores,
    normal_groups,
    alphas=(0.01, 0.05),
    level: float = 0.95,
) -> pd.DataFrame:
    """Nominal vs achieved FPR per held-out normal group, per nominal level.

    Adds the no-shift reference band from ``coverage_distribution`` so a reader
    can tell sampling variation of the calibration set apart from genuine
    benign-shift failure ("a nominal 1% threshold can fail badly under shift").
    """
    out = []
    for a in alphas:
        flags = conformal.flag(normal_scores, a)
        df = per_group_fpr(normal_scores, normal_groups, flags=flags, level=level)
        df.insert(0, "nominal_fpr", a)
        df["ratio_to_nominal"] = df["fpr"] / a
        try:
            law = conformal.coverage_law(a)
            q = law.fpr_quantiles((0.05, 0.95))
            df["no_shift_band_low"], df["no_shift_band_high"] = q[0.05], q[0.95]
        except ValueError:
            df["no_shift_band_low"] = df["no_shift_band_high"] = float("nan")
        df["exceeds_no_shift_band"] = df["ci_low"] > df["no_shift_band_high"]
        out.append(df)
    return pd.concat(out, ignore_index=True)
