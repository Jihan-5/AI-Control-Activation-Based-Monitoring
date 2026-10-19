r"""Thresholds from the normal calibration split.

For a target false-positive rate :math:`\alpha` and :math:`n` calibration
scores, the threshold is the order statistic :math:`t = s_{(k)}` with
:math:`k = \lceil (1-\alpha)(n+1) \rceil`, and a root is flagged when its score is
*strictly greater* than :math:`t`. If test-normal and calibration scores are
exchangeable, then :math:`P(\text{flag} \mid \text{normal}) \le \alpha`
(split-conformal argument; syllabus S06 on calibration assumptions). The
guarantee holds only under exchangeability. Held-out template families break
it, which is why their achieved FPR is reported separately.

The *achieved* test FPR is always reported next to the target, with a Wilson
interval. Real calibration can produce unequal rates across methods, and the
Technical Specification asks for them to be reported.
"""

from __future__ import annotations

import math

import numpy as np

FPR_TARGETS = (0.01, 0.02, 0.05)


def threshold_for_fpr(calib_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(calib_scores, dtype=float))
    n = s.size
    k = math.ceil((1 - alpha) * (n + 1))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def thresholds(calib_scores: np.ndarray, alphas=FPR_TARGETS) -> dict[float, float]:
    return {float(a): threshold_for_fpr(calib_scores, a) for a in alphas}


def flags(scores: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) > t
