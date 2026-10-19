"""Split-conformal calibration and per-group FPR auditing."""

from .conformal import (
    CoverageLaw,
    SplitConformal,
    coverage_distribution,
    fpr_drift_report,
    per_group_fpr,
)

__all__ = ["CoverageLaw", "SplitConformal", "coverage_distribution", "fpr_drift_report", "per_group_fpr"]
