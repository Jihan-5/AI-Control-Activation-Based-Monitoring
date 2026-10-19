"""Split design, schema separation, manifests, leakage guards and audits."""

from .schema import HELD_OUT_MECHANISMS, KNOWN_MECHANISMS, LABEL_COLUMNS, attribute_mechanism
from .splits import (
    CalibrationLock,
    LeakageError,
    SplitGuard,
    SplitPlan,
    SplitQuota,
    assign_splits,
    check_split_invariants,
    load_manifest,
    write_manifest,
)

__all__ = [
    "HELD_OUT_MECHANISMS",
    "KNOWN_MECHANISMS",
    "LABEL_COLUMNS",
    "CalibrationLock",
    "LeakageError",
    "SplitGuard",
    "SplitPlan",
    "SplitQuota",
    "assign_splits",
    "attribute_mechanism",
    "check_split_invariants",
    "load_manifest",
    "write_manifest",
]
