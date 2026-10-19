"""Finite-state decision environment: resource allocation, review requirements,
authenticated handoff and revised priorities, labelled by an exact rule oracle."""

from .env import CONTEXT_FEATURES, FiniteStateEnv
from .families import FAMILIES, FAMILY_BY_NAME, families_in_tier
from .generator import ALLOCATION_BRANCHES, BRANCHES, OVERLAY_BRANCHES, FiniteStateGenerator
from .state import SLOTS, VIOLATION_TYPES, FSState, FSTask, Surface

__all__ = [
    "ALLOCATION_BRANCHES",
    "BRANCHES",
    "CONTEXT_FEATURES",
    "FAMILIES",
    "FAMILY_BY_NAME",
    "FSState",
    "FSTask",
    "FiniteStateEnv",
    "FiniteStateGenerator",
    "OVERLAY_BRANCHES",
    "SLOTS",
    "Surface",
    "VIOLATION_TYPES",
    "families_in_tier",
]
