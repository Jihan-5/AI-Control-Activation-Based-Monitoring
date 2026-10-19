"""Bounded program-reasoning environment: small deterministic functions,
specification changes, and candidate implementations labelled by executing
them against hidden held-out tests in a restricted sandbox."""

from .env import PROGRAM_CONTEXT_FEATURES, ProgramEnv, conflict_opportunity, verify_task
from .families import FAMILIES, FAMILY_BY_NAME, families_in_tier
from .generator import BRANCHES, ProgramGenerator, ProgramGeneratorConfig
from .sandbox import SandboxTimeout, SandboxViolation, run_tests
from .tasks import CODE_SLOTS, SLOTS, VIOLATION_TYPES, ProgState, ProgSurface, ProgTask

__all__ = [
    "BRANCHES",
    "CODE_SLOTS",
    "FAMILIES",
    "FAMILY_BY_NAME",
    "PROGRAM_CONTEXT_FEATURES",
    "ProgState",
    "ProgSurface",
    "ProgTask",
    "ProgramEnv",
    "ProgramGenerator",
    "ProgramGeneratorConfig",
    "SLOTS",
    "SandboxTimeout",
    "SandboxViolation",
    "VIOLATION_TYPES",
    "conflict_opportunity",
    "families_in_tier",
    "run_tests",
    "verify_task",
]
