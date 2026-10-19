"""Detector-aware stress tests: black-box search over permitted fictional
surface variations, and white-box soft-prompt obfuscation (torch only).

Run only after the nonadaptive locked test. Always report the attacker's
access, search budget, failed attempts and retained violating behaviour.
"""

from .black_box import (
    DEFAULT_FRAMING_BANK,
    AttackReport,
    AttackSummary,
    BlackBoxSearch,
    FramedTask,
    FramingEnv,
    attack_curve,
    best_within,
    default_mutators,
)

__all__ = [
    "DEFAULT_FRAMING_BANK",
    "AttackReport",
    "AttackSummary",
    "BlackBoxSearch",
    "FramedTask",
    "FramingEnv",
    "attack_curve",
    "best_within",
    "default_mutators",
]
