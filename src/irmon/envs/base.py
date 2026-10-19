"""Environment-agnostic types shared by every task environment.

An *environment* turns a latent task (exact, rule-checkable state) into a chat
prompt, enumerates a bounded candidate-action set with canonical slots, and
labels any candidate with an exact policy oracle.

Canonical slots matter: interventions can rename identifiers, so the literal
action string for "allocate to the restricted pool" changes between an input and
its edit. Action-score responses are therefore aligned by *slot*, never by
string.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Outcome(str, Enum):
    """Episode-level outcome of the actor's proposed action (eval-only label)."""

    CORRECT = "correct"
    ORDINARY_ERROR = "ordinary_error"  # allowed by policy but not the best action, or infeasible
    VIOLATION = "violation"  # prohibited by an explicit policy rule
    SCHEMA_INVALID = "schema_invalid"  # free-generation output that does not parse


class SlotStatus(str, Enum):
    """Policy status of one candidate action in one state."""

    CORRECT = "correct"
    ALLOWED = "allowed_suboptimal"
    INFEASIBLE = "infeasible"
    VIOLATION = "violation"

    def to_outcome(self) -> Outcome:
        if self is SlotStatus.CORRECT:
            return Outcome.CORRECT
        if self is SlotStatus.VIOLATION:
            return Outcome.VIOLATION
        return Outcome.ORDINARY_ERROR


@dataclass(frozen=True)
class ChatPrompt:
    """Model-visible input. Never contains root IDs or evaluation-only fields."""

    system: str
    user: str

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def fingerprint(self) -> str:
        payload = json.dumps([self.system, self.user], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    """One action in the bounded candidate set, addressed by canonical slot."""

    slot: str
    text: str


@dataclass(frozen=True)
class Decision:
    """Exact oracle output for one latent state.

    ``statuses`` and ``violation_types`` are keyed by canonical slot. Every slot of
    the environment appears in ``statuses``.
    """

    correct_slot: str
    statuses: tuple[tuple[str, SlotStatus], ...]
    violation_types: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())

    def status(self, slot: str) -> SlotStatus:
        return dict(self.statuses)[slot]

    def violations(self, slot: str) -> tuple[str, ...]:
        return dict(self.violation_types).get(slot, ())

    def status_vector(self) -> tuple[str, ...]:
        return tuple(s.value for _, s in self.statuses)

    def violating_slots(self) -> list[str]:
        return [k for k, s in self.statuses if s is SlotStatus.VIOLATION]


@runtime_checkable
class Environment(Protocol):
    """Minimal protocol every environment implements."""

    name: str
    slots: tuple[str, ...]

    def render(self, task: Any) -> ChatPrompt: ...

    def candidates(self, task: Any) -> list[Candidate]: ...

    def decide(self, task: Any) -> Decision: ...

    def canonical(self, task: Any) -> tuple: ...

    def observable_context(self, task: Any) -> dict[str, float]: ...

    def context_group(self, task: Any) -> str: ...

    def parse_action(self, task: Any, text: str) -> str | None: ...


@dataclass(frozen=True)
class RootTask:
    """A root instance plus evaluation-only metadata.

    ``task`` is the environment-specific latent task. Every field other than the
    rendered prompt is evaluation-only and must never reach the actor.
    """

    root_id: str
    env: str
    family: str
    familiarity: str  # "familiar" | "dev_unfamiliar" | "test_unfamiliar"
    branch: str  # decision-determining scenario type (factor-design cell)
    task: Any
    factors: tuple[tuple[str, Any], ...] = ()

    def factor(self, key: str, default: Any = None) -> Any:
        return dict(self.factors).get(key, default)
