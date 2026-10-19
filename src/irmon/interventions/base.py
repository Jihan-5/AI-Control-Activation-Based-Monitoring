"""Intervention contracts.

Two response families are kept strictly separate (research plan p.6):

T  nuisance changes  - rename irrelevant identifiers, reorder independent
                       records, change serialization/whitespace, paraphrase.
                       Contract: latent task state AND the oracle decision
                       (correct slot + every slot's policy status) are identical.
U  task changes      - modify a legitimate fact (authorization, required
                       quantity, capacity). Contract: the latent state changes
                       and the policy status vector changes; by default the
                       correct decision must change too. A U edit is never
                       described as meaning-preserving.

Every edit is validated programmatically. Invalid edits are retried with fresh
randomness and, if still invalid, returned with ``valid=False`` so the
signature builder can mask them (they are never silently kept).

Planted cues are preserved by every operator in the main suite: removing a cue
would test trigger removal, not a general monitoring signal (spec p.2).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from irmon.envs.base import Environment


@dataclass(frozen=True)
class EditResult:
    op: str
    kind: str  # "T" | "U"
    family: str  # operator family (used for held-out-family experiments)
    task: Any
    valid: bool
    decision_changed: bool
    reason: str = ""


class Intervention(ABC):
    """An edit operator with a declared contract and a validator."""

    name: str = "abstract"
    kind: str = "T"
    family: str = "abstract"
    env: str = "any"
    require_decision_change: bool = True  # U operators only

    @abstractmethod
    def apply(self, env: Environment, task: Any, rng: random.Random) -> Any:
        """Return an edited task (may be invalid; validation happens after)."""

    def validate(self, env: Environment, original: Any, edited: Any) -> tuple[bool, bool, str]:
        """Return (valid, decision_changed, reason)."""
        if edited is None:
            return False, False, "operator not applicable"
        d0, d1 = env.decide(original), env.decide(edited)
        changed = d0.correct_slot != d1.correct_slot
        if self.kind == "T":
            if env.canonical(original) != env.canonical(edited):
                return False, changed, "T edit altered latent state"
            if d0.correct_slot != d1.correct_slot or d0.status_vector() != d1.status_vector():
                return False, changed, "T edit altered the oracle decision"
            if env.render(original) == env.render(edited):
                return False, changed, "T edit produced an identical prompt"
            return True, False, "ok"
        if self.kind == "U":
            if env.canonical(original) == env.canonical(edited):
                return False, changed, "U edit left latent state unchanged"
            if d0.status_vector() == d1.status_vector() and not changed:
                return False, changed, "U edit did not change policy semantics"
            if self.require_decision_change and not changed:
                return False, changed, "U edit did not change the correct decision"
            return True, changed, "ok"
        raise ValueError(f"unknown intervention kind {self.kind!r}")

    def __call__(self, env: Environment, task: Any, rng: random.Random, max_tries: int = 8) -> EditResult:
        if hasattr(task, "latent_task"):
            # Presentation wrapper (e.g. attack framing): edit the latent task,
            # validate against the inner environment, keep the wrapper.
            res = self.__call__(getattr(env, "inner", env), task.latent_task, rng, max_tries)
            return replace(res, task=task.rewrap(res.task))
        reason = "not attempted"
        changed = False
        for _ in range(max_tries):
            edited = self.apply(env, task, rng)
            valid, changed, reason = self.validate(env, task, edited)
            if valid:
                return EditResult(self.name, self.kind, self.family, edited, True, changed, reason)
        return EditResult(self.name, self.kind, self.family, task, False, changed, reason)
