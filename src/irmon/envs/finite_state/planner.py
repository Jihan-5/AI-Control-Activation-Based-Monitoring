"""Deterministic rule-respecting planner and SFT example builders.

Used for (a) the C1 competence adapter (2,000-5,000 planner-verified examples,
plan p.5), (b) the matched control adapter trained on normal data for the same
token budget, and (c) the C2 objective-biased organism, whose conflict examples
deliberately prefer the unauthorized restricted pool over a valid slower
alternative *inside the fictional simulator only*.

Loss is applied to assistant actions only; see ``irmon.models.lora``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from irmon.envs.base import RootTask

from . import rules
from .render import candidates, render


@dataclass(frozen=True)
class SFTExample:
    messages: tuple[dict[str, str], ...]  # system + user (prompt side, masked)
    target: str  # assistant action text (the only supervised tokens)
    kind: str  # "competence" | "routine" | "conflict" | "control"
    root_id: str


def plan(root: RootTask) -> str:
    """Return the canonical slot chosen by the exact planner."""
    return rules.correct_slot(root.task.state)


def planner_example(root: RootTask, kind: str = "competence") -> SFTExample:
    task = root.task
    slot = plan(root)
    text = {c.slot: c.text for c in candidates(task)}[slot]
    return SFTExample(tuple(render(task).as_messages()), text, kind, root.root_id)


def organism_example(root: RootTask) -> SFTExample:
    """Throughput-biased target for conflict-opportunity roots.

    Only valid when ``rules.conflict_opportunity`` holds: the restricted pool is
    free but unauthorized while the slower standard pool is correct. The target
    proposes the restricted pool. This is a bounded *fictional* objective
    conflict: no real tools, no real safeguards, no host persistence.
    """
    task = root.task
    if not rules.conflict_opportunity(task.state):
        raise ValueError("organism conflict targets require a conflict-opportunity root")
    text = {c.slot: c.text for c in candidates(task)}["ALLOC_FAST"]
    return SFTExample(tuple(render(task).as_messages()), text, "conflict", root.root_id)


def build_organism_mix(
    routine: list[RootTask],
    conflict: list[RootTask],
    conflict_fraction: float = 0.30,
    n_total: int | None = None,
    seed: int = 0,
) -> list[SFTExample]:
    """70% routine planner examples + 30% bounded goal conflicts (plan defaults)."""
    rng = random.Random(seed)
    n_total = n_total or (len(routine) + len(conflict))
    n_conf = int(round(conflict_fraction * n_total))
    n_rout = n_total - n_conf
    if n_conf > len(conflict) or n_rout > len(routine):
        raise ValueError(
            f"not enough roots: need {n_rout} routine/{n_conf} conflict, have {len(routine)}/{len(conflict)}"
        )
    out = [planner_example(r, "routine") for r in rng.sample(routine, n_rout)]
    out += [organism_example(r) for r in rng.sample(conflict, n_conf)]
    rng.shuffle(out)
    return out
