"""Action grammar.

The actor answers with one of::

    allocate <pool_id>      move the focal ticket's full amount out of <pool_id>
    review                  ask for a review of the focal ticket
    reject                  refuse the focal ticket
    handoff                 stop and hand control to the operator

The focal ticket is implicit, so renaming ticket identifiers never changes an
action string. That keeps the rename transformation task-preserving at the
action level too. Amounts are not chosen by the actor: allocations always move
the requested amount. This keeps the candidate set finite and small, so
decoding can be constrained exactly (:mod:`actmon.actor.decode`).

Identical action strings can be permitted in one state and prohibited in
another (``allocate p2`` with and without headroom). The experiment plan asks
for such pairs, and :func:`actmon.evaluation.metrics.ambiguous_action_strings`
counts them.
"""

from __future__ import annotations

from dataclasses import dataclass

from actmon.sim.state import Scenario

VERBS = ("allocate", "review", "reject", "handoff")


@dataclass(frozen=True, order=True)
class Action:
    verb: str
    pool: str = ""

    def __post_init__(self):
        if self.verb not in VERBS:
            raise ValueError(f"unknown verb {self.verb!r}")
        if (self.verb == "allocate") != bool(self.pool):
            raise ValueError("allocate needs a pool; other verbs take none")

    def text(self) -> str:
        return f"{self.verb} {self.pool}" if self.pool else self.verb

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text()


def parse_action(text: str) -> Action | None:
    """Parse free text; returns ``None`` for anything outside the grammar."""
    toks = text.strip().lower().replace(":", " ").split()
    if not toks:
        return None
    if toks[0] == "action" and len(toks) > 1:
        toks = toks[1:]
    verb = toks[0]
    if verb == "allocate":
        if len(toks) < 2:
            return None
        return Action("allocate", toks[1])
    if verb in VERBS:
        return Action(verb)
    return None


def candidate_actions(s: Scenario) -> list[Action]:
    """The finite candidate set: allocate from any listed pool, plus the three other verbs.

    Pools of the wrong resource type are included on purpose. Choosing one is a
    possible ordinary mistake, and the oracle labels it.
    """
    pools = sorted(p.pid for p in s.pools)
    return [Action("allocate", p) for p in pools] + [Action("review"), Action("reject"), Action("handoff")]
