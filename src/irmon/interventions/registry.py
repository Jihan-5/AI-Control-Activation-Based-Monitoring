"""Operator registry and edit-suite construction.

An *edit suite* is an ordered list of slots, each bound to one operator. Slot
order fixes the layout of the response signature. The held-out-family
experiment keeps the slot layout but rebinds one slot to a held-out operator
of the same kind at test time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from irmon.envs.base import Environment

from .base import EditResult, Intervention
from .finite_state_ops import FINITE_STATE_OPERATORS

MAIN_T = ("t_rename", "t_reorder", "t_format", "t_paraphrase")
MAIN_U = ("u_authorization", "u_quantity")
HELD_OUT = {"finite_state": ("t_distractor", "u_capacity")}

_REGISTRY: dict[str, dict[str, Intervention]] = {"finite_state": dict(FINITE_STATE_OPERATORS)}

try:  # The program-reasoning environment registers its own operators.
    from .program_ops import HELD_OUT_PROGRAM, PROGRAM_OPERATORS

    _REGISTRY["program"] = dict(PROGRAM_OPERATORS)
    HELD_OUT["program"] = HELD_OUT_PROGRAM
except ImportError:  # pragma: no cover - module optional during development
    pass


def register(env_name: str, op: Intervention) -> None:
    _REGISTRY.setdefault(env_name, {})[op.name] = op


def get_operator(env_name: str, name: str) -> Intervention:
    try:
        return _REGISTRY[env_name][name]
    except KeyError as exc:
        raise KeyError(f"no operator {name!r} registered for env {env_name!r}") from exc


def operators_for(env_name: str) -> dict[str, Intervention]:
    return dict(_REGISTRY.get(env_name, {}))


@dataclass(frozen=True)
class EditSuite:
    """Ordered (slot_name, operator_name) bindings."""

    env: str
    slots: tuple[tuple[str, str], ...]

    @classmethod
    def default(cls, env: str = "finite_state", t_ops=MAIN_T, u_ops=MAIN_U) -> EditSuite:
        return cls(env, tuple((op, op) for op in (*t_ops, *u_ops)))

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(s for s, _ in self.slots)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(get_operator(self.env, op).kind for _, op in self.slots)

    def rebind(self, slot: str, new_op: str) -> EditSuite:
        """Return a suite where ``slot`` uses ``new_op`` (must be the same kind)."""
        old = dict(self.slots)[slot]
        if get_operator(self.env, old).kind != get_operator(self.env, new_op).kind:
            raise ValueError("held-out rebinding must preserve the T/U kind of a slot")
        return EditSuite(self.env, tuple((s, new_op if s == slot else o) for s, o in self.slots))

    def apply(self, env: Environment, task: Any, seed: int, n_draws: int = 1) -> list[list[EditResult]]:
        """Apply every slot ``n_draws`` times with independent, reproducible RNGs."""
        out = []
        for slot, op_name in self.slots:
            op = get_operator(self.env, op_name)
            draws = []
            for k in range(n_draws):
                rng = random.Random(f"edit:{seed}:{slot}:{op_name}:{k}")
                draws.append(op(env, task, rng))
            out.append(draws)
        return out
