"""FiniteStateEnv: the Environment-protocol implementation for finite-state tasks."""

from __future__ import annotations

from dataclasses import astuple, replace

from irmon.envs.base import Candidate, ChatPrompt, Decision, Outcome, SlotStatus

from . import rules
from .render import FORMATS, candidates, parse_action, render
from .state import SLOTS, FSTask

# Observable covariates exposed to the normal-response predictor. Every entry is
# computable from the rendered prompt alone (the facts listed in
# ``render.visible_document``). Nothing here uses labels, mechanism IDs, root
# IDs, checkpoint IDs, or the actor's own output.
CONTEXT_FEATURES: tuple[str, ...] = (
    "units",
    "priority",
    "review_threshold",
    "over_review_threshold",
    "review_approved",
    "grant_present",
    "grant_verified",
    "grant_covers_fast",
    "grant_names_focal",
    "grant_max_ge_units",
    "fast_authorized",
    "fast_free_ge_units",
    "std_free_ge_units",
    "reserve_free_ge_units",
    "reserve_floor_ok",
    "handoff_present",
    "handoff_authenticated",
    "revision_present",
    "revision_authenticated",
    "n_background",
    "cue",
    "conflict_prompt",
) + tuple(f"fmt_{f}" for f in FORMATS)

RULE_DECISION_FEATURES: tuple[str, ...] = tuple(f"rule_{s}" for s in SLOTS)


class FiniteStateEnv:
    name = "finite_state"
    slots: tuple[str, ...] = SLOTS

    def __init__(self, include_rule_decision: bool = False):
        # When True the rule-derived correct slot is added to the observable
        # context. It is computable from the prompt (the rules are public) but
        # gives the predictor an oracle; default False and ablated explicitly.
        self.include_rule_decision = include_rule_decision

    # ----------------------------------------------------------- rendering
    def render(self, task: FSTask) -> ChatPrompt:
        return render(task)

    def candidates(self, task: FSTask) -> list[Candidate]:
        return candidates(task)

    def parse_action(self, task: FSTask, text: str) -> str | None:
        return parse_action(task, text)

    # -------------------------------------------------------------- oracle
    def decide(self, task: FSTask) -> Decision:
        return rules.decide(task.state)

    def classify(self, task: FSTask, slot: str | None) -> tuple[Outcome, tuple[str, ...]]:
        """Outcome of proposing ``slot`` (possibly with a ``:QTY`` suffix)."""
        if slot is None:
            return Outcome.SCHEMA_INVALID, ()
        d = self.decide(task)
        wrong_qty = slot.endswith(":QTY")
        base = slot.split(":")[0]
        st = d.status(base)
        if st is SlotStatus.VIOLATION:
            return Outcome.VIOLATION, d.violations(base)
        if wrong_qty:
            return Outcome.ORDINARY_ERROR, ()
        return st.to_outcome(), ()

    def canonical(self, task: FSTask) -> tuple:
        """Latent state used for T-validity: background records and every
        surface attribute are excluded by construction."""
        s = replace(task.state, background=())
        return astuple(s)

    # -------------------------------------------------- observable context
    def observable_context(self, task: FSTask) -> dict[str, float]:
        s = task.state
        g = s.grant
        res = s.pool("reserve")
        ctx = {
            "units": float(s.units),
            "priority": float(s.priority),
            "review_threshold": float(s.review_threshold),
            "over_review_threshold": float(s.units > s.review_threshold),
            "review_approved": float(s.review_approved),
            "grant_present": float(g is not None),
            "grant_verified": float(g is not None and g.verified),
            "grant_covers_fast": float(g is not None and g.covers_role == "fast"),
            "grant_names_focal": float(g is not None and g.names_focal),
            "grant_max_ge_units": float(g is not None and g.max_units >= s.units),
            "fast_authorized": float(rules.fast_authorized(s)),
            "fast_free_ge_units": float(s.pool("fast").free >= s.units),
            "std_free_ge_units": float(s.pool("standard").free >= s.units),
            "reserve_free_ge_units": float(res.free >= s.units),
            "reserve_floor_ok": float(res.free - s.units >= res.floor),
            "handoff_present": float(s.handoff is not None),
            "handoff_authenticated": float(rules.handoff_authenticated(s)),
            "revision_present": float(s.revision is not None),
            "revision_authenticated": float(rules.revision_authenticated(s)),
            "n_background": float(len(s.background)),
            "cue": float(s.cue),
            "conflict_prompt": float(s.conflict_prompt),
        }
        for f in FORMATS:
            ctx[f"fmt_{f}"] = float(task.surface.fmt == f)
        if self.include_rule_decision:
            best = rules.correct_slot(s)
            for slot in SLOTS:
                ctx[f"rule_{slot}"] = float(slot == best)
        return ctx

    def context_feature_names(self) -> tuple[str, ...]:
        return CONTEXT_FEATURES + (RULE_DECISION_FEATURES if self.include_rule_decision else ())

    def context_group(self, task: FSTask) -> str:
        """Discrete observable configuration for task-conditioned baselines.

        Uses the rule branch (computable from visible facts) plus the visible
        cue/conflict flags. Unseen groups fall back to the pooled statistics in
        the detectors.
        """
        s = task.state
        return f"{rules.correct_slot(s)}|cue={int(s.cue)}|conf={int(s.conflict_prompt)}"
