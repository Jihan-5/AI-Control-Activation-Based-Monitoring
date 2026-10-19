"""Intervention-contract tests: T edits preserve meaning, U edits change it."""

from __future__ import annotations

import random

import pytest

from irmon.envs.finite_state import BRANCHES, FAMILIES, FiniteStateEnv, FiniteStateGenerator
from irmon.interventions import EditSuite, get_operator
from irmon.interventions.registry import HELD_OUT, MAIN_T, MAIN_U

ENV = FiniteStateEnv()
GEN = FiniteStateGenerator()
ROOTS = [
    GEN.sample_root(21, i, FAMILIES[i % len(FAMILIES)].name, BRANCHES[i % len(BRANCHES)], cue=i % 2 == 0)
    for i in range(360)
]


@pytest.mark.parametrize("op", [*MAIN_T, HELD_OUT["finite_state"][0]])
def test_t_operators_always_valid_and_meaning_preserving(op):
    operator = get_operator("finite_state", op)
    for i, r in enumerate(ROOTS):
        res = operator(ENV, r.task, random.Random(i))
        assert res.valid, (op, res.reason)
        assert ENV.canonical(res.task) == ENV.canonical(r.task)
        assert ENV.decide(res.task) == ENV.decide(r.task)
        assert ENV.render(res.task) != ENV.render(r.task)
        # Planted cue / conflict prompt are preserved by the main suite.
        assert res.task.state.cue == r.task.state.cue
        assert res.task.state.conflict_prompt == r.task.state.conflict_prompt


@pytest.mark.parametrize("op", [*MAIN_U, HELD_OUT["finite_state"][1]])
def test_u_operators_change_the_decision_when_valid(op):
    operator = get_operator("finite_state", op)
    n_valid = 0
    for i, r in enumerate(ROOTS):
        res = operator(ENV, r.task, random.Random(i))
        if res.valid:
            n_valid += 1
            assert res.decision_changed
            assert ENV.decide(res.task).correct_slot != ENV.decide(r.task).correct_slot
        else:
            assert res.task is r.task  # invalid edits return the original, flagged
    # Structural limits: quantity cannot change a handoff/revision decision.
    assert n_valid / len(ROOTS) >= 0.6


def test_authorization_edit_is_always_available():
    op = get_operator("finite_state", "u_authorization")
    for i, r in enumerate(ROOTS):
        assert op(ENV, r.task, random.Random(i)).valid


def test_outcome_matched_pair_via_authorization_flip():
    """Same action string, authorized in one context and prohibited in the other."""
    op = get_operator("finite_state", "u_authorization")
    r = GEN.sample_root(22, 0, "compute_cluster", "fast_authorized")
    res = op(ENV, r.task, random.Random(0))
    assert res.valid
    # Find an edit that specifically invalidated the grant.
    for k in range(20):
        res = op(ENV, r.task, random.Random(k))
        if res.task.state.grant is not None and not res.task.state.grant.verified:
            break
    a = {c.slot: c.text for c in ENV.candidates(r.task)}["ALLOC_FAST"]
    b = {c.slot: c.text for c in ENV.candidates(res.task)}["ALLOC_FAST"]
    assert a == b  # identical action string
    assert ENV.classify(r.task, "ALLOC_FAST")[0].value == "correct"
    assert ENV.classify(res.task, "ALLOC_FAST")[0].value == "violation"


def test_suite_is_reproducible_and_rebinding_keeps_kind():
    suite = EditSuite.default()
    a = suite.apply(ENV, ROOTS[0].task, seed=5)
    b = suite.apply(ENV, ROOTS[0].task, seed=5)
    assert [[x.task for x in d] for d in a] == [[x.task for x in d] for d in b]
    held = suite.rebind("t_paraphrase", "t_distractor")
    assert held.kinds == suite.kinds
    with pytest.raises(ValueError):
        suite.rebind("t_paraphrase", "u_capacity")


def test_multiple_draws_differ():
    suite = EditSuite.default()
    out = suite.apply(ENV, ROOTS[3].task, seed=1, n_draws=3)
    renames = [ENV.render(r.task) for r in out[0]]
    assert len(set(renames)) == 3
