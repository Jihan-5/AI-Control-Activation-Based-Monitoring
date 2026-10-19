"""Bounded program-reasoning environment: execution oracle, sandbox, operators."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from irmon.data.schema import assert_prompt_clean
from irmon.envs.base import Outcome
from irmon.envs.program import (
    BRANCHES,
    FAMILIES,
    ProgramEnv,
    ProgramGenerator,
    SandboxTimeout,
    SandboxViolation,
    conflict_opportunity,
    families_in_tier,
    run_tests,
    verify_task,
)
from irmon.envs.program.sandbox import call_with_budget, load_function
from irmon.envs.program.tasks import IMPL_OF_SLOT, SLOTS
from irmon.interventions.program_ops import HELD_OUT_PROGRAM, PROGRAM_MAIN_T, PROGRAM_MAIN_U
from irmon.interventions.registry import HELD_OUT, get_operator

ENV = ProgramEnv()
GEN = ProgramGenerator()


@pytest.fixture(scope="module")
def roots():
    out = []
    for i in range(200):
        fam = FAMILIES[i % len(FAMILIES)].name
        out.append(GEN.sample_root(11, i, fam, BRANCHES[i % len(BRANCHES)], cue=i % 3 == 0, conflict_prompt=i % 4 == 0))
    return out


def test_families_and_tiers():
    assert len(FAMILIES) >= 10
    assert families_in_tier("familiar") and families_in_tier("dev_unfamiliar") and families_in_tier("test_unfamiliar")
    assert HELD_OUT["program"] == HELD_OUT_PROGRAM


@pytest.mark.parametrize("family", [f.name for f in FAMILIES])
@pytest.mark.parametrize("branch", BRANCHES)
def test_oracle_labels_every_family_and_variant(family, branch):
    root = GEN.sample_root(3, hash((family, branch)) % 10_000, family, branch)
    verify_task(root.task)  # executes displayed code vs trusted references
    d = ENV.decide(root.task)
    if branch == "clarification_needed":
        assert d.correct_slot == "REQUEST_CLARIFICATION"
    else:
        v = 0 if "_A_" in branch else 1
        e = int(branch[-1])
        assert IMPL_OF_SLOT[d.correct_slot] == (v, e)
    assert ENV.classify(root.task, "SUBMIT_GAMING") == (Outcome.VIOLATION, ("test_special_casing",))
    for slot in IMPL_OF_SLOT:
        expected = Outcome.CORRECT if slot == d.correct_slot else Outcome.ORDINARY_ERROR
        assert ENV.classify(root.task, slot)[0] is expected
    clar = Outcome.CORRECT if d.correct_slot == "REQUEST_CLARIFICATION" else Outcome.ORDINARY_ERROR
    assert ENV.classify(root.task, "REQUEST_CLARIFICATION")[0] is clar
    assert ENV.classify(root.task, None)[0] is Outcome.SCHEMA_INVALID
    assert conflict_opportunity(root.task)


def test_candidates_parse_roundtrip_and_prompt_clean(roots):
    for r in roots[:60]:
        cands = ENV.candidates(r.task)
        assert tuple(c.slot for c in cands) == SLOTS
        for c in cands:
            assert ENV.parse_action(r.task, c.text) == c.slot
            assert ENV.parse_action(r.task, f"  {c.text}\nextra words") == c.slot
        assert ENV.parse_action(r.task, "SUBMIT impl-zz99") is None
        assert ENV.parse_action(r.task, "I would submit the second one") is None
        assert_prompt_clean(r, ENV.render(r.task))


def test_observable_context_matches_declared_names(roots):
    ctx = ENV.observable_context(roots[0].task)
    assert tuple(ctx) == ENV.context_feature_names()
    env2 = ProgramEnv(include_rule_decision=True)
    assert tuple(env2.observable_context(roots[0].task)) == env2.context_feature_names()


def test_t_operators_fully_valid(roots):
    for op_name in PROGRAM_MAIN_T + ("p_comment",):
        op = get_operator("program", op_name)
        for i, r in enumerate(roots):
            res = op(ENV, r.task, random.Random(i))
            assert res.valid, (op_name, res.reason)
            if i % 25 == 0:
                verify_task(res.task)  # edited code still behaves identically


def test_u_operator_validity_rates(roots):
    rates = {}
    for op_name in PROGRAM_MAIN_U + ("p_signature",):
        op = get_operator("program", op_name)
        n_valid = 0
        changed = Counter()
        for i, r in enumerate(roots):
            res = op(ENV, r.task, random.Random(i))
            n_valid += res.valid
            changed[res.decision_changed] += 1
            if res.valid and i % 25 == 0:
                verify_task(res.task)
        rates[op_name] = n_valid / len(roots)
    print("program U-operator validity:", rates)
    assert rates["p_spec_change"] >= 0.8 and rates["p_visible_tests"] >= 0.8
    assert rates["p_signature"] >= 0.5  # held-out family; invalid only on contradiction roots


def test_sandbox_runs_and_compares_types():
    src = "def f(xs):\n    return sorted(xs)"
    out = run_tests(src, [(([3, 1],), [1, 3]), (([],), None)])
    assert out[0].passed and not out[1].passed  # [] is not None


@pytest.mark.parametrize(
    "src",
    [
        "import os\ndef f(x):\n    return x",
        "def f(x):\n    import os\n    return x",
        "def f(x):\n    return x.__class__",
        "def f(x):\n    return eval('1')",
        "def f(x):\n    return open('/etc/passwd')",
        "def f(x):\n    return __builtins__",
        "def f(x):\n    return (lambda y: y)(x)",
        "def f(x):\n    try:\n        return x\n    except Exception:\n        return 0",
        "x = 1\ndef f(y):\n    return y",
        "def f(x):\n    return f'{x}'",
    ],
)
def test_sandbox_rejects_disallowed_source(src):
    with pytest.raises(SandboxViolation):
        load_function(src)


def test_sandbox_step_budget_stops_infinite_loop():
    fn = load_function("def f(x):\n    while True:\n        x += 1\n    return x")
    with pytest.raises(SandboxTimeout):
        call_with_budget(fn, (0,), max_steps=1000)
    out = run_tests("def f(x):\n    while True:\n        pass\n    return x", [((1,), 1)], max_steps=500)
    assert not out[0].passed and out[0].error.startswith("timeout")


def test_sandbox_isolates_mutation():
    src = "def f(xs):\n    xs.append(9)\n    return len(xs)"
    args = ([1, 2],)
    out = run_tests(src, [(args, 3), (args, 3)])
    assert all(t.passed for t in out) and args[0] == [1, 2]
