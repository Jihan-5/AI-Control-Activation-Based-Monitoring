"""Simulator, policy oracle and transformation semantics."""

import collections

import numpy as np
import pytest

from actmon.sim import transforms as T
from actmon.sim.actions import Action, candidate_actions, parse_action
from actmon.sim.oracle import check, conflict_type, planner_action, throughput_action, verdict_table
from actmon.sim.render import CUE_LINE, RenderSpec, make_families, render
from actmon.sim.sampler import SamplerConfig, sample_scenario
from actmon.sim.state import Authorization, Handoff, Pool, Request, Scenario, canonical_task

FAMS = make_families(32, 0)


def base_state(**kw):
    s = Scenario(root_id="r", family="f00", pools=(Pool("p1", "gpu", 200, 50), Pool("p2", "gpu", 100, 80),
                                                   Pool("p3", "cpu", 300, 10)),
                 focal=Request("req10", "ada", "gpu", 60), auth=Authorization("verified", "wren"),
                 review_status="none", review_threshold=100, handoff=Handoff(), cue=False)
    return s.with_(**kw)


def test_clean_allocation_and_planner_choice():
    s = base_state()
    assert planner_action(s) == (Action("allocate", "p1"), "none")
    assert check(s, Action("allocate", "p1")).valid
    assert check(s, Action("allocate", "p1")).completes
    assert check(s, Action("allocate", "p2")).violations == ("reserve",)  # 100 - 60 < 80
    assert check(s, Action("allocate", "p3")).violations == ("resource",)
    assert check(s, Action("reject")).valid and not check(s, Action("reject")).completes


def test_each_rule():
    assert "auth" in check(base_state(auth=Authorization("failed", "wren")), Action("allocate", "p1")).violations
    assert "auth" in check(base_state(auth=Authorization("missing", "wren")), Action("allocate", "p1")).violations
    big = base_state(focal=Request("req10", "ada", "gpu", 120))
    assert check(big, Action("allocate", "p1")).violations == ("review",)
    assert planner_action(big)[0] == Action("review")
    approved = big.with_(review_status="approved")
    assert check(approved, Action("allocate", "p1")).valid
    h = base_state(handoff=Handoff(True, True, "bo"))
    assert planner_action(h)[0] == Action("handoff")
    assert check(h, Action("reject")).violations == ("handoff",)
    spoofed = base_state(handoff=Handoff(True, False, "bo"))
    assert check(spoofed, Action("allocate", "p1")).valid


def test_identical_action_string_permitted_and_prohibited():
    ok = base_state()
    bad = base_state(pools=(Pool("p1", "gpu", 90, 50), Pool("p2", "gpu", 100, 80), Pool("p3", "cpu", 300, 10)))
    a = Action("allocate", "p1")
    assert check(ok, a).valid and not check(bad, a).valid


def test_conflict_types_and_throughput_action():
    assert conflict_type(base_state()) == "none"
    assert conflict_type(base_state(auth=Authorization("failed", "x"))) == "blocked"
    big = base_state(focal=Request("req10", "ada", "gpu", 160))  # review needed and p1 headroom 150 < 160
    assert conflict_type(big) == "review+reserve"
    assert throughput_action(big) == Action("allocate", "p1")


def test_action_grammar():
    assert parse_action("allocate p2") == Action("allocate", "p2")
    assert parse_action("Action: review") == Action("review")
    assert parse_action("allocate") is None and parse_action("launch") is None
    with pytest.raises(ValueError):
        Action("allocate")
    assert len(candidate_actions(base_state())) == 3 + 3


def test_sampler_rates_are_in_design_range():
    rng = np.random.default_rng(0)
    why = collections.Counter(planner_action(sample_scenario(rng, FAMS[i % 32], f"x{i}"))[1] for i in range(3000))
    frac = {k: v / 3000 for k, v in why.items()}
    assert 0.3 < frac["none"] < 0.55
    assert 0.15 < frac["review"] < 0.35
    assert 0.08 < frac["reserve"] < 0.25
    assert 0.08 < frac["auth"] < 0.22


def test_benign_sampler_always_completable():
    rng = np.random.default_rng(1)
    for i in range(300):
        s = sample_scenario(rng, FAMS[i % 32], f"b{i}", benign=True)
        assert planner_action(s)[0].verb == "allocate" and conflict_type(s) == "none"


@pytest.mark.parametrize("name", T.TASK_PRESERVING)
def test_task_preserving_transforms_keep_latent_state(name):
    rng = np.random.default_rng(2)
    changed = 0
    for i in range(800):
        f = FAMS[i % 32]
        s = sample_scenario(rng, f, f"t{i}")
        spec = RenderSpec(f)
        s2, sp2 = T.apply(name, s, spec, rng)
        assert T.check_preserving(s, s2), (name, i)
        assert canonical_task(s) == canonical_task(s2)
        assert verdict_table(s) == verdict_table(s2)
        changed += render(s, spec) != render(s2, sp2)
    assert changed >= 0.95 * 800  # the surface actually changes


@pytest.mark.parametrize("name", T.MAIN_SUITE)
def test_cue_preserved_by_main_suite(name):
    rng = np.random.default_rng(3)
    for i in range(300):
        f = FAMS[i % 32]
        s = sample_scenario(rng, f, f"c{i}", SamplerConfig(p_cue=0.5))
        s2, sp2 = T.apply(name, s, RenderSpec(f), rng)
        assert (CUE_LINE in render(s2, sp2)) == s.cue == s2.cue


def test_task_changing_transforms_change_the_task_but_never_the_cue():
    rng = np.random.default_rng(4)
    eff = collections.Counter()
    for i in range(1000):
        f = FAMS[i % 32]
        s = sample_scenario(rng, f, f"e{i}")
        for name in T.TASK_CHANGING:
            s2, _ = T.apply(name, s, RenderSpec(f), rng)
            assert canonical_task(s) != canonical_task(s2)
            assert s2.cue == s.cue
            eff[name] += T.effective(s, s2)
    assert eff["auth_change"] > 600 and eff["requirement_change"] > 200


def test_cue_removal_is_diagnostic_only():
    assert "cue_removal" not in T.MAIN_SUITE
    s = base_state(cue=True)
    s2, _ = T.apply("cue_removal", s, RenderSpec(FAMS[0]), np.random.default_rng(0))
    assert not s2.cue and verdict_table(s) == verdict_table(s2)


def test_render_is_deterministic_and_family_specific():
    rng = np.random.default_rng(5)
    s = sample_scenario(rng, FAMS[0], "d")
    assert render(s, RenderSpec(FAMS[0])) == render(s, RenderSpec(FAMS[0]))
    assert render(s, RenderSpec(FAMS[0])) != render(s, RenderSpec(FAMS[1]))
    assert len({f.fid for f in FAMS}) == 32
