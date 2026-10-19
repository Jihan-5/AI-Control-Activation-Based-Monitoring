"""Finite-state resource-allocation simulator, policy oracle and transformations."""

from actmon.sim.actions import VERBS, Action, candidate_actions, parse_action
from actmon.sim.oracle import Verdict, check, conflict_type, planner_action, throughput_action, verdict_table
from actmon.sim.render import CUE_LINE, Family, RenderSpec, make_families, render, render_lines
from actmon.sim.sampler import SamplerConfig, sample_scenario
from actmon.sim.state import Scenario, canonical_task

__all__ = [
    "VERBS", "Action", "candidate_actions", "parse_action", "Verdict", "check", "conflict_type", "planner_action",
    "throughput_action", "verdict_table", "CUE_LINE", "Family", "RenderSpec", "make_families", "render",
    "render_lines", "SamplerConfig", "sample_scenario", "Scenario", "canonical_task",
]
