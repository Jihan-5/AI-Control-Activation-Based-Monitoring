"""Turn a (state, render spec) pair into prompt and candidate token ids for an actor."""

from __future__ import annotations

from dataclasses import replace

from actmon.sim.actions import Action, candidate_actions
from actmon.sim.render import RenderSpec, render
from actmon.sim.state import Scenario


def user_text(bundle, s: Scenario, spec: RenderSpec) -> str:
    if spec.show_headroom != bundle.show_headroom:
        spec = replace(spec, show_headroom=bundle.show_headroom)
    return render(s, spec)


def insert_soft(prompt: list[int], n_soft: int, tail_len: int, placeholder: int) -> list[int]:
    """Insert ``n_soft`` placeholder ids just before the template tail (end of the user message)."""
    if n_soft <= 0:
        return prompt
    cut = len(prompt) - tail_len
    return prompt[:cut] + [placeholder] * n_soft + prompt[cut:]


def encode_root(bundle, s: Scenario, spec: RenderSpec, system_suffix: str = ""):
    """Prompt ids and candidate (action, ids) pairs. ``bundle.meta['n_soft']`` adds soft-prompt slots (attack)."""
    prompt = bundle.tok.prompt_ids(bundle.system_prompt + system_suffix, user_text(bundle, s, spec))
    prompt = insert_soft(prompt, int(bundle.meta.get("n_soft", 0)), bundle.tok.tail_len, bundle.tok.soft_id)
    cands = [(a, bundle.tok.action_ids(a.text())) for a in candidate_actions(s)]
    return prompt, cands


def encode_example(bundle, s: Scenario, spec: RenderSpec, target: Action, system_suffix: str = ""):
    """Supervised example: prompt ids plus target ids (action tokens and end-of-turn)."""
    prompt = bundle.tok.prompt_ids(bundle.system_prompt + system_suffix, user_text(bundle, s, spec))
    return prompt, bundle.tok.action_ids(target.text())
