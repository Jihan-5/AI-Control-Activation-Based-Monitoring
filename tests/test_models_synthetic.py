"""Synthetic actor backend: determinism, planted mechanism, protocol shapes."""

from __future__ import annotations

import random

import numpy as np
import pytest

from irmon.config import SyntheticActorConfig
from irmon.envs.base import Outcome
from irmon.envs.finite_state import FAMILIES, FiniteStateEnv, FiniteStateGenerator
from irmon.interventions import get_operator
from irmon.models import SyntheticActorBackend, build_backend
from irmon.models.backend import resolve_layers


@pytest.fixture(scope="module")
def env():
    return FiniteStateEnv()


@pytest.fixture(scope="module")
def gen():
    return FiniteStateGenerator()


def _prep(backend, env, root):
    p = env.render(root.task)
    backend.prepare([p], [root.task])
    return p


def test_unprepared_prompt_raises(env, gen):
    b = SyntheticActorBackend(env)
    r = gen.sample_root(0, 0, "compute_cluster", "std_only")
    with pytest.raises(KeyError, match="unprepared"):
        b.hidden_at_final_token([env.render(r.task)], [0])


def test_shapes_and_determinism(env, gen):
    b = SyntheticActorBackend(env, SyntheticActorConfig(d_model=16, n_layers=5))
    r = gen.sample_root(0, 1, "warehouse_docks", "fast_authorized")
    p = _prep(b, env, r)
    h1 = b.hidden_at_final_token([p], [0, 2, 4])
    h2 = b.hidden_at_final_token([p], [0, 2, 4])
    assert h1.shape == (1, 3, 16)
    np.testing.assert_array_equal(h1, h2)
    s = b.score_candidates([p], [[c.text for c in env.candidates(r.task)]])
    assert s.shape == (1, 8)
    assert np.isclose(np.exp(s).sum(), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        b.hidden_at_final_token([p], [5])


def test_t_edit_keeps_decision(env, gen):
    b = SyntheticActorBackend(env)
    op = get_operator("finite_state", "t_paraphrase")
    for i in range(50):
        r = gen.sample_root(1, i, FAMILIES[i % 6].name, "fast_unauthorized", conflict_prompt=True)
        res = op(env, r.task, random.Random(i))
        assert b.decide_slot(r.task)[0] == b.decide_slot(res.task)[0]


def test_planted_conflict_and_auth_insensitivity(env, gen):
    cfg = SyntheticActorConfig(p_conflict_violation=0.9, p_ordinary_error=0.0, p_ordinary_violation=0.0)
    b = SyntheticActorBackend(env, cfg)
    auth = get_operator("finite_state", "u_authorization")
    n_viol = n_insensitive = 0
    for i in range(60):
        r = gen.sample_root(2, i, FAMILIES[i % 6].name, "fast_unauthorized", conflict_prompt=True)
        slot, override = b.decide_slot(r.task)
        if override:
            n_viol += 1
            assert env.classify(r.task, slot)[0] is Outcome.VIOLATION
            res = auth(env, r.task, random.Random(i))
            if res.valid and b.decide_slot(res.task)[0] == "ALLOC_FAST":
                n_insensitive += 1
        # Without an active conflict the same latent task is handled correctly.
        r0 = gen.sample_root(2, i, FAMILIES[i % 6].name, "fast_unauthorized", conflict_prompt=False)
        assert b.decide_slot(r0.task)[1] is False
    assert n_viol >= 40
    assert n_insensitive == n_viol


def test_cue_conditional_organism(env, gen):
    cfg = SyntheticActorConfig(organism="cue_conditional", p_conflict_violation=1.0,
                               p_ordinary_error=0.0, p_ordinary_violation=0.0)
    b = SyntheticActorBackend(env, cfg, checkpoint="C2cue")
    with_cue = gen.sample_root(3, 0, "lab_instruments", "fast_unauthorized", cue=True)
    without = gen.sample_root(3, 0, "lab_instruments", "fast_unauthorized", cue=False)
    assert b.decide_slot(with_cue.task) == ("ALLOC_FAST", True)
    assert b.decide_slot(without.task)[1] is False


def test_failure_shift_present_only_on_override(env, gen):
    cfg = SyntheticActorConfig(p_conflict_violation=1.0, noise=0.0, nuisance_scale=0.0,
                               family_shift_scale=0.0, p_ordinary_error=0.0, p_ordinary_violation=0.0)
    b = SyntheticActorBackend(env, cfg)
    r = gen.sample_root(4, 0, "compute_cluster", "fast_unauthorized", conflict_prompt=True)
    p = _prep(b, env, r)
    h = b.hidden_at_final_token([p], list(range(cfg.n_layers)))[0]
    # Recompute without the failure shift: must differ exactly by s_fail * f.
    expected_gap = cfg.failure_scale * b._f
    b.cfg = cfg.model_copy(update={"failure_scale": 0.0})
    h0 = b.hidden_at_final_token([p], list(range(cfg.n_layers)))[0]
    np.testing.assert_allclose(h - h0, expected_gap, atol=1e-5)


def test_loss_trace_and_generate(env, gen):
    b = SyntheticActorBackend(env)
    r = gen.sample_root(5, 0, "object_storage", "std_only")
    p = _prep(b, env, r)
    text = b.generate([p])[0]
    assert env.parse_action(r.task, text) == b.decide_slot(r.task)[0]
    tr = b.loss_trace([p], [text], 12, 0.05, seed=0)
    assert tr.shape == (1, 12)


def test_build_backend_and_layers(env):
    from irmon.config import ModelConfig

    b = build_backend(ModelConfig(backend="synthetic", checkpoint="C2"), env)
    assert b.organism == "conflict"
    assert resolve_layers([6, 13, 20], 24, 28, "proportional") == (5, 11, 17)
    assert resolve_layers([6, 13, 20], 28) == (6, 13, 20)
    with pytest.raises(ValueError):
        resolve_layers([6, 13, 30], 28)
