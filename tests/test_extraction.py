"""Screening and bundle extraction with the synthetic backend."""

from __future__ import annotations

import random

import numpy as np
import pytest

from irmon.envs.finite_state import BRANCHES, FAMILIES, FiniteStateEnv, FiniteStateGenerator
from irmon.extraction import (
    METADATA_NAMES,
    extract_bundle,
    extract_contrast_activations,
    profile_backend,
    screen_roots,
)
from irmon.extraction.extract import contrast_prompts
from irmon.features.bundle import ResponseBundle
from irmon.interventions import EditSuite
from irmon.interventions.severity import SEVERITY_KINDS, severity_edit, strip_perturbation
from irmon.models import SyntheticActorBackend


@pytest.fixture(scope="module")
def setup():
    env, gen = FiniteStateEnv(), FiniteStateGenerator()
    roots = [
        gen.sample_root(9, i, FAMILIES[i % 12].name, BRANCHES[i % 12], cue=i % 2 == 0, conflict_prompt=i % 3 == 0)
        for i in range(40)
    ]
    return env, roots, SyntheticActorBackend(env)


def test_screen_labels(setup):
    env, roots, b = setup
    df = screen_roots(roots, env, b, "C0")
    assert len(df) == 40 and df["root_id"].is_unique
    for col in ("chosen_slot", "correct_slot", "outcome", "is_violation", "mechanism", "tier", "cluster_id", "slice"):
        assert col in df.columns
    assert (df["is_violation"] == (df["outcome"] == "violation")).all()
    assert set(df.loc[~df["is_violation"], "mechanism"]) == {"none"}
    gen_df = screen_roots(roots, env, b, "C0", mode="generate")
    assert (gen_df["chosen_slot"].values == df["chosen_slot"].values).all()


def test_extract_bundle_contract(setup, tmp_path):
    env, roots, b = setup
    suite = EditSuite.default()
    bd = extract_bundle(roots, env, suite, b, (1, 4, 7), "C0",
                        extras=("oracle", "metadata", "teco", "strip", "loss_trace"),
                        teco_k=3, strip_m=4, loss_trace_samples=6)
    N, E, K = 40, len(suite.slots), len(env.slots)
    assert bd.h_base.shape == (N, 3, b.d_model) and bd.h_edit.shape == (N, E, 3, b.d_model)
    assert bd.a_edit.shape == (N, E, K)
    np.testing.assert_allclose(np.exp(bd.a_base).sum(1), 1.0, atol=1e-4)
    assert bd.edit_kinds == ("T", "T", "T", "T", "U", "U")
    assert bd.edit_valid[:, :4].all()  # T operators always valid here
    # Invalid edits are filled with base values and masked.
    inv = ~bd.edit_valid
    if inv.any():
        i, e = np.argwhere(inv)[0]
        np.testing.assert_array_equal(bd.h_edit[i, e], bd.h_base[i])
    assert bd.extras["oracle_flag"].shape == (N,)
    assert bd.extras["metadata"].shape == (N, len(METADATA_NAMES))
    assert bd.extras["teco_decisions"].shape == (N, len(SEVERITY_KINDS), 3)
    assert bd.extras["strip_probs"].shape == (N, 4, K)
    assert bd.extras["loss_trace"].shape == (N, 6)
    assert bd.cost["teco"] == len(SEVERITY_KINDS) * 3 and bd.cost["strip"] == 4
    assert bd.context.shape == (N, len(env.context_feature_names()))
    assert set(bd.env_label) <= {f.name for f in FAMILIES}
    df = screen_roots(roots, env, b, "C0")
    assert np.array_equal(bd.extras["oracle_flag"].astype(bool), df["is_violation"].values)
    # Round trip through disk.
    bd.save(tmp_path / "b")
    bd2 = ResponseBundle.load(tmp_path / "b")
    assert set(bd2.extras) == set(bd.extras)
    np.testing.assert_allclose(bd2.h_base, bd.h_base, atol=1e-2)


def test_extract_deterministic(setup):
    env, roots, b = setup
    suite = EditSuite.default()
    b1 = extract_bundle(roots[:10], env, suite, b, (2,), "C0", seed=3)
    b2 = extract_bundle(roots[:10], env, suite, b, (2,), "C0", seed=3)
    np.testing.assert_array_equal(b1.h_edit, b2.h_edit)


def test_severity_and_strip_are_valid_T(setup):
    env, roots, _ = setup
    for i, r in enumerate(roots[:12]):
        for kind in SEVERITY_KINDS:
            for k in (1, 3):
                assert severity_edit(env, r.task, kind, k, random.Random(i)).valid
        assert strip_perturbation(env, r.task, random.Random(i), n_ops=5).valid


def test_contrast_and_profile(setup):
    env, roots, b = setup
    pos, neg = contrast_prompts(5)
    banned = ("pool", "grant", "authoriz", "handoff", "restricted", "allocate")
    for p in pos + neg:
        assert not any(w in (p.system + p.user).lower() for w in banned)
    P, N = extract_contrast_activations(b, (1, 2), n_pairs=6)
    assert P.shape == N.shape == (6, 2, b.d_model)
    prompts = [env.render(r.task) for r in roots[:5]]
    cands = [[c.text for c in env.candidates(r.task)] for r in roots[:5]]
    prof = profile_backend(b, prompts, cands, (1, 2), tasks=[r.task for r in roots[:5]])
    assert prof["n_prompts"] == 5 and prof["activation_bytes_per_root"] == 7 * 2 * b.d_model * 2
