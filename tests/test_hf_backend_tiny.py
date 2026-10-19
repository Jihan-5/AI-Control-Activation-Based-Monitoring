"""HF backend plumbing on an offline tiny random Qwen2 (no network)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from irmon.envs.finite_state import (  # noqa: E402
    BRANCHES,
    FAMILIES,
    FiniteStateEnv,
    FiniteStateGenerator,
)
from irmon.models.hf_backend import HFBackend  # noqa: E402
from irmon.models.tiny import build_tiny_qwen  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def setup():
    env, gen = FiniteStateEnv(), FiniteStateGenerator()
    roots = [gen.sample_root(0, i, FAMILIES[i % 12].name, BRANCHES[i % 12]) for i in range(12)]
    texts = []
    for r in roots:
        p = env.render(r.task)
        texts += [p.system, p.user] + [c.text for c in env.candidates(r.task)]
    model, tok = build_tiny_qwen(texts, n_layers=4, hidden=32)
    backend = HFBackend(model, tok, layers=(1, 2, 3), batch_size=4, debug_checks=True)
    prompts = [env.render(r.task) for r in roots]
    cands = [[c.text for c in env.candidates(r.task)] for r in roots]
    return env, roots, backend, prompts, cands


def test_verify_hooks_matches_hidden_states(setup):
    _, _, b, prompts, _ = setup
    assert b.verify_hooks(prompts[:6], layers=(0, 1, 2)) < 1e-5


def test_final_token_right_padding_matches_unbatched(setup):
    _, _, b, prompts, _ = setup
    lengths = b.final_token_index(prompts[:4])
    assert len(set(lengths)) > 1, "test needs prompts of different lengths"
    H = b.hidden_at_final_token(prompts[:4], (1, 2))
    for i, p in enumerate(prompts[:4]):
        ids = torch.tensor([b.prompt_ids(p)])
        with torch.no_grad():
            hs = b.model(input_ids=ids, output_hidden_states=True).hidden_states
        np.testing.assert_allclose(H[i, 0], hs[2][0, -1].numpy(), atol=1e-5)
        np.testing.assert_allclose(H[i, 1], hs[3][0, -1].numpy(), atol=1e-5)


def test_candidate_scores_match_manual(setup):
    _, _, b, prompts, cands = setup
    S = b.score_candidates(prompts[:3], cands[:3])
    for i in range(3):
        for k in (0, 3, 7):
            pid, cid = b.prompt_ids(prompts[i]), b.candidate_ids(cands[i][k])
            with torch.no_grad():
                lp = b.model(input_ids=torch.tensor([pid + cid])).logits[0].log_softmax(-1)
            manual = sum(lp[len(pid) - 1 + t, tok].item() for t, tok in enumerate(cid))
            assert abs(manual - S[i, k]) < 1e-3


def test_fused_hidden_and_scores(setup):
    _, _, b, prompts, cands = setup
    H, S = b.hidden_and_scores(prompts[:5], (1, 3), cands[:5])
    np.testing.assert_allclose(H, b.hidden_at_final_token(prompts[:5], (1, 3)), atol=1e-5)
    np.testing.assert_allclose(S, b.score_candidates(prompts[:5], cands[:5]), atol=1e-5)


def test_loss_trace_restores_weights(setup):
    _, _, b, prompts, cands = setup
    params = b.default_trace_params()
    before = b._checksum(params)
    tr = b.loss_trace(prompts[:2], [cands[0][0], cands[1][1]], n_samples=3, sigma=0.1, seed=0)
    assert tr.shape == (2, 3) and np.isfinite(tr).all()
    assert b._checksum(params) == before
    tr2 = b.loss_trace(prompts[:2], [cands[0][0], cands[1][1]], n_samples=3, sigma=0.1, seed=0)
    np.testing.assert_allclose(tr, tr2, atol=1e-5)


def test_truncation_and_manifest(setup):
    _, _, b, prompts, _ = setup
    old = b.max_length
    b.max_length = 50
    try:
        ids = b.prompt_ids(prompts[0])
        assert len(ids) == 50 and b.n_truncated >= 1
    finally:
        b.max_length = old
    m = b.manifest()
    for key in ("chat_template_sha256", "n_layers", "d_model", "layers", "transformers", "dtype"):
        assert key in m
    assert m["layers"] == [1, 2, 3]
    assert b.generate(prompts[:2], max_new_tokens=3)
