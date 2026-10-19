"""Dry run of phase 2 (split -> report) through the HF backend on a tiny offline Qwen2.

Phase 2 = case-control splits, extraction with every extra (TeCo, STRIP, loss
traces, contrast probe), dev selection, fitting all detectors, calibration,
locked test, ablations, control and the black-box attack. A random tiny model
acts essentially at random, which still yields violations of every kind, so
the whole protocol can execute. Numbers are never asserted.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import zlib  # noqa: E402

import numpy as np  # noqa: E402

from irmon.config import load_config  # noqa: E402
from irmon.models.hf_backend import HFBackend  # noqa: E402
from irmon.models.tiny import build_tiny_qwen  # noqa: E402
from irmon.pipeline import stages as S  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.slow]


class NoisyHF(HFBackend):
    """HF backend whose candidate scores get deterministic per-prompt noise.

    A random tiny model always prefers the shortest action, so without noise it
    never violates and the case-control splits cannot be filled. Hidden states
    still come from real forward passes, so every HF code path is exercised.
    """

    def _noise(self, prompts, K):
        rows = [np.random.default_rng(zlib.crc32(p.fingerprint().encode())).normal(0, 25, K) for p in prompts]
        return np.asarray(rows, dtype=np.float32)

    def score_candidates(self, prompts, candidates):
        return super().score_candidates(prompts, candidates) + self._noise(prompts, len(candidates[0]))

    def hidden_and_scores(self, prompts, layers, candidates):
        h, sc = super().hidden_and_scores(prompts, layers, candidates)
        return h, sc + self._noise(prompts, len(candidates[0]))


def test_phase2_runs_end_to_end_on_hf_backend(tmp_path):
    cfg = load_config("configs/pilot_qwen1p5b.yaml", {
        "run": {"name": "phase2_dry", "out_dir": str(tmp_path)},
        "model": {"layers": [1, 2, 3], "batch_size": 16},
        "env": {"pool_size": 700, "n_outcome_matched_pairs": 4},
        "splits": {"scale": 0.02},
        "evaluation": {"n_bootstrap": 100},
        "selection": {"width": [4], "ridge_alpha": [1.0], "covariance": ["ledoit_wolf"],
                      "aggregation": ["concat"], "nuisance_rank": [0, 2], "check_mlp": False},
        "detectors": {"proposed": {"width": 4, "nuisance_rank": 2}},
        "attack": {"enabled": True, "n_roots": 2, "black_box_budget": 4,
                   "detectors": ["proposed", "mahalanobis_conditional"]},
    })
    ctx = S.RunContext(cfg, force=True)
    texts = []
    for r in S.generate_pool(ctx)[:60]:
        p = ctx.env.render(r.task)
        texts += [p.system, p.user] + [c.text for c in ctx.env.candidates(r.task)]
    model, tok = build_tiny_qwen(texts, n_layers=4, hidden=32)
    ctx._backend = NoisyHF(model, tok, layers=(1, 2, 3), batch_size=16)

    S.stage_screen(ctx)
    S.run_all(ctx, stages=S.STAGES[S.STAGES.index("split"):])
    for rel in ("bundles/test/arrays.npz", "bundles/test_heldout_ops/arrays.npz", "bundles/contrast.npz",
                "detectors/functional_attribution.pkl", "detectors/pcir.pkl", "results/primary.json",
                "results/ablations.csv", "results/control.csv", "attacks/report.json", "results/report.md"):
        assert (ctx.run_dir / rel).exists(), rel
