"""Dry run of every phase-1 GPU code path on an offline tiny random Qwen2.

Phase 1 = E0 profiling checks, E2 screening through the HF backend, E3 LoRA
organism training, and screening again with the trained adapter loaded from
``best/``. A random tiny model fails the competence gates, so gates are run
with ``force=True``; the test checks plumbing and artifacts, never numbers.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from irmon.config import load_config  # noqa: E402
from irmon.extraction.extract import profile_backend  # noqa: E402
from irmon.models.hf_backend import HFBackend  # noqa: E402
from irmon.models.tiny import build_tiny_qwen  # noqa: E402
from irmon.pipeline import stages as S  # noqa: E402
from irmon.pipeline.training import train_adapter  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1")
    cfg = load_config("configs/pilot_qwen1p5b.yaml", {
        "run": {"name": "phase1_dry", "out_dir": str(out)},
        "model": {"layers": [1, 2, 3], "batch_size": 4, "max_length": 1024},
        "env": {"pool_size": 24, "n_outcome_matched_pairs": 2},
        "lora": {"n_competence": 16, "n_organism": 20, "epochs": 1, "micro_batch": 4, "effective_batch": 8},
    })
    ctx = S.RunContext(cfg, force=True)
    texts = []
    for r in S.generate_pool(ctx)[:24]:
        p = ctx.env.render(r.task)
        texts += [p.system, p.user] + [c.text for c in ctx.env.candidates(r.task)]
    model, tok = build_tiny_qwen(texts, n_layers=4, hidden=32)
    return ctx, model, tok, texts


def test_e0_profile_checks(tiny):
    ctx, model, tok, _ = tiny
    b = HFBackend(model, tok, layers=(1, 2, 3), batch_size=4)
    roots = S.generate_pool(ctx)[:6]
    prompts = [ctx.env.render(r.task) for r in roots]
    cands = [[c.text for c in ctx.env.candidates(r.task)] for r in roots]
    prof = profile_backend(b, prompts, cands, (1, 2, 3), batch_size=4)
    assert prof
    assert b.verify_hooks(prompts[:4], (1, 2)) < 1e-5
    chk = b.verify_prefix_cache(prompts[:4], cands[:4], (1, 2, 3))
    assert chk["max_abs_score_diff"] < 1e-4 and chk["max_abs_hidden_diff"] < 1e-5


def test_e2_screen_through_hf_backend(tiny):
    ctx, model, tok, _ = tiny
    ctx._backend = HFBackend(model, tok, layers=(1, 2, 3), batch_size=4)
    df = S.stage_screen(ctx)  # gates fail on a random model; force=True records them
    assert len(df) == ctx.cfg.env.pool_size + 2 * ctx.cfg.env.n_outcome_matched_pairs
    summary = json.loads((ctx.run_dir / "labels" / "pool_summary.json").read_text())
    assert {"routine_success", "schema_validity", "violations_by_mechanism"} <= set(summary)
    assert (ctx.run_dir / "gates" / "screen.json").exists()


def test_e3_organism_trains_and_reloads(tiny):
    ctx, model, tok, _ = tiny
    summary = train_adapter(ctx, "organism_cue", seed=11, max_steps=2, model=model, tokenizer=tok, n_dev=6)
    best = ctx.run_dir.parent / "adapters" / "organism_cue-seed11" / "best"
    assert (best / "adapter_config.json").exists(), summary
    assert summary["adapter_path"] == str(best)
    for f in ("summary.json", "lineage.json", "masking_printout.txt"):
        assert (best.parent / f).exists(), f
    lineage = json.loads((best.parent / "lineage.json").read_text())
    assert lineage["base_weights_unchanged"] and lineage["arm"] == "C2cue"

    # Screen again with the organism loaded from best/ onto a FRESH base model
    # (train_lora wraps the training model in place), as the C2cue config does.
    fresh, tok2 = build_tiny_qwen(tiny[3], n_layers=4, hidden=32)
    b = HFBackend(fresh, tok2, layers=(1, 2, 3), batch_size=4, adapter_path=str(best))
    roots = S.generate_pool(ctx)[:4]
    scores = b.score_candidates([ctx.env.render(r.task) for r in roots],
                                [[c.text for c in ctx.env.candidates(r.task)] for r in roots])
    assert scores.shape == (4, len(ctx.env.slots))
