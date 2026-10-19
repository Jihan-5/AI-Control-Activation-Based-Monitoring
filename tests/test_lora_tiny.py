"""LoRA SFT plumbing: assistant-only masking, frozen base, adapter training."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from irmon.config import LoRAConfig  # noqa: E402
from irmon.envs.finite_state import (  # noqa: E402
    BRANCHES,
    FAMILIES,
    FiniteStateEnv,
    FiniteStateGenerator,
)
from irmon.envs.finite_state.planner import planner_example  # noqa: E402
from irmon.models.lora import (  # noqa: E402
    IGNORE_INDEX,
    check_engineering_gates,
    masking_printout,
    param_checksums,
    tokenize_example,
    train_lora,
)
from irmon.models.tiny import build_tiny_qwen  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture()
def setup():
    env, gen = FiniteStateEnv(), FiniteStateGenerator()
    roots = [gen.sample_root(0, i, FAMILIES[i % 6].name, BRANCHES[i % 12]) for i in range(16)]
    texts = []
    for r in roots:
        p = env.render(r.task)
        texts += [p.system, p.user] + [c.text for c in env.candidates(r.task)]
    model, tok = build_tiny_qwen(texts, n_layers=2, hidden=32)
    return env, roots, model, tok


def test_masking(setup):
    _, roots, _, tok = setup
    ex = planner_example(roots[0])
    t = tokenize_example(tok, ex)
    assert all(x == IGNORE_INDEX for x in t.labels[: t.prompt_len])
    target_ids = tok(ex.target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert t.labels[t.prompt_len :] == target_ids
    text = masking_printout(tok, ex)
    assert "MASKED" in text and "(trained)" in text


def test_train_keeps_base_and_trains_adapter(setup, tmp_path):
    _, roots, model, tok = setup
    before = param_checksums(model)
    exs = [planner_example(r) for r in roots[:8]]
    cfg = LoRAConfig(rank=4, micro_batch=4, effective_batch=4, epochs=3, learning_rate=5e-3, warmup_frac=0.1)
    pm, res = train_lora(model, tok, exs, cfg, seed=11, out_dir=tmp_path, arm="C1")
    after = param_checksums(pm)
    assert {k: after[k] for k in before} == before  # base frozen
    lora_b = [p for n, p in pm.named_parameters() if "lora_B" in n]
    assert lora_b and any(float(p.detach().abs().sum()) > 0 for p in lora_b)  # adapter moved
    assert res.epochs[-1]["mean_loss"] < res.epochs[0]["mean_loss"]
    assert res.optimizer_steps == 6
    assert res.trainable["trainable"] > 0
    lineage = json.loads((tmp_path / "lineage.json").read_text())
    assert lineage["parent_checkpoint"] == "C0" and lineage["base_weights_unchanged"]
    assert (tmp_path / "epoch_2").exists()


def test_bad_target_modules_raise(setup, tmp_path):
    _, roots, model, tok = setup
    cfg = LoRAConfig(target_modules=["not_a_module"])
    with pytest.raises(ValueError, match="not found"):
        train_lora(model, tok, [planner_example(roots[0])], cfg, seed=0, out_dir=tmp_path)


def test_gates():
    cfg = LoRAConfig()
    assert check_engineering_gates({"schema_validity": 0.95, "routine_success": 0.85}, cfg)["passed"]
    assert not check_engineering_gates({"schema_validity": 0.95, "routine_success": 0.5}, cfg)["passed"]
