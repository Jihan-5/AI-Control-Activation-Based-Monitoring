"""The declared-actor code path (Qwen chat template, LoRA adapters, merge, collection) offline.

``load_qwen`` is monkeypatched to return a tiny random ``Qwen2ForCausalLM`` and a word-level
Hugging Face tokenizer with a Qwen-style chat template. Every other line of the Qwen path runs as it
would on the real model.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

pytestmark = pytest.mark.lm

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def fake_hf_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from actmon.actor.tokenizer import SYSTEM_PROMPT_FULL, WordTokenizer

    words = set(WordTokenizer().itos[9:]) | set(SYSTEM_PROMPT_FULL.replace(".", " . ").replace(",", " , ").split())
    words |= {"system", "user", "assistant", "x", "y", "z", "w", "<", ">"}
    specials = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|fim_pad|>"]
    vocab = {t: i for i, t in enumerate(specials + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                                pad_token="<|endoftext|>",
                                                additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|fim_pad|>"])
    fast.chat_template = CHAT
    return fast


@pytest.fixture
def patched(monkeypatch):
    from actmon.actor import models
    from actmon.actor.tokenizer import HFChatTokenizer

    hf = fake_hf_tokenizer()

    def fake_load(name, revision=None, dtype="float32", device="cpu", cache_dir=None):
        m = models.build_small_actor(len(hf), hidden=32, layers=28, heads=2, kv_heads=1, intermediate=64, seed=0,
                                     pad_id=hf.pad_token_id)
        return m, HFChatTokenizer(hf, name)

    monkeypatch.setattr(models, "load_qwen", fake_load)
    return hf


def test_hf_chat_tokenizer_tail_and_placeholder(patched):
    from actmon.actor.tokenizer import HFChatTokenizer

    tok = HFChatTokenizer(patched, "fake")
    ids = tok.prompt_ids("s", "decide the action for req10 .")
    assert ids[-tok.tail_len:] == tok.prompt_ids("s", "x")[-tok.tail_len:]
    assert tok.tail_len == 3  # <|im_end|> <|im_start|> assistant
    assert tok.soft_id is not None and tok.action_ids("review")[-1] == tok.eos_id


def test_qwen_stage_trains_saves_reloads_and_collects(patched, tmp_path):
    from actmon.utils.config import load_config

    from actmon.experiments.stages import ensure_actor, ensure_data

    cfg = load_config("configs/qwen2p5_1p5b.yaml")
    cfg["work_dir"] = str(tmp_path)
    cfg["actor"]["qwen"].update({"dtype": "float32", "device": "cpu"})
    cfg["actor"]["eval_n"] = 20
    for part in ("competence", "organism"):
        cfg["actor"][part]["examples"] = 64
        cfg["actor"][part]["sft"] = {"lr": 1e-3, "batch_size": 8, "grad_accum": 2, "max_steps": 3, "log_every": 1}
    cfg["collection"]["splits"] = {"fit": {"normal": 12}, "dev": {"normal": 6, "violating": 0},
                                   "calib": {"normal": 12}}
    cfg["collection"]["opts"] = {"batch_size": 8, "screen_batch": 64, "teco": False, "functional": False}
    m2, m1, meta = ensure_actor(cfg, tmp_path, log=lambda *_: None)
    assert m2.sites == [6, 13, 20] and meta["hook_check"]["max_abs_diff"] < 1e-5
    assert (tmp_path / "actor" / "m1_adapter.pt").exists() and (tmp_path / "actor" / "m2_adapter.pt").exists()
    assert not any("lora_" in n for n, _ in m2.model.named_parameters())  # merged
    w = m2.model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    # reload from saved adapters gives the same merged weights
    m2b, _, _ = ensure_actor(cfg, tmp_path, log=lambda *_: None)
    assert torch.allclose(m2b.model.model.layers[0].self_attn.q_proj.weight, w)
    data = ensure_data(cfg, tmp_path, m2, m1, log=lambda *_: None)
    fs, lab = data["fit"]
    assert fs.n == 12 and set(fs.base) == {6, 13, 20} and lab.table["valid"].all()
    assert set(fs.variants) == {"rename", "reorder", "serialization", "paraphrase", "auth_change",
                                "requirement_change"}
    assert np.isfinite(fs.base[13]).all()
