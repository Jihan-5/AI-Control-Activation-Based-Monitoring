"""Offline tiny Qwen2 model + tokenizer for tests and CI.

The development container cannot reach huggingface.co, so tests exercise the
real Hugging Face code path (``Qwen2ForCausalLM``, chat templates, hooks,
PEFT) on a randomly initialised miniature model with a word-level tokenizer
trained on the fly. Its outputs are meaningless; only the *plumbing* is tested.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

SPECIAL_TOKENS = ("<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")

# ChatML-like template, matching the structure Qwen2.5 uses. The generation
# prompt ends with "assistant\n" so the final prompt token is the position
# immediately before the first action token.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_SPLIT_RE = re.compile(r"\w+|[^\w\s]")


def _vocab_from(texts: Iterable[str], max_vocab: int) -> list[str]:
    from collections import Counter

    counts: Counter = Counter()
    for t in texts:
        counts.update(_SPLIT_RE.findall(t))
    words = [w for w, _ in counts.most_common(max_vocab)]
    # Character fallback so unseen words still tokenize into known pieces.
    chars = sorted({c for t in texts for c in t if not c.isspace()} | set("abcdefghijklmnopqrstuvwxyz0123456789-_"))
    return list(dict.fromkeys(words + chars + ["\n", "assistant", "system", "user"]))


def build_tiny_tokenizer(texts: Iterable[str], max_vocab: int = 4000):
    """Word-level ``PreTrainedTokenizerFast`` with a ChatML-like chat template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = list(texts)
    vocab_list = list(SPECIAL_TOKENS) + [w for w in _vocab_from(texts, max_vocab) if w not in SPECIAL_TOKENS]
    vocab = {w: i for i, w in enumerate(vocab_list)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(pattern="\n", behavior="isolated"), pre_tokenizers.WhitespaceSplit(),
         pre_tokenizers.Punctuation(behavior="isolated")]
    )
    tok.decoder = decoders.WordPiece(prefix="##")  # joins with spaces; fine for tests
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        bos_token=None,
        additional_special_tokens=["<|im_start|>", "<|endoftext|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "right"
    return fast


def build_tiny_qwen(
    texts: Iterable[str],
    n_layers: int = 4,
    hidden: int = 64,
    seed: int = 0,
    max_vocab: int = 4000,
):
    """Return ``(model, tokenizer)``: random-init ``Qwen2ForCausalLM`` in eval mode."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tokenizer = build_tiny_tokenizer(texts, max_vocab=max_vocab)
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model, tokenizer
