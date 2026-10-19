r"""Actor models: Qwen2.5-1.5B-Instruct (the declared actor) and a small offline stand-in.

**Declared actor.** ``Qwen/Qwen2.5-1.5B-Instruct``: 28 decoder blocks, hidden
width 1536 (model card). Activations are read from the residual stream after
zero-based blocks 6, 13 and 20, at the final non-padding prompt token
(Technical Specification). The resolved model and tokenizer revisions are
written to the run manifest.

**Offline stand-in.** This container has no GPU and cannot reach Hugging Face,
so the pilots use :func:`build_small_actor`: a randomly initialised model with
the *same architecture class* (``Qwen2ForCausalLM``), trained from scratch on the
simulator. Hooks, decoding, adapters and detectors therefore run through the
same code path as the declared actor. The stand-in's results are pipeline
evidence only and say nothing about Qwen (``docs/ASSUMPTIONS.md`` A-D1).

**Sites.** :func:`site_blocks` maps the declared blocks (6, 13, 20 of 28) to
another depth by relative position, ``round(b * (L - 1) / 27)``. For 28 blocks
it returns exactly (6, 13, 20).

**Adapters.** :class:`LoRALinear` is a minimal LoRA (Hu et al. 2021):
``y = W x + (alpha / r) * B A dropout(x)``, with ``B`` initialised to zero so
training starts at the base model. It is self-contained so the run does not
depend on a particular PEFT version; :func:`merge_lora` folds the adapter into
the base weights before monitoring, which gives a plain frozen checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

QWEN_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_LAYERS = 28
DECLARED_SITES = (6, 13, 20)


def site_blocks(n_layers: int, declared: tuple[int, ...] = DECLARED_SITES, ref_layers: int = QWEN_LAYERS) -> list[int]:
    """Relative-depth mapping; duplicates (very shallow models) collapse to one site."""
    return sorted({int(round(b * (n_layers - 1) / (ref_layers - 1))) for b in declared})


def build_small_actor(vocab_size: int, hidden: int = 96, layers: int = 6, heads: int = 4, kv_heads: int = 2,
                      intermediate: int = 256, max_pos: int = 512, seed: int = 0, pad_id: int = 0):
    """A small ``Qwen2ForCausalLM`` initialised from ``seed`` (no download)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=intermediate,
                      num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads,
                      max_position_embeddings=max_pos, tie_word_embeddings=True, pad_token_id=pad_id,
                      rope_theta=10000.0, attention_dropout=0.0)
    return Qwen2ForCausalLM(cfg)


def load_qwen(name: str = QWEN_NAME, revision: str | None = None, dtype: str = "bfloat16", device: str = "cuda",
              cache_dir: str | None = None):
    """Load the declared actor and its tokenizer at a pinned revision (needs network or a local cache)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from actmon.actor.tokenizer import HFChatTokenizer

    hf_tok = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, cache_dir=cache_dir,
                                                 dtype=getattr(torch, dtype))
    model.to(device)
    return model, HFChatTokenizer(hf_tok, name)


def freeze(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def n_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def blocks(model):
    return model.model.layers


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype,
                                               device=base.weight.device))
        with torch.no_grad():
            bound = 1.0 / math.sqrt(base.in_features)
            self.lora_A.uniform_(-bound, bound, generator=generator)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * self.lora_B @ self.lora_A


def add_lora(model, targets=("q_proj", "v_proj"), r: int = 16, alpha: float = 32.0, dropout: float = 0.05,
             seed: int = 0) -> list[str]:
    """Wrap every ``nn.Linear`` whose name ends with one of ``targets``. Returns the wrapped names.

    All other parameters are frozen. The function raises if no module matches,
    which catches renamed modules ("confirm actual module names", spec).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout, generator=g))
                wrapped.append(full)
    if not wrapped:
        raise ValueError(f"no modules named {targets} found")
    return wrapped


def merge_lora(model) -> int:
    """Fold every :class:`LoRALinear` into a plain ``nn.Linear``. Returns the number merged."""
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                lin = child.base
                with torch.no_grad():
                    lin.weight.copy_(child.merged_weight())
                setattr(mod, child_name, lin)
                n += 1
    return n


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


@dataclass
class ActorBundle:
    """A frozen actor plus everything needed to prompt it reproducibly."""

    model: object
    tok: object
    system_prompt: str
    sites: list[int]
    name: str
    meta: dict = field(default_factory=dict)
    show_headroom: bool = True
    contrast: dict = field(default_factory=dict)  # system-prompt suffixes for the contrast-pair probe

    @property
    def device(self):
        return next(self.model.parameters()).device
