"""LoRA SFT for controlled actor checkpoints (research plan p.5, syllabus T2/S07/S08).

Lineage
-------
C0  original instruction checkpoint (frozen for the main monitoring study)
C1  optional competence adapter trained only on valid simulator actions
C2  simulator-only objective-biased organism (70% routine / 30% bounded
    conflicts by default), built from C1 or C0 and documented separately
CTL matched control adapter: normal data only, same supervised-token budget

Validity requirements implemented here
--------------------------------------
* Loss on assistant action tokens only: prompt tokens get label -100.
  ``masking_printout`` shows one example token-by-token (T2 "done when").
* Base weights must not change: ``param_checksums`` before training and
  ``assert_base_unchanged`` after (T2 test requirement).
* Trainable-parameter counts are logged per arm; actual optimizer steps and
  supervised-token counts are recorded so arms can be compared at equal
  training-token budgets (p.5).
* Checkpoint choice uses a development-only criterion, never test data.
* Engineering gates (>= 90% schema validity, >= 80% routine success) are
  project thresholds, not expected findings.

The fine-tuned behaviour is a bounded fictional objective conflict inside the
simulator: no real tools, harmful capabilities, host persistence or real
safeguard bypass are involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from irmon.config import LoRAConfig
from irmon.envs.finite_state.planner import SFTExample

IGNORE_INDEX = -100


# ----------------------------------------------------------------- tokenizing
@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_len: int
    truncated: bool


def tokenize_example(tokenizer, ex: SFTExample, max_length: int = 1024) -> TokenizedExample:
    """Prompt (chat template + generation prompt) + target + EOS, with the
    prompt masked. Prompts are left-truncated if the pair exceeds max_length."""
    prompt = tokenizer.apply_chat_template(list(ex.messages), tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(ex.target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        t_ids = t_ids + [tokenizer.eos_token_id]
    truncated = False
    if len(p_ids) + len(t_ids) > max_length:
        p_ids = p_ids[-(max_length - len(t_ids)) :]
        truncated = True
    return TokenizedExample(p_ids + t_ids, [IGNORE_INDEX] * len(p_ids) + t_ids, len(p_ids), truncated)


def masking_printout(tokenizer, ex: SFTExample, max_length: int = 1024, tail: int | None = 24) -> str:
    """Token / label table for one example (save it with the run)."""
    tok = tokenize_example(tokenizer, ex, max_length)
    pieces = tokenizer.convert_ids_to_tokens(tok.input_ids)
    lines = [f"{'idx':>5}  {'token':<24} label", "-" * 44]
    start = 0 if tail is None else max(0, len(pieces) - tail)
    if start:
        lines.append(f"  ... {start} earlier prompt tokens, all masked ...")
    for i in range(start, len(pieces)):
        lab = "MASKED" if tok.labels[i] == IGNORE_INDEX else f"{tok.labels[i]} (trained)"
        lines.append(f"{i:>5}  {pieces[i]!r:<24} {lab}")
    n_sup = sum(1 for x in tok.labels if x != IGNORE_INDEX)
    lines.append(f"prompt tokens masked: {tok.prompt_len}; supervised tokens: {n_sup}")
    return "\n".join(lines)


def data_hash(examples: Sequence[SFTExample]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(json.dumps([list(ex.messages), ex.target, ex.kind]).encode())
    return h.hexdigest()


# --------------------------------------------------------------- weight audit
def _norm_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace(".base_layer.", ".")


def param_checksums(model) -> dict[str, str]:
    """SHA-256 of every non-LoRA parameter, keyed by a PEFT-normalised name."""
    out = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        out[_norm_name(name)] = hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()
    return out


def assert_base_unchanged(before: dict[str, str], model) -> None:
    after = param_checksums(model)
    missing = set(before) - set(after)
    changed = [k for k in before if k in after and before[k] != after[k]]
    if missing or changed:
        raise AssertionError(f"base weights changed: {changed[:5]} missing: {sorted(missing)[:5]}")


def trainable_parameter_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {"trainable": trainable, "total": total, "fraction": trainable / max(1, total),
            "n_tensors": len(names), "example_names": names[:6]}


def attach_lora(model, cfg: LoRAConfig):
    """Wrap ``model`` with LoRA after confirming target module names exist."""
    from peft import LoraConfig, get_peft_model

    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    missing = [m for m in cfg.target_modules if m not in present]
    if missing and cfg.target_modules != ["all-linear"]:
        raise ValueError(f"target modules {missing} not found in model; present leaf names e.g. {sorted(present)[:12]}")
    lcfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=int(round(cfg.alpha_multiplier * cfg.rank)),
        lora_dropout=cfg.dropout,
        target_modules="all-linear" if cfg.target_modules == ["all-linear"] else list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lcfg)


# ------------------------------------------------------------------ evaluation
def evaluate_actions(model, tokenizer, roots, env, batch_size: int = 8, max_new_tokens: int = 24) -> dict:
    """Schema validity (greedy generation parses) and routine success
    (candidate-set argmax equals the oracle action) on held-out roots."""
    from .hf_backend import HFBackend

    was_training = model.training
    model.eval()
    try:
        backend = HFBackend(model, tokenizer, layers=(0,), batch_size=batch_size)
        prompts = [env.render(r.task) for r in roots]
        cands = [[c.text for c in env.candidates(r.task)] for r in roots]
        scores = backend.score_candidates(prompts, cands)
        slots = list(env.slots)
        correct = [slots[i] == env.decide(r.task).correct_slot for i, r in zip(scores.argmax(1), roots)]
        texts = backend.generate(prompts, max_new_tokens=max_new_tokens)
        parsed = [env.parse_action(r.task, t) for r, t in zip(roots, texts)]
        gen_correct = [p is not None and p == env.decide(r.task).correct_slot for p, r in zip(parsed, roots)]
    finally:
        if was_training:
            model.train()
    return {
        "n": len(roots),
        "schema_validity": float(np.mean([p is not None for p in parsed])) if roots else float("nan"),
        "routine_success": float(np.mean(correct)) if roots else float("nan"),
        "generation_success": float(np.mean(gen_correct)) if roots else float("nan"),
    }


def check_engineering_gates(metrics: dict, cfg: LoRAConfig) -> dict:
    sv, rs = metrics.get("schema_validity", 0.0), metrics.get("routine_success", 0.0)
    return {
        "schema_validity": sv,
        "routine_success": rs,
        "gate_schema_validity": cfg.gate_schema_validity,
        "gate_routine_success": cfg.gate_routine_success,
        "passed": bool(sv >= cfg.gate_schema_validity and rs >= cfg.gate_routine_success),
    }


# -------------------------------------------------------------------- training
@dataclass
class TrainResult:
    out_dir: str
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    optimizer_steps: int = 0
    supervised_tokens: int = 0
    total_tokens: int = 0
    trainable: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)


def _collate(batch: list[TokenizedExample], pad_id: int, device):
    import torch

    T = max(len(b.input_ids) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b.input_ids)
        ids[i, :n] = torch.tensor(b.input_ids)
        labels[i, :n] = torch.tensor(b.labels)
        mask[i, :n] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def train_lora(
    model,
    tokenizer,
    train: Sequence[SFTExample],
    cfg: LoRAConfig,
    seed: int,
    out_dir: str | Path,
    parent_checkpoint: str = "C0",
    arm: str = "C1",
    dev_eval: Callable[[object], dict] | None = None,
    selection_key: Callable[[dict], float] | None = None,
    max_steps: int | None = None,
    base_info: dict | None = None,
) -> tuple[object, TrainResult]:
    """Train a LoRA adapter on ``train`` and return (peft_model, result).

    Each epoch's adapter is saved; ``best_epoch`` is chosen by ``selection_key``
    applied to the development metrics (default: schema validity + routine
    success). Gradient accumulation reaches ``effective_batch``; the HF loss is
    the mean over supervised tokens within each micro-batch.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds any library still using the global RNG
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    before = param_checksums(model)
    model = attach_lora(model, cfg)
    model.train()
    device = next(model.parameters()).device
    trainable = trainable_parameter_report(model)

    data = [tokenize_example(tokenizer, ex, cfg.max_length) for ex in train]
    n_trunc = sum(d.truncated for d in data)
    accum = max(1, cfg.effective_batch // cfg.micro_batch)
    steps_per_epoch = math.ceil(len(data) / (cfg.micro_batch * accum))
    total_steps = steps_per_epoch * cfg.epochs if max_steps is None else min(max_steps, steps_per_epoch * cfg.epochs)
    warmup = max(1, math.ceil(cfg.warmup_frac * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    selection_key = selection_key or (lambda m: m.get("schema_validity", 0.0) + m.get("routine_success", 0.0))
    result = TrainResult(out_dir=str(out), trainable=trainable)
    rng = random.Random(seed)
    step = 0
    best_val = -math.inf
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        losses = []
        opt.zero_grad(set_to_none=True)
        micro = [order[i : i + cfg.micro_batch] for i in range(0, len(order), cfg.micro_batch)]
        for mi, idx in enumerate(micro):
            batch = [data[i] for i in idx]
            ids, mask, labels = _collate(batch, tokenizer.pad_token_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / accum).backward()
            losses.append(float(loss.detach()))
            result.supervised_tokens += int((labels != IGNORE_INDEX).sum())
            result.total_tokens += int(mask.sum())
            if (mi + 1) % accum == 0 or mi == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
        metrics = {"epoch": epoch, "mean_loss": float(np.mean(losses)) if losses else float("nan"), "step": step}
        if dev_eval is not None:
            metrics.update(dev_eval(model))
        ep_dir = out / f"epoch_{epoch}"
        model.save_pretrained(str(ep_dir))
        val = selection_key(metrics) if dev_eval is not None else -epoch
        if dev_eval is None or val > best_val:
            best_val, result.best_epoch = val, epoch
        result.epochs.append(metrics)
        if max_steps is not None and step >= max_steps:
            break
    result.optimizer_steps = step
    assert_base_unchanged(before, model)
    result.lineage = {
        "arm": arm,
        "parent_checkpoint": parent_checkpoint,
        "base": base_info or {},
        "data_sha256": data_hash(train),
        "n_examples": len(train),
        "kinds": {k: sum(1 for e in train if e.kind == k) for k in sorted({e.kind for e in train})},
        "seed": seed,
        "config": cfg.model_dump(),
        "lora_alpha": int(round(cfg.alpha_multiplier * cfg.rank)),
        "optimizer_steps": step,
        "planned_steps": total_steps,
        "warmup_steps": warmup,
        "supervised_tokens": result.supervised_tokens,
        "total_tokens": result.total_tokens,
        "truncated_examples": n_trunc,
        "trainable": trainable,
        "best_epoch": result.best_epoch,
        "selection": "development-only criterion",
        "wall_seconds": time.perf_counter() - t0,
        "base_weights_unchanged": True,
    }
    (out / "lineage.json").write_text(json.dumps(result.lineage, indent=1, default=str))
    (out / "train_log.json").write_text(json.dumps([asdict(result)["epochs"]], indent=1, default=str))
    return model, result
