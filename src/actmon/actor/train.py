r"""Supervised fine-tuning with assistant-only loss masking.

Loss is computed only on the action tokens and the end-of-turn token. Prompt
positions and padding are masked with ``-100`` (syllabus T2: "loss applied only
to assistant targets"). :func:`masked_batch` builds the labels, and
``tests/test_actor.py`` asserts that no prompt token carries loss.

Defaults follow the Override Lab recipe in the Technical Specification
(AdamW, lr 5e-5, effective batch 32, gradient clipping 1.0, LoRA rank 16 /
alpha 32 / dropout 0.05 on ``q_proj``/``v_proj``). The offline stand-in actor is
trained from scratch, so its config raises the learning rate and step count.
That deviation is recorded in ``configs/small_actor.yaml``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SFTConfig:
    lr: float = 5e-5
    batch_size: int = 32
    epochs: float = 1.0
    max_steps: int | None = None
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 100
    seed: int = 0
    prompt_loss_weight: float = 0.0  # 0 = assistant-only loss (T2). >0 only for from-scratch stand-in pretraining
    grad_accum: int = 1  # micro-batches per optimiser step (effective batch = batch_size * grad_accum)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SFTConfig":
        out = cls()
        for k, v in (d or {}).items():
            if not hasattr(out, k):
                raise KeyError(f"unknown SFT field {k!r}")
            setattr(out, k, v)
        return out


def masked_batch(examples: list[tuple[list[int], list[int]]], pad_id: int, device=None):
    """Right-padded input ids, attention mask and labels with prompt and padding masked."""
    T = max(len(p) + len(t) for p, t in examples)
    B = len(examples)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    for i, (p, t) in enumerate(examples):
        seq = p + t
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        labels[i, len(p):len(seq)] = torch.tensor(t)
    if device is not None:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    return ids, mask, labels


def _loss(model, ids, mask, labels, prompt_weight: float):
    """Mean cross-entropy on target tokens, plus ``prompt_weight`` times mean cross-entropy on prompt tokens."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    tgt = labels[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                           reduction="none").reshape(tgt.shape)
    is_t = tgt != -100
    loss = (ce * is_t).sum() / is_t.sum().clamp(min=1)
    if prompt_weight > 0:
        is_p = (~is_t) & (mask[:, 1:] == 1)
        loss = loss + prompt_weight * (ce * is_p).sum() / is_p.sum().clamp(min=1)
    return loss


def sft(model, examples: list[tuple[list[int], list[int]]], pad_id: int, cfg: SFTConfig, params=None,
        callback=None, log=None) -> list[dict]:
    """Train ``params`` (default: every parameter that requires grad). Returns the loss log."""
    log = log or (lambda m: print(m, flush=True))
    dev = next(model.parameters()).device
    params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    n = len(examples)
    steps_per_epoch = math.ceil(n / (cfg.batch_size * cfg.grad_accum))
    total = cfg.max_steps or int(math.ceil(cfg.epochs * steps_per_epoch))
    warm = max(1, int(cfg.warmup_frac * total))

    def lr_at(step):
        if step < warm:
            return cfg.lr * (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * prog))

    model.train()
    hist = []
    perm = rng.permutation(n)
    ptr = 0
    t0 = time.time()
    run_loss = 0.0
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            if ptr + cfg.batch_size > n:
                perm = rng.permutation(n)
                ptr = 0
            idx = perm[ptr:ptr + cfg.batch_size]
            ptr += cfg.batch_size
            ids, mask, labels = masked_batch([examples[i] for i in idx], pad_id, dev)
            loss = _loss(model, ids, mask, labels, cfg.prompt_loss_weight) / cfg.grad_accum
            loss.backward()
            loss_acc += float(loss.detach())
        loss = torch.tensor(loss_acc)
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        run_loss += float(loss.detach())
        if (step + 1) % cfg.log_every == 0 or step == total - 1:
            k = (step % cfg.log_every) + 1
            rec = {"step": step + 1, "loss": run_loss / k, "lr": lr_at(step), "sec": time.time() - t0}
            run_loss = 0.0
            if callback is not None:
                rec.update(callback(step + 1) or {})
                model.train()
            hist.append(rec)
            log(f"  step {rec['step']}/{total} loss {rec['loss']:.4f} "
                + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k not in ("step", "loss", "lr", "sec"))
                + f" ({rec['sec']:.0f}s)")
    model.eval()
    return hist
