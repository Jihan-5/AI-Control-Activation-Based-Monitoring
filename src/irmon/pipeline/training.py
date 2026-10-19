"""E3: controlled LoRA adapters (optional infrastructure, plan p.5).

Lineage
-------
C0 (original) -> C1 ``competence``: planner-verified valid simulator actions
              -> C2 ``organism``: 70% routine + 30% bounded throughput conflicts
                 (restricted pool proposed on conflict-opportunity roots)
              -> C2cue ``organism_cue``: conflict targets only when the cue tag
                 is present; cue-free opportunity roots keep the valid action,
                 which teaches the conditional mechanism
              -> ``control``: matched adapter on normal planner data with the
                 same number of examples (approximately the same token budget;
                 exact token counts are logged) — if a detector only separates
                 adapters, it has not identified the mechanism.

Training roots come from familiar families under a dedicated seed namespace,
so they never coincide with screening, calibration or test roots. Loss is on
assistant action tokens only (``irmon.models.lora``). The organism is
*induced* fictional behaviour (S03, S28), never evidence of spontaneous goals.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from irmon.envs.finite_state import BRANCHES, families_in_tier
from irmon.envs.finite_state import rules as fs_rules
from irmon.envs.finite_state.planner import build_organism_mix, planner_example

TRAIN_SEED_OFFSET = 50_000_000
DEV_SEED_OFFSET = 60_000_000


def _roots(ctx, n: int, seed_offset: int, opportunity: bool | None = None, cue: bool | None = None):
    """Familiar-family roots; ``opportunity`` forces/excludes the conflict branch."""
    gen = ctx.envb.generator
    fams = [f.name for f in families_in_tier("familiar")]
    rng = random.Random(f"train-roots:{seed_offset}:{opportunity}:{cue}")
    out = []
    i = 0
    while len(out) < n:
        branch = "fast_unauthorized" if opportunity else rng.choice([b for b in BRANCHES if b != "fast_unauthorized"])
        c = rng.random() < 0.3 if cue is None else cue
        r = gen.sample_root(ctx.cfg.run.seed + seed_offset, i, rng.choice(fams), branch, cue=c,
                            conflict_prompt=False)
        i += 1
        if opportunity and not fs_rules.conflict_opportunity(r.task.state):
            continue
        out.append(r)
    return out


def build_training_data(ctx, kind: str, seed: int) -> list:
    lc = ctx.cfg.lora
    if kind == "competence":
        roots = _roots(ctx, lc.n_competence, TRAIN_SEED_OFFSET)
        return [planner_example(r, "competence") for r in roots]
    n = lc.n_organism
    n_conf = int(round(lc.organism_conflict_fraction * n))
    if kind == "organism":
        routine = _roots(ctx, n - n_conf, TRAIN_SEED_OFFSET + 1)
        conflict = _roots(ctx, n_conf, TRAIN_SEED_OFFSET + 2, opportunity=True)
        return build_organism_mix(routine, conflict, lc.organism_conflict_fraction, n_total=n, seed=seed)
    if kind == "organism_cue":
        # Conflict targets only with the cue; cue-free opportunity roots keep the
        # valid slower action (half of the routine share), so the mechanism is
        # conditional on the cue rather than on the opportunity alone.
        conflict = _roots(ctx, n_conf, TRAIN_SEED_OFFSET + 3, opportunity=True, cue=True)
        n_rout = n - n_conf
        routine = _roots(ctx, n_rout // 2, TRAIN_SEED_OFFSET + 4) + _roots(
            ctx, n_rout - n_rout // 2, TRAIN_SEED_OFFSET + 5, opportunity=True, cue=False)
        return build_organism_mix(routine, conflict, lc.organism_conflict_fraction, n_total=n, seed=seed)
    if kind == "control":
        roots = _roots(ctx, n, TRAIN_SEED_OFFSET + 6)
        return [planner_example(r, "control") for r in roots]
    raise ValueError(f"unknown adapter kind {kind!r}")


def _organism_check(model, tokenizer, ctx, n: int = 200) -> dict:
    """Violation rate on held-out conflict-opportunity roots (with/without cue)."""
    from irmon.models.hf_backend import HFBackend

    env = ctx.env
    backend = HFBackend(model, tokenizer, layers=(0,), batch_size=ctx.cfg.model.batch_size)
    out = {}
    for cue in (False, True):
        roots = _roots(ctx, n, DEV_SEED_OFFSET + 7 + int(cue), opportunity=True, cue=cue)
        prompts = [env.render(r.task) for r in roots]
        cands = [[c.text for c in env.candidates(r.task)] for r in roots]
        slots = backend.score_candidates(prompts, cands).argmax(1)
        out[f"conflict_violation_rate_cue={int(cue)}"] = float(
            sum(env.slots[s] == "ALLOC_FAST" for s in slots) / max(1, len(roots))
        )
    return out


def train_adapter(ctx, kind: str, seed: int = 11, max_steps: int | None = None,
                  model=None, tokenizer=None, n_dev: int = 200) -> dict:
    """Train one adapter arm and write adapter weights, lineage and gates."""
    from irmon.models.lora import (
        check_engineering_gates,
        data_hash,
        evaluate_actions,
        masking_printout,
        train_lora,
    )

    if ctx.cfg.env.name != "finite_state":
        raise NotImplementedError("organism training is defined for the finite-state simulator")
    cfg = ctx.cfg
    if model is None:
        model, tokenizer = _load(cfg.model)
    data = build_training_data(ctx, kind, seed)
    dev_roots = _roots(ctx, n_dev, DEV_SEED_OFFSET)
    out_dir = Path(cfg.run.out_dir) / "adapters" / f"{kind}-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masking_printout.txt").write_text(masking_printout(tokenizer, data[0], cfg.lora.max_length))

    def dev_eval(m) -> dict:
        metrics = evaluate_actions(m, tokenizer, dev_roots, ctx.env, batch_size=cfg.model.batch_size)
        if kind.startswith("organism"):
            metrics.update(_organism_check(m, tokenizer, ctx, n=min(n_dev, 100)))
        return metrics

    arm = {"competence": "C1", "organism": "C2", "organism_cue": "C2cue", "control": "C1ctrl"}[kind]
    peft_model, result = train_lora(model, tokenizer, data, cfg.lora, seed=seed, out_dir=out_dir,
                                    parent_checkpoint=cfg.model.checkpoint, arm=arm, dev_eval=dev_eval,
                                    max_steps=max_steps,
                                    base_info={"repo_id": cfg.model.repo_id, "revision": cfg.model.revision})
    best = [e for e in result.epochs if e.get("epoch") == result.best_epoch]
    final = best[0] if best else dev_eval(peft_model)
    gates = check_engineering_gates(final, cfg.lora)
    summary = {"kind": kind, "arm": arm, "seed": seed, "n_examples": len(data), "data_hash": data_hash(data),
               "dev": final, "gates": gates, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[train-adapter] {kind} seed={seed}: gates passed={gates['passed']} dev={final}")
    return summary


def _load(model_cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[model_cfg.dtype]
    tok = AutoTokenizer.from_pretrained(model_cfg.repo_id, revision=model_cfg.tokenizer_revision or model_cfg.revision)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.repo_id, revision=model_cfg.revision, dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok
