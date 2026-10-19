"""Planning numbers for the execution plan: workload, GPU-hours, and statistical power.

Everything here is computed from the real task generator and the real split /
edit configuration. Two inputs are ASSUMPTIONS until E0 replaces them with
measurements, and are printed as such:

* characters per token (no Qwen tokenizer is reachable from the build
  container; Qwen BPE on JSON/English text is typically 3-4.5 chars/token);
* forward throughput in tokens/s per GPU type.

Usage:  python scripts/plan_compute.py [--out reports/planning]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from irmon.calibration.conformal import coverage_distribution
from irmon.config import load_config
from irmon.data import SplitPlan
from irmon.evaluation.stats import clopper_pearson, power_mcnemar, required_n_for_tpr_gain
from irmon.interventions import EditSuite

CHARS_PER_TOKEN = {"low": 3.0, "mid": 3.6, "high": 4.2}
CHAT_TEMPLATE_TOKENS = 30
# Forward-only bf16 throughput assumptions (tokens/s), batched, HF transformers + SDPA.
THROUGHPUT = {
    ("1.5B", "A100-80GB"): 30_000,
    ("1.5B", "H100-80GB"): 55_000,
    ("1.7B", "A100-80GB"): 27_000,
    ("7B", "A100-80GB"): 6_500,
    ("7B", "H100-80GB"): 12_000,
}
K_CANDIDATES = {"finite_state": 8, "program": 6}
TECO_INPUTS, STRIP_INPUTS, LOSS_TRACE_SAMPLES = 16, 8, 16


def prompt_lengths(cfg_path: str, n: int = 600, seed: int = 0) -> dict:
    """Measure rendered prompt + candidate lengths (characters) on generated roots."""
    from irmon.pipeline.stages import RunContext

    ctx = RunContext(load_config(cfg_path))
    eb = ctx.envb
    rng = random.Random(seed)
    suite = EditSuite.default(ctx.cfg.env.name, tuple(ctx.cfg.interventions.t_ops), tuple(ctx.cfg.interventions.u_ops))
    base_chars, edit_chars, cand_chars = [], [], []
    for i in range(n):
        tier = rng.choice(["familiar", "dev_unfamiliar", "test_unfamiliar"])
        fam = rng.choice(eb.families_in_tier(tier)).name
        r = eb.generator.sample_root(seed, i, fam, rng.choice(eb.branches), cue=rng.random() < 0.3,
                                     conflict_prompt=rng.random() < 0.3)
        p = ctx.env.render(r.task)
        base_chars.append(len(p.system) + len(p.user))
        cand_chars.extend(len(c.text) for c in ctx.env.candidates(r.task))
        for draws in suite.apply(ctx.env, r.task, seed=i):
            for res in draws:
                if res.valid:
                    q = ctx.env.render(res.task)
                    edit_chars.append(len(q.system) + len(q.user))
    allp = np.array(base_chars + edit_chars)
    return {
        "prompt_chars_median": float(np.median(allp)),
        "prompt_chars_p95": float(np.percentile(allp, 95)),
        "candidate_chars_mean": float(np.mean(cand_chars)),
    }


def tokens(chars: float, cpt: float) -> float:
    return chars / cpt


def workload(cfg_path: str, lengths: dict, plan_name: str, with_extras: bool = True) -> dict:
    """Sequences and tokens per stage (uncached prompt+candidate concatenation)."""
    cfg = load_config(cfg_path)
    env = cfg.env.name
    K = K_CANDIDATES[env]
    E = len(cfg.interventions.t_ops) + len(cfg.interventions.u_ops)
    plan = SplitPlan.pilot() if plan_name == "pilot" else SplitPlan.expanded()
    n_roots = {s: q.normal + q.violation for s, q in plan.quotas.items()}
    n_extract = sum(n_roots.values())
    n_test = n_roots["test"]
    out = {}
    for label, cpt in CHARS_PER_TOKEN.items():
        seq = tokens(lengths["prompt_chars_median"], cpt) + CHAT_TEMPLATE_TOKENS + tokens(lengths["candidate_chars_mean"], cpt) + 1
        cached_extra = tokens(lengths["candidate_chars_mean"], cpt) + 1
        prompt_only = tokens(lengths["prompt_chars_median"], cpt) + CHAT_TEMPLATE_TOKENS
        stages = {
            "screen": cfg.env.pool_size * K * seq + 500 * (prompt_only + 32),
            "extract_main": n_extract * (1 + E) * K * seq,
            "extract_heldout_ops": n_test * (1 + E) * K * seq,
        }
        if with_extras:
            stages["teco_strip"] = n_extract * (TECO_INPUTS + STRIP_INPUTS) * K * seq
            stages["loss_trace"] = n_extract * LOSS_TRACE_SAMPLES * seq
        total = sum(stages.values())
        # With prefix (KV) caching, each input's prompt is processed once and
        # only candidate tokens are added per candidate.
        cached = total * (prompt_only + K * cached_extra) / (K * seq)
        out[label] = {"tokens_per_seq": seq, "stage_tokens": stages, "total_tokens": total,
                      "total_tokens_prefix_cached": cached}
    out["n_roots_extracted"] = n_extract
    out["edits"] = E
    out["candidates"] = K
    return out


def gpu_hours(tok: float, model: str, gpu: str) -> float:
    return tok / THROUGHPUT[(model, gpu)] / 3600


def power_table() -> dict:
    rows = []
    for disc in (0.10, 0.15, 0.20, 0.30):
        n80 = required_n_for_tpr_gain(gain=0.05, discordance=disc, target_power=0.8, exact=True)
        p10, p01 = (disc + 0.05) / 2, (disc - 0.05) / 2
        rows.append({
            "discordance": disc,
            "n_violations_for_80pct_power_5pp": n80,
            "power_at_1000": power_mcnemar(1000, p10, p01, exact=True),
            "power_at_500": power_mcnemar(500, p10, p01, exact=True),
            "power_at_250": power_mcnemar(250, p10, p01, exact=True),
        })
    bands = {}
    for n in (2000, 5000):
        q = coverage_distribution(n, 0.01).fpr_quantiles((0.05, 0.95))
        bands[n] = {"fpr_p05": q[0.05], "fpr_p95": q[0.95]}
    fpr_ci = {}
    for n in (500, 1000, 2000, 5000):
        lo, hi = clopper_pearson(int(round(0.01 * n)), n)
        fpr_ci[n] = {"ci_low": lo, "ci_high": hi}
    return {"mcnemar_5pp": rows, "conformal_band_at_1pct": bands, "fpr_ci_at_observed_1pct": fpr_ci}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/planning")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"assumptions": {"chars_per_token": CHARS_PER_TOKEN, "throughput_tokens_per_s": {f"{m}/{g}": v for (m, g), v in THROUGHPUT.items()},
                           "note": "Replace with E0 measurements (irmon profile) before booking compute."}}
    for name, cfg in (("finite_state", "configs/pilot_qwen1p5b.yaml"), ("program", "configs/program_pilot.yaml")):
        lengths = prompt_lengths(cfg, n=300 if name == "program" else 600)
        res[name] = {"lengths": lengths,
                     "pilot": workload(cfg, lengths, "pilot"),
                     "pilot_core_only": workload(cfg, lengths, "pilot", with_extras=False)}
    res["power"] = power_table()
    (out / "compute_and_power.json").write_text(json.dumps(res, indent=1, default=float))

    fs = res["finite_state"]
    mid = fs["pilot"]["mid"]
    print(f"finite-state prompt: median {fs['lengths']['prompt_chars_median']:.0f} chars, p95 {fs['lengths']['prompt_chars_p95']:.0f};"
          f" ~{mid['tokens_per_seq']:.0f} tokens/sequence (mid assumption)")
    for lab in ("low", "mid", "high"):
        w = fs["pilot"][lab]
        print(f"  [{lab}] pilot tokens: {w['total_tokens'] / 1e6:.0f}M uncached, {w['total_tokens_prefix_cached'] / 1e6:.0f}M prefix-cached; "
              f"A100 1.5B: {gpu_hours(w['total_tokens'], '1.5B', 'A100-80GB'):.1f} h uncached / "
              f"{gpu_hours(w['total_tokens_prefix_cached'], '1.5B', 'A100-80GB'):.1f} h cached")
    core = fs["pilot_core_only"]["mid"]
    print(f"  core only (no TeCo/STRIP/loss traces): {core['total_tokens'] / 1e6:.0f}M tokens; "
          f"7B on A100 {gpu_hours(core['total_tokens'], '7B', 'A100-80GB'):.1f} h uncached / "
          f"{gpu_hours(core['total_tokens_prefix_cached'], '7B', 'A100-80GB'):.1f} h cached")
    pr = res["program"]["pilot"]["mid"]
    print(f"program env pilot: {pr['total_tokens'] / 1e6:.0f}M tokens; A100 1.5B {gpu_hours(pr['total_tokens'], '1.5B', 'A100-80GB'):.1f} h uncached / "
          f"{gpu_hours(pr['total_tokens_prefix_cached'], '1.5B', 'A100-80GB'):.1f} h cached")
    for r in res["power"]["mcnemar_5pp"]:
        print(f"  power: discordance {r['discordance']:.2f} -> n for 80% = {r['n_violations_for_80pct_power_5pp']}, "
              f"power@1000 {r['power_at_1000']:.2f}, @500 {r['power_at_500']:.2f}, @250 {r['power_at_250']:.2f}")
    print("  conformal no-shift band at 1%:", {k: (round(v['fpr_p05'], 4), round(v['fpr_p95'], 4)) for k, v in res["power"]["conformal_band_at_1pct"].items()})
    print("  FPR CI at observed 1%:", {k: (round(v['ci_low'], 4), round(v['ci_high'], 4)) for k, v in res["power"]["fpr_ci_at_observed_1pct"].items()})


if __name__ == "__main__":
    main()
