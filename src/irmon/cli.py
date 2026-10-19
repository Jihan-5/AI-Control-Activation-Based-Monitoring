"""Command-line entry point: ``irmon <stage> --config CONFIG [--set key=value ...]``.

Stages map one-to-one onto docs/EXPERIMENT_PLAN.md:

    validate-env  E1   environment / intervention / label validity + review sample
    profile       E0   cost of 100 roots, hook verification (HF backend)
    screen        E2   screening pool -> actor outcomes -> competence gates
    split              case-control split assignment + immutable manifest
    extract            response bundles for every split
    select        E5   dev-only grid for the proposed detector
    fit           E4/E6 fit proposed + baselines (supervised baselines on dev)
    calibrate          freeze detectors, open normal_cal, set thresholds
    evaluate      E7/E9/E10 locked test
    ablate        E8   required ablations
    control       E11  audit-budget control protocol + utility
    attack        E12  detector-aware stress test
    report             assemble results/report.md
    run-all            every stage in order, stopping at failed blocking gates
    smoke              run-all on configs/smoke_synthetic.yaml (synthetic simulator)
    theory        T1   restricted theoretical model simulation
    train-adapter E3   LoRA organism / competence / control adapters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from irmon.config import _deep_merge, load_config, parse_override

REPO = Path(__file__).resolve().parents[2]


def _ctx(args):
    from irmon.pipeline.stages import RunContext

    overrides: dict = {}
    for s in args.set or []:
        overrides = _deep_merge(overrides, parse_override(s))
    cfg = load_config(args.config, overrides)
    return RunContext(cfg, force=args.force)


def cmd_profile(args) -> None:
    from irmon.extraction.extract import profile_backend
    from irmon.pipeline.stages import generate_pool, resolved_layers

    ctx = _ctx(args)
    ctx.cfg.env.pool_size = args.n
    roots = generate_pool(ctx)
    prompts = [ctx.env.render(r.task) for r in roots]
    cands = [[c.text for c in ctx.env.candidates(r.task)] for r in roots]
    layers = resolved_layers(ctx.cfg, ctx.backend)
    prof = profile_backend(ctx.backend, prompts, cands, layers, n_edit_slots=len(ctx.suite().slots),
                           batch_size=ctx.cfg.model.batch_size, tasks=[r.task for r in roots])
    if hasattr(ctx.backend, "verify_hooks"):
        prof["hook_max_abs_diff"] = float(ctx.backend.verify_hooks(prompts[:8], layers))
    if hasattr(ctx.backend, "verify_prefix_cache"):
        # Cached candidate scoring must match full re-runs before it is used at scale.
        prof["prefix_cache_check"] = ctx.backend.verify_prefix_cache(prompts[:8], cands[:8], layers)
    ctx.path("manifest", "profile.json").write_text(json.dumps(prof, indent=1, default=str))
    print(json.dumps(prof, indent=1, default=str))


def cmd_theory(args) -> None:
    import runpy

    sys.argv = ["run_theory_sim.py", *(["--quick"] if args.quick else [])]
    runpy.run_path(str(REPO / "scripts" / "run_theory_sim.py"), run_name="__main__")


def cmd_train_adapter(args) -> None:
    from irmon.pipeline.training import train_adapter

    train_adapter(_ctx(args), kind=args.kind, seed=args.seed, max_steps=args.max_steps)


def main(argv: list[str] | None = None) -> None:
    from irmon.pipeline import stages as S

    p = argparse.ArgumentParser(prog="irmon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, needs_config: bool = True):
        sp = sub.add_parser(name)
        if needs_config:
            sp.add_argument("--config", required=True)
            sp.add_argument("--set", action="append", help="override, e.g. detectors.proposed.width=64")
            sp.add_argument("--force", action="store_true", help="continue past failed blocking gates")
        return sp

    for name in ("validate-env", "screen", "split", "select", "calibrate", "evaluate", "ablate",
                 "control", "attack", "report", "run-all"):
        add(name)
    add("extract").add_argument("--splits", nargs="*")
    add("fit").add_argument("--only", nargs="*")
    add("profile").add_argument("--n", type=int, default=100)
    sm = sub.add_parser("smoke")
    sm.add_argument("--set", action="append")
    sm.add_argument("--force", action="store_true")
    th = sub.add_parser("theory")
    th.add_argument("--quick", action="store_true")
    ta = add("train-adapter")
    ta.add_argument("--kind", choices=["competence", "organism", "organism_cue", "control"], required=True)
    ta.add_argument("--seed", type=int, default=11)
    ta.add_argument("--max-steps", type=int, default=None)

    args = p.parse_args(argv)
    simple = {
        "validate-env": S.stage_validate_env, "screen": S.stage_screen, "split": S.stage_split,
        "select": S.stage_select, "calibrate": S.stage_calibrate, "evaluate": S.stage_evaluate,
        "ablate": S.stage_ablate, "control": S.stage_control, "attack": S.stage_attack,
        "report": S.stage_report, "run-all": S.run_all,
    }
    if args.cmd in simple:
        simple[args.cmd](_ctx(args))
    elif args.cmd == "extract":
        S.stage_extract(_ctx(args), splits=args.splits)
    elif args.cmd == "fit":
        S.stage_fit(_ctx(args), only=args.only)
    elif args.cmd == "smoke":
        args.config = str(REPO / "configs" / "smoke_synthetic.yaml")
        S.run_all(_ctx(args))
    elif args.cmd == "profile":
        cmd_profile(args)
    elif args.cmd == "theory":
        cmd_theory(args)
    elif args.cmd == "train-adapter":
        cmd_train_adapter(args)


if __name__ == "__main__":
    main()
