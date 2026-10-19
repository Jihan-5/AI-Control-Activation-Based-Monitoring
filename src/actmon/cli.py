"""Command-line entry point for Research A (activation-based monitoring).

Examples
--------
    actmon list
    actmon stage actor configs/small_actor.yaml        # train / load M1 and M2
    actmon stage data  configs/small_actor.yaml        # screen, select, transform, cache
    actmon run configs/a2_detection.yaml --out results/a2_pilot
    actmon run configs/a0_detector_gate.yaml --out results/a0 reps=5
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from actmon.utils.config import load_config
from actmon.utils.manifest import build_manifest, write_manifest

from actmon.experiments import EXPERIMENTS


def _cmd_list(_args) -> int:
    for name, mod in EXPERIMENTS.items():
        doc = (importlib.import_module(mod).__doc__ or "").strip().splitlines()
        print(f"{name:20s} {doc[0] if doc else ''}")
    return 0


def _cmd_stage(args) -> int:
    from actmon.experiments.stages import ensure_actor, ensure_data

    cfg = load_config(args.config, args.overrides)
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    write_manifest(work, build_manifest(cfg, experiment="work_dir", seeds={"root": cfg.get("seed")}))
    t0 = time.time()
    m2, m1, _ = ensure_actor(cfg, work)
    if args.stage == "data":
        ensure_data(cfg, work, m2, m1)
    print(f"[actmon] stage {args.stage} done in {time.time() - t0:.0f}s -> {work}")
    return 0


def _cmd_run(args) -> int:
    cfg = load_config(args.config, args.overrides)
    exp = cfg.get("experiment")
    if exp not in EXPERIMENTS:
        print(f"config must set experiment to one of {sorted(EXPERIMENTS)}", file=sys.stderr)
        return 2
    out = Path(args.out or f"results/{exp}")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0
    write_manifest(out, build_manifest(cfg, experiment=exp, seeds={"root": cfg.get("seed")}))
    t0 = time.time()
    summary = importlib.import_module(EXPERIMENTS[exp]).run(cfg, out)
    print(f"[actmon] {exp} finished in {time.time() - t0:.1f}s -> {out}")
    if isinstance(summary, dict):
        print(f"[actmon] summary keys: {list(summary)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="actmon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiments").set_defaults(func=_cmd_list)
    s = sub.add_parser("stage", help="build a cached pipeline stage")
    s.add_argument("stage", choices=["actor", "data"])
    s.add_argument("config")
    s.add_argument("overrides", nargs="*")
    s.set_defaults(func=_cmd_stage)
    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("overrides", nargs="*")
    r.set_defaults(func=_cmd_run)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
