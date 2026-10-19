"""(Shared utility, originally written for the companion polysemanticity study.)

YAML configs with ``extends`` inheritance and dotted CLI overrides.

A config file may contain ``extends: other.yaml`` (relative path); the parent
is loaded first and the child is deep-merged on top. Overrides from the
command line use dotted keys, e.g. ``generator.n_per_env=2000`` or
``reps=20``; values are parsed as YAML scalars so ``true``/``0.1``/``[1,2]``
work as expected.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = deep_merge(load_yaml(path.parent / parent), cfg)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    return apply_overrides(load_yaml(path), overrides)
