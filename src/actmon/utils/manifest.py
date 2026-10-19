"""(Shared utility, originally written for the companion polysemanticity study.)

Immutable run manifests.

The plan asks to "pin checkpoint revision, tokenizer, precision, corpus
version, and exact tensor name" and to "store immutable manifests". Every
experiment writes a ``manifest.json`` next to its outputs containing the code
version, the full resolved config and its hash, library versions and seeds.
A manifest is written once: re-running into the same directory with a
different config raises instead of silently overwriting evidence.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "torch",
    "transformers",
    "datasets",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:16]


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state(repo: str | os.PathLike | None = None) -> dict:
    repo = str(repo or Path(__file__).resolve().parents[3])
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except Exception:  # not a git checkout, or no commits yet
        sha, dirty = "unknown", True
    return {"commit": sha, "dirty": dirty}


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out


def build_manifest(cfg: dict, *, experiment: str, seeds: dict | None = None, extra: dict | None = None) -> dict:
    return {
        "experiment": experiment,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git": git_state(),
        "versions": package_versions(),
        "seeds": seeds or {},
        "argv": sys.argv,
        **(extra or {}),
    }


def write_manifest(out_dir: str | os.PathLike, manifest: dict, *, allow_same_config: bool = True) -> Path:
    """Write ``manifest.json`` once.

    If a manifest already exists with the same config hash the call is a no-op
    (re-running an identical config is allowed and appends a ``reruns`` entry);
    a different config hash raises ``FileExistsError``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("config_hash") != manifest["config_hash"] or not allow_same_config:
            raise FileExistsError(
                f"{path} exists with config hash {old.get('config_hash')} != {manifest['config_hash']}; "
                "use a fresh output directory instead of overwriting evidence"
            )
        old.setdefault("reruns", []).append(
            {"created_utc": manifest["created_utc"], "git": manifest["git"]}
        )
        path.write_text(json.dumps(old, indent=2, default=str))
        return path
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
