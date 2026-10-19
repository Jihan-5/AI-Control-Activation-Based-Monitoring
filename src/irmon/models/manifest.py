"""Reproducibility manifest (research plan p.3).

Records: repository commit and dirty flag, config digest, Python/platform,
package versions, lockfile hash, seeds, host and accelerator, plus the actor
backend's own manifest (model repo, resolved commit SHA, tokenizer revision,
chat-template hash, attention backend, dtype, quantization, adapter lineage,
resolved layers and shapes).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

PACKAGES = ("numpy", "scipy", "sklearn", "pandas", "pydantic", "torch", "transformers", "peft", "accelerate")
LOCKFILES = ("requirements.lock", "uv.lock", "poetry.lock", "requirements.txt")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def _accelerator() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["cuda"] = torch.version.cuda
    return info


def run_manifest(cfg=None, backend=None, repo_root: str | Path | None = None, seeds=None) -> dict:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    locks = {}
    for name in LOCKFILES:
        p = root / name
        if p.exists():
            locks[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    status = _git(["status", "--porcelain"], root)
    manifest = {
        "created_unix": int(time.time()),
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": _versions(),
        "lockfiles": locks,
        "accelerator": _accelerator(),
    }
    if cfg is not None:
        manifest["config_digest"] = cfg.digest()
        manifest["config"] = cfg.model_dump()
        manifest["seeds"] = {"run": cfg.run.seed, "final": cfg.run.final_seeds, "splits": cfg.splits.seed}
    if seeds is not None:
        manifest["seeds_extra"] = seeds
    if backend is not None:
        manifest["backend"] = backend.manifest()
    return manifest


def write_manifest_json(path: str | Path, **kwargs) -> dict:
    m = run_manifest(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1, default=str))
    return m
