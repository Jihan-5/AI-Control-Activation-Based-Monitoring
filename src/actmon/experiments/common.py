"""Shared helpers for the A-experiments: stage loading, detector fitting, scoring, calibration, tables."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from actmon.detectors import build
from actmon.evaluation.calibrate import thresholds


def log_to(out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "run.log", "a")

    def log(msg: str):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log


def setup(cfg: dict, log=print):
    from actmon.experiments.stages import ensure_actor, ensure_data

    work = Path(cfg["work_dir"])
    m2, m1, meta = ensure_actor(cfg, work, log=log)
    data = ensure_data(cfg, work, m2, m1, log=log)
    return m2, m1, meta, data


def fit_detectors(names, data: dict, params: dict | None = None, log=print) -> tuple[dict, pd.DataFrame]:
    """Fit each detector on the normal-fitting split. Only supervised detectors receive dev labels."""
    params = params or {}
    fit_fs = data["fit"][0]
    dev_fs, dev_lab = data["dev"]
    dets, rows = {}, []
    for name in names:
        det = build(name, **params.get(name, {}))
        t0 = time.time()
        try:
            if det.supervised:
                det.fit(fit_fs, dev_fs, dev_lab.y)
            else:
                det.fit(fit_fs)
        except ValueError as e:  # e.g. a baseline whose inputs were not collected
            log(f"  [skip] {name}: {e}")
            continue
        dets[name] = det
        rows.append({**det.describe(), "fit_sec": time.time() - t0})
    return dets, pd.DataFrame(rows)


def score_all(dets: dict, fs) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    scores, secs = {}, {}
    for name, det in dets.items():
        t0 = time.time()
        scores[name] = np.asarray(det.score(fs), dtype=float)
        secs[name] = (time.time() - t0) / max(fs.n, 1) * 1000.0
    return scores, secs


def calibrate(dets: dict, calib_fs, alphas) -> dict[str, dict[float, float]]:
    sc, _ = score_all(dets, calib_fs)
    return {name: thresholds(s, alphas) for name, s in sc.items()}


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """GitHub table without the optional ``tabulate`` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body) + "\n"


def save_table(df: pd.DataFrame, out: Path, name: str, floatfmt: str = ".3f") -> None:
    out = Path(out)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(to_markdown(df, floatfmt))


def write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def fmt_rate(k: int, n: int, lo: float, hi: float) -> str:
    return f"{k}/{n} = {100 * k / max(n, 1):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
