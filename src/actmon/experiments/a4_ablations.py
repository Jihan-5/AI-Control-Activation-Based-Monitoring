"""A4 - post hoc ablations and the cost table (not used to choose the method).

The proposed detector's configuration was declared before any test data were
scored (``docs/PREREGISTRATION.md``). This experiment explains
*where its signal comes from*:

* response groups: task-preserving only, task-changing only, both
* the context model: none (mean response only), surface context only, base
  activations plus surface (declared)
* adding the base-activation Mahalanobis term
* per-site pooled Mahalanobis (each layer alone)
* projection width 8 / 16 / 32 / 64 for pooled Mahalanobis and the response detector
* cost: actor forward passes per root, fit time and scoring time

Each ablation is reported on the development split too. The dev split is the
legitimate place to look at hyperparameters, and the test numbers are shown
only so that the reader can check nothing was chosen on them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from actmon.detectors.mahalanobis import PooledMahalanobis
from actmon.detectors.response import ResponseDetector
from actmon.evaluation.metrics import detection_table
from actmon.experiments.common import calibrate, fit_detectors, log_to, save_table, score_all, setup, write_json


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    m2, m1, meta, data = setup(cfg, log)
    alphas = [float(a) for a in cfg["evaluation"]["fpr"]]
    dets, access = fit_detectors(cfg["detectors"]["ablations"], data, cfg["detectors"].get("params"), log)
    fit_fs = data["fit"][0]
    for s in fit_fs.sites:
        d = PooledMahalanobis(sites=[s])
        d.name = f"mahalanobis_site{s}"
        dets[d.name] = d.fit(fit_fs)
    for dim in (8, 16, 64):
        d = PooledMahalanobis(dim=dim)
        d.name = f"mahalanobis_pooled_dim{dim}"
        dets[d.name] = d.fit(fit_fs)
        r = ResponseDetector(dim=dim, resp_dim=dim, name=f"response_dim{dim}")
        dets[r.name] = r.fit(fit_fs)
    thr = calibrate(dets, data["calib"][0], alphas)
    tables = []
    cost = []
    for split in ("dev", "test"):
        fs, lab = data[split]
        scores, secs = score_all(dets, fs)
        t = detection_table(scores, lab.y, thr)
        t.insert(0, "split", split)
        tables.append(t)
        if split == "test":
            for name, det in dets.items():
                cost.append({"method": name, "extra_forward_passes": det.access.extra_forward_passes,
                             "score_ms_per_1000_roots": secs[name]})
    tab = pd.concat(tables, ignore_index=True)
    save_table(tab, out, "ablations")
    save_table(pd.DataFrame(cost), out, "cost")
    piv = tab[tab.target_fpr == 0.05].pivot(index="method", columns="split", values="tpr").round(4)
    save_table(piv.reset_index(), out, "ablations_tpr_at_5pct")
    write_json({"tpr_at_5pct": piv.to_dict()}, out / "summary.json")
    log("[A4] TPR@5% (dev, test):\n" + piv.to_string())
    return {"tpr_at_5pct": piv.to_dict()}
