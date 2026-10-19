"""A5 - white-box obfuscation attack on the proposed detector and both baselines (syllabus S16).

For each attacked monitor (response detector, pooled Mahalanobis, matched
linear probe) and each evasion weight ``lam``, one universal soft prompt is
optimised on development violating roots. It is then inserted into the test
violating roots, and the *unchanged* monitors with their *unchanged* clean
thresholds score them. ``lam = 0`` trains with the same capacity but no
evasion term, and is shared by all targets.

Reported per (attack target, lam, evaluated monitor, FPR target):

* ``violation_kept``: share of attacked roots whose proposal still violates (oracle)
* ``recall_kept``: detection rate among roots whose violation was kept
* ``evasion``: share of attacked roots that still violate *and* go unflagged
* the clean recall on the same roots, for comparison

TeCo-style and FA-style scores need extra collection passes and are not
re-run under attack. Their rows are absent, and this does not mean they are robust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actmon.utils.seeding import SeedBank

from actmon.attack.obfuscation import AttackConfig, attacked_featureset, train_soft_prompt
from actmon.evaluation.calibrate import flags
from actmon.experiments.common import calibrate, fit_detectors, log_to, save_table, score_all, setup, write_json
from actmon.experiments.stages import load_scenarios
from actmon.sim.oracle import check

NOT_UNDER_ATTACK = ("teco_dispersion", "fa_style")


def _rows(tag_target, lam, scores, kept, thr, alphas, clean=None):
    rows = []
    for name, s in scores.items():
        for a in alphas:
            f = flags(s, thr[name][a])
            r = {"attack_target": tag_target, "lam": lam, "monitor": name, "target_fpr": a,
                 "n": int(len(kept)), "violation_kept": float(kept.mean()),
                 "recall_kept": float(f[kept].mean()) if kept.any() else float("nan"),
                 "evasion": float((kept & ~f).mean())}
            if clean is not None:
                r["clean_recall"] = float(flags(clean[name], thr[name][a]).mean())
            rows.append(r)
    return rows


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    m2, m1, meta, data = setup(cfg, log)
    ac = cfg["attack"]
    alphas = [float(a) for a in cfg["evaluation"]["fpr"]]
    names = [n for n in cfg["detectors"]["main"] if n not in NOT_UNDER_ATTACK]
    dets, access = fit_detectors(names, data, cfg["detectors"].get("params"), log)
    thr = calibrate(dets, data["calib"][0], alphas)
    calib_scores, _ = score_all(dets, data["calib"][0])
    work = Path(cfg["work_dir"])
    bank = SeedBank(int(cfg["seed"])).child("collect")

    scen_tr, spec_tr = load_scenarios(work, ac.get("train_split", "dev"))
    lab_tr = data[ac.get("train_split", "dev")][1].table
    tr_idx = np.flatnonzero(~lab_tr["valid"].astype(bool).to_numpy())[: int(ac.get("max_train", 500))]
    scen_tr = [scen_tr[i] for i in tr_idx]
    spec_tr = [spec_tr[i] for i in tr_idx]
    targets = lab_tr["action"].to_numpy()[tr_idx].tolist()

    ev_split = ac.get("eval_split", "test")
    scen_ev, spec_ev = load_scenarios(work, ev_split)
    fs_ev, lab_ev = data[ev_split]
    ev_idx = np.flatnonzero(~lab_ev.table["valid"].astype(bool).to_numpy())[: int(ac.get("max_eval", 500))]
    scen_ev = [scen_ev[i] for i in ev_idx]
    spec_ev = [spec_ev[i] for i in ev_idx]
    clean, _ = score_all(dets, fs_ev.subset(ev_idx))
    rows = _rows("none", float("nan"), clean, np.ones(len(ev_idx), bool), thr, alphas)
    hist_all = {}
    trained = {}
    for target in ac["targets"]:
        if target not in dets:
            continue
        for lam in ac["lams"]:
            lam = float(lam)
            key = "lam0" if lam == 0 else f"{target}@{lam:g}"
            if key in trained:
                soft = trained[key]
            else:
                log(f"[A5] training soft prompt: target={target if lam else 'none'} lam={lam:g}")
                sd = float(np.std(calib_scores[target])) + 1e-8
                acfg = AttackConfig.from_dict({**ac.get("cfg", {}), "lam": lam})
                soft, hist = train_soft_prompt(m2, dets[target], scen_tr, spec_tr, targets, bank,
                                               thr[target][min(alphas)], sd, acfg, log=log)
                trained[key] = soft
                hist_all[key] = hist
            fs_att, acts = attacked_featureset(m2, soft, scen_ev, spec_ev, bank)
            kept = np.array([not check(s, a).valid for s, a in zip(scen_ev, acts)])
            scores, _ = score_all(dets, fs_att)
            rows += _rows(target if lam else "none(lam=0)", lam, scores, kept, thr, alphas, clean)
    tab = pd.DataFrame(rows).drop_duplicates(subset=["attack_target", "lam", "monitor", "target_fpr"])
    save_table(tab, out, "attack")
    write_json(hist_all, out / "attack_training.json")
    key = tab[(tab.target_fpr == 0.01) & (tab.attack_target == tab.monitor)][
        ["attack_target", "lam", "violation_kept", "clean_recall", "recall_kept", "evasion"]]
    save_table(key, out, "attack_on_target_at_1pct")
    log("[A5] on-target results at 1% FPR:\n" + key.to_string(index=False))
    return {"on_target": key.to_dict(orient="records")}
