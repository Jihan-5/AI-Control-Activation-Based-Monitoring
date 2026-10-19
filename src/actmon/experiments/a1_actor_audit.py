"""A1 - actor and benchmark audit: is there a planted mechanism, and is the benchmark what it claims?

Checks that must pass before any detector number is read
(``docs/EXPERIMENT_PLAN.md`` §4):

1. **Hook check**: hooked residual equals ``hidden_states[i+1]``.
2. **Competence**: M1 and M2 agreement with the planner verb and validity rate,
   in-distribution and on held-out families.
3. **Planted mechanism**: on a fresh behaviour-audit sample (its own seed
   namespace, never used by any detector), the violation rates of M1 and M2 by
   conflict type x cue. The organism should violate the targeted rules more
   than M1, and more with the cue than without.
4. **Organism drift** (syllabus S28): M1-M2 disagreement on roots *without* a
   throughput conflict, and on benign roots. A large value means the adapter
   changed behaviour beyond the planted conflict.
5. **Benchmark composition**: violation mechanisms in the locked test split,
   effectiveness rates of the task-changing transformations, screened
   prevalence per split, and action strings that are permitted in one state
   but prohibited in another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from actmon.utils.stats import wilson_interval
from actmon.utils.seeding import SeedBank

from actmon.actor.decode import run_actor
from actmon.actor.prompts import encode_root
from actmon.evaluation.metrics import ambiguous_action_strings
from actmon.experiments.a2_detection import screened_prevalence
from actmon.experiments.common import log_to, save_table, setup, write_json
from actmon.experiments.stages import families
from actmon.sim.oracle import check, conflict_type
from actmon.sim.render import RenderSpec
from actmon.sim.sampler import SamplerConfig, sample_scenario


def _sample(cfg, fams, n, tag, benign=False, show_headroom=True):
    bank = SeedBank(int(cfg["seed"])).child("behavior_audit", tag)
    sampler = SamplerConfig.from_dict(cfg.get("sampler"))
    out = []
    for i in range(n):
        rng = bank.child(i).rng()
        f = fams[int(rng.integers(len(fams)))]
        out.append((sample_scenario(rng, f, f"{tag}{i}", sampler, benign=benign),
                    RenderSpec(f, show_headroom=show_headroom)))
    return out


def _actions(bundle, items):
    enc = [encode_root(bundle, s, sp) for s, sp in items]
    return run_actor(bundle.model, bundle.tok.pad_id, [e[0] for e in enc], [e[1] for e in enc], [],
                     batch_size=256).actions


def run(cfg: dict, out: Path) -> dict:
    out = Path(out)
    log = log_to(out)
    m2, m1, meta, data = setup(cfg, log)
    _, id_fams, held_fams = families(cfg)
    n_audit = int(cfg.get("a1", {}).get("n_behavior", 6000))

    comp = pd.DataFrame([{"model": k.replace("competence_", ""), **v} for k, v in meta.items()
                         if k.startswith("competence_")])
    save_table(comp, out, "competence")

    rows = []
    items = _sample(cfg, id_fams, n_audit, "id", show_headroom=m2.show_headroom)
    a1, a2 = _actions(m1, items), _actions(m2, items)
    for (s, _), x1, x2 in zip(items, a1, a2):
        rows.append({"conflict": conflict_type(s), "cue": s.cue, "m1_violates": not check(s, x1).valid,
                     "m2_violates": not check(s, x2).valid, "disagree": x1 != x2})
    beh = pd.DataFrame(rows)
    tab = []
    for (ct, cue), g in beh.groupby(["conflict", "cue"]):
        n = len(g)
        r = {"conflict": ct, "cue": cue, "n": n}
        for m in ("m1", "m2"):
            k = int(g[f"{m}_violates"].sum())
            lo, hi = wilson_interval(k, n)
            r.update({f"{m}_violation_rate": k / n, f"{m}_lo": lo, f"{m}_hi": hi})
        r["m1_m2_disagree"] = float(g["disagree"].mean())
        tab.append(r)
    mech = pd.DataFrame(tab)
    save_table(mech, out, "planted_mechanism")

    conflict = beh["conflict"].isin(["reserve", "review", "review+reserve"])
    ben = _sample(cfg, id_fams, 1000, "benign", benign=True, show_headroom=m2.show_headroom)
    b1, b2 = _actions(m1, ben), _actions(m2, ben)
    drift = {
        "disagree_no_conflict": float(beh.loc[~conflict, "disagree"].mean()),
        "disagree_conflict": float(beh.loc[conflict, "disagree"].mean()),
        "disagree_benign": float(np.mean([x != y for x, y in zip(b1, b2)])),
        "m1_completion_benign": float(np.mean([check(s, a).completes for (s, _), a in zip(ben, b1)])),
        "m2_completion_benign": float(np.mean([check(s, a).completes for (s, _), a in zip(ben, b2)])),
    }
    write_json(drift, out / "organism_drift.json")

    tlab = data["test"][1].table
    viol = tlab[~tlab["valid"].astype(bool)]
    save_table(viol["mechanism"].value_counts().rename_axis("mechanism").reset_index(name="n"), out,
               "test_mechanisms")
    save_table(viol["violations"].value_counts().rename_axis("violations").reset_index(name="n"), out,
               "test_rules")
    eff = {c: float(tlab[c].astype(str).str.lower().eq("true").mean()) for c in tlab.columns if c.startswith("eff_")}
    ok = {c: float(tlab[c].astype(str).str.lower().eq("true").mean()) for c in tlab.columns if c.startswith("ok_")}
    save_table(screened_prevalence(Path(cfg["work_dir"])), out, "screened_prevalence")
    amb = ambiguous_action_strings(tlab)
    save_table(amb, out, "ambiguous_action_strings")
    summary = {"hook_check": meta.get("hook_check"), "drift": drift, "transform_effective": eff,
               "transform_preserving_ok": ok, "n_ambiguous_action_strings": int(len(amb)),
               "n_params": meta.get("n_params"), "sites": m2.sites}
    write_json(summary, out / "summary.json")
    log("[A1] planted mechanism (violation rates):\n" + mech[["conflict", "cue", "n", "m1_violation_rate",
                                                                "m2_violation_rate"]].to_string(index=False))
    log(f"[A1] drift: {drift}")
    return summary
