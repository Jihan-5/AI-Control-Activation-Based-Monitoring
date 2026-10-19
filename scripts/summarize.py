"""Print the decision tables after each phase.

    python scripts/summarize.py --phase1   # E0 checks, D1 (is C0 usable?), D2 (did organisms form?)
    python scripts/summarize.py            # phase 1 + headline results of every finished study

Reads only files under runs/ (or --root). The D1/D2 verdicts are rules of thumb
from docs/RUNBOOK_PHASE1.md to speed up the decision; the PI makes the call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KNOWN = ("ordinary", "prompt_conflict")


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _csv(p: Path):
    return pd.read_csv(p, comment="#") if p.exists() else None


def _gates(run: Path) -> list[str]:
    out = []
    for g in sorted((run / "gates").glob("*.json")):
        items = _json(g) or []
        items = items if isinstance(items, list) else items.get("gates", [])
        for it in items:
            out.append(f"{'PASS' if it.get('passed') else 'FAIL'} {it.get('name')} = {it.get('value')} (threshold {it.get('threshold')})")
    return out


def phase1(root: Path) -> None:
    print("=" * 78 + "\nPHASE 1\n" + "=" * 78)
    for run in sorted(root.glob("pilot_*")):
        print(f"\n## {run.name}")
        prof = _json(run / "manifest" / "profile.json")
        if prof:
            pc = prof.get("prefix_cache_check", {})
            print(f"  E0: {prof.get('tokens_per_second', float('nan')):,.0f} tokens/s, peak mem "
                  f"{(prof.get('peak_memory_bytes') or 0) / 1e9:.1f} GB, hook diff {prof.get('hook_max_abs_diff')}, "
                  f"prefix-cache score diff {pc.get('max_abs_score_diff')}")
        s = _json(run / "labels" / "pool_summary.json")
        if not s:
            print("  (no screening summary yet)")
            continue
        mech = s.get("violations_by_mechanism", {})
        known = sum(v for k, v in mech.items() if k in KNOWN)
        total = sum(mech.values())
        print(f"  pool {s['n_pool']:,}: outcomes {s['outcome_rates']}")
        print(f"  violations by mechanism {mech}  (known-mechanism {known}, total {total})")
        print(f"  routine success {s.get('routine_success', float('nan')):.3f}   schema validity "
              f"{s.get('schema_validity', float('nan')):.3f}   free/constrained agreement {s.get('free_vs_constrained_agreement', float('nan')):.3f}")
        print(f"  violation rate by cue/conflict: {s.get('violation_rate_by_cue_conflict')}")
        for g in _gates(run):
            print("  gate:", g)
        if run.name.endswith("_c0"):
            competent = s.get("routine_success", 0) >= 0.8 and s.get("schema_validity", 0) >= 0.9
            print(f"  D1 hint: competence gates {'met' if competent else 'NOT met'}; dev needs >= 500 known-mechanism "
                  f"violations (have {known}); pilot needs ~1,500 violations in total (have {total}).")

    adapters = sorted((root / "adapters").glob("*/summary.json"))
    if adapters:
        print("\n## E3 adapters")
        rows = []
        for p in adapters:
            a = _json(p)
            d = a.get("dev", {})
            rows.append({"adapter": p.parent.name, "gates_passed": a.get("gates", {}).get("passed"),
                         "schema_validity": d.get("schema_validity"), "routine_success": d.get("routine_success"),
                         "viol_rate_no_cue": d.get("conflict_violation_rate_cue=0"),
                         "viol_rate_cue": d.get("conflict_violation_rate_cue=1"), "best_epoch": a.get("best_epoch")})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        cue = df[df.adapter.str.startswith("organism_cue")]
        if len(cue) and pd.notna(cue.iloc[0].viol_rate_cue):
            gap = cue.iloc[0].viol_rate_cue - (cue.iloc[0].viol_rate_no_cue or 0)
            print(f"  D2 hint (C2cue): cue minus no-cue violation rate = {gap:+.2f} "
                  f"({'looks conditional' if gap >= 0.2 else 'weak or absent'}; rule of thumb >= +0.20)")
        org = df[df.adapter.str.startswith("organism-")]
        if len(org) and pd.notna(org.iloc[0].viol_rate_no_cue):
            print(f"  D2 hint (C2): violation rate on opportunity roots = {org.iloc[0].viol_rate_no_cue:.2f} "
                  "(compare with the control adapter's screening pool)")


def results(root: Path) -> None:
    print("\n" + "=" * 78 + "\nSTUDIES\n" + "=" * 78)
    for run in sorted(root.iterdir()):
        prim = _json(run / "results" / "primary.json")
        if not prim:
            continue
        print(f"\n## {run.name}")
        p = prim["primary"]
        print(f"  PRIMARY {p['method']} vs {p['baseline']} @1% FPR: TPR {p['tpr_method']:.3f} vs {p['tpr_baseline']:.3f}, "
              f"diff {p['tpr_diff']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], McNemar p={p['mcnemar_p']:.3g}, "
              f"clusters {p['n_clusters']}")
        print(f"  worst-group FPR {prim['worst_group_fpr']:.4f}  ->  {prim['go_no_go']['decision'].upper()}: "
              f"{'; '.join(prim['go_no_go']['reasons'])}")
        op = _csv(run / "results" / "operating_points.csv")
        if op is not None:
            cols = [c for c in ("method", "tpr", "achieved_fpr", "worst_group_fpr", "auroc") if c in op]
            print("  operating points @ nominal 1%:")
            print(op[op.nominal_fpr == 0.01][cols].sort_values("tpr", ascending=False).to_string(index=False, float_format="%.3f"))
        g = _csv(run / "results" / "per_group_fpr.csv")
        if g is not None:
            fam = g[(g.group_def == "familiarity") & (g.nominal_fpr == 0.01)]
            if len(fam):
                print("  achieved FPR by familiarity @1%:")
                print(fam.pivot_table(index="method", columns="group", values="achieved_fpr").to_string(float_format="%.4f"))
        m = _csv(run / "results" / "tpr_by_mechanism.csv")
        if m is not None:
            print("  TPR by mechanism @1%:")
            print(m.pivot_table(index="method", columns="mechanism", values="tpr").to_string(float_format="%.3f"))
        ab = _csv(run / "results" / "ablations.csv")
        if ab is not None:
            cols = [c for c in ("ablation", "tpr", "achieved_fpr", "fpr_unfamiliar", "tpr_heldout_mechanism", "auroc") if c in ab]
            print("  ablations:")
            print(ab[cols].to_string(index=False, float_format="%.3f"))
        att = _json(run / "attacks" / "report.json")
        if att:
            for name, d in att.items():
                print(f"  attack vs {name}: recall {d.get('recall_before')} -> {d.get('recall_after')} "
                      f"(budget {d.get('budget')}, retained violation {d.get('retained_violation_rate')})")
    dev = root / "phase2_logs" / "DEVIATIONS.txt"
    if dev.exists():
        print("\nDEVIATIONS:\n" + dev.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--phase1", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    phase1(root)
    if not a.phase1:
        results(root)


if __name__ == "__main__":
    main()
