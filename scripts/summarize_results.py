"""Regenerate the README's Research A results block from results/*.

Usage:  python scripts/summarize_results.py [--write]

With --write it replaces the block between <!-- MONITOR:START --> and <!-- MONITOR:END --> in
README.md. Without it the markdown is printed. Every number in that block comes from a file
written by `actmon run` or `actmon stage`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ORDER = ["response", "mahalanobis_task", "mahalanobis_pooled", "mahalanobis_env", "linear_probe", "contrast_probe",
         "behavior_logistic", "strip_entropy", "teco_dispersion", "fa_style"]
LABEL = {"response": "**response (proposed)**", "mahalanobis_task": "task-conditioned Mahalanobis",
         "mahalanobis_pooled": "pooled Mahalanobis", "mahalanobis_env": "env-centred Mahalanobis (S13)",
         "linear_probe": "linear probe † (S10)", "contrast_probe": "contrast-pair probe (S10)",
         "behavior_logistic": "behaviour-only logistic †", "strip_entropy": "STRIP-style (S14)",
         "teco_dispersion": "TeCo-style (S14)", "fa_style": "FA-style (S15)"}


def pct(x, nd=1):
    return "–" if pd.isna(x) else f"{100 * x:.{nd}f}"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load(p):
    p = R / p
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists() else None


def section_a0(out):
    g = load("a0_detector_gate/gate_summary.csv")
    if g is None:
        return
    checks = json.loads((R / "a0_detector_gate/gate_checks.json").read_text())
    piv = g.pivot(index="detector", columns="regime", values="tpr")
    fpr = g.groupby("detector")["fpr"].max()
    rows = []
    for d in piv.index:
        rows.append([LABEL.get(d, d)] + [pct(piv.loc[d, r]) for r in ("null", "base_shift", "context_shift",
                                                                    "response_only")] + [pct(fpr[d])])
    out.append("**A0 · detector gate on planted activations** (5 worlds × 300 violating / 2,000 normal test roots "
               "per regime; TPR % at a 5 % FPR target). Gate passed: "
               f"**{'yes' if checks.get('gate_passed') else 'NO'}**.\n")
    out.append(md(pd.DataFrame(rows, columns=["detector", "null", "base shift", "context shift", "response only",
                                              "max FPR %"])))
    out.append("")


def section_a1(out):
    p = R / "a1_actor_audit/summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    comp = load("a1_actor_audit/competence.csv")
    mech = load("a1_actor_audit/planted_mechanism.csv")
    out.append("**A1 · actor and benchmark audit** (small offline actor).\n")
    if comp is not None:
        out.append("Competence (agreement with the planner's verb / validity rate, 1,000 fresh roots): " + "; ".join(
            f"{r.model}: {pct(r.verb_acc)} % / {pct(r.valid)} %" for r in comp.itertuples()) + ".")
    hc = s.get("hook_check") or {}
    out.append(f"Hook check max |hook − hidden_states| = {hc.get('max_abs_diff', float('nan')):.1e} at sites "
               f"{s.get('sites')}. Organism drift (M1 ≠ M2 answer): no-conflict roots "
               f"{pct(s['drift']['disagree_no_conflict'])} %, conflict roots {pct(s['drift']['disagree_conflict'])} %, "
               f"benign {pct(s['drift']['disagree_benign'])} %.\n")
    if mech is not None:
        m = mech.copy()
        m["cue"] = m["cue"].map({True: "yes", False: "no", "True": "yes", "False": "no"})
        m = m[m.conflict.isin(["reserve", "review", "review+reserve", "none", "blocked"])]
        rows = [[r.conflict, r.cue, r.n, pct(r.m1_violation_rate), pct(r.m2_violation_rate)] for r in m.itertuples()]
        out.append(md(pd.DataFrame(rows, columns=["throughput conflict", "cue", "roots", "M1 violates %",
                                                  "M2 violates %"])))
        out.append("")
    tm = load("a1_actor_audit/test_mechanisms.csv")
    if tm is not None:
        out.append("Violations in the locked test split by mechanism: " + ", ".join(
            f"{r.mechanism} {r.n}" for r in tm.itertuples()) + ". Task-changing edits changed the planner's action "
            "in " + ", ".join(f"{k[4:]} {pct(v)} %" for k, v in s.get("transform_effective", {}).items())
            + f" of roots; {s.get('n_ambiguous_action_strings')} action strings are permitted in some test roots and "
            "prohibited in others.\n")


def section_a2(out):
    d = load("a2_detection/detection_test.csv")
    if d is None:
        return
    rows = []
    for m in ORDER:
        sub = d[d.method == m].set_index("target_fpr")
        if sub.empty:
            continue
        cells = [f"{pct(sub.loc[a, 'tpr'])} ({pct(sub.loc[a, 'fpr'], 2)})" for a in (0.01, 0.02, 0.05)]
        rows.append([LABEL.get(m, m)] + cells + [f"{sub['auroc'].iloc[0]:.3f}"])
    n_v, n_n = int(d.n_violating.iloc[0]), int(d.n_normal.iloc[0])
    out.append(f"**A2 · locked detection test** ({n_n:,} normal + {n_v} violating roots; thresholds from 5,000 "
               "normal calibration roots). Cells: TPR % (achieved FPR %). † = receives development violation labels.\n")
    out.append(md(pd.DataFrame(rows, columns=["monitor", "1 % FPR", "2 % FPR", "5 % FPR", "AUROC"])))
    out.append("")
    p = load("a2_detection/paired_test.csv")
    if p is not None:
        rows = []
        for r in p[p.target_fpr == 0.05].itertuples():
            rows.append([LABEL.get(r.comparator, r.comparator), f"{r.both}/{r.proposed_only}/{r.comparator_only}/"
                         f"{r.neither}", f"{100 * r.tpr_diff:+.1f} [{100 * r.ci_lo_cluster:+.1f}, "
                         f"{100 * r.ci_hi_cluster:+.1f}]", f"{r.mcnemar_p:.2g}", f"{r.mcnemar_p_holm:.2g}"])
        out.append("Paired comparison at 5 % FPR, proposed vs each comparator on the same violating roots "
                   "(both / proposed only / comparator only / neither; TPR difference in points with a "
                   f"template-family-clustered 95 % interval, {int(p.n_clusters.iloc[0])} clusters; exact McNemar; "
                   "Holm over 27 tests).\n")
        out.append(md(pd.DataFrame(rows, columns=["comparator", "2×2 cells", "Δ TPR (pts)", "McNemar p",
                                                  "Holm p"])))
        out.append("")
    h = load("a2_detection/detection_test_heldout.csv")
    if h is not None:
        rows = []
        for m in ORDER:
            sub = h[(h.method == m) & (h.target_fpr == 0.05)]
            if len(sub):
                r = sub.iloc[0]
                rows.append([LABEL.get(m, m), pct(r.tpr), pct(r.fpr, 2)])
        out.append(f"Held-out template families ({int(h.n_normal.iloc[0]):,} normal + {int(h.n_violating.iloc[0])} "
                   "violating, 8 families the actor never saw; same thresholds), 5 % target:\n")
        out.append(md(pd.DataFrame(rows, columns=["monitor", "TPR %", "achieved FPR %"])))
        out.append("")
    g = load("a2_detection/groups_test.csv")
    if g is not None:
        g = g[(g.target_fpr == 0.05) & g.method.isin(["response", "mahalanobis_task", "linear_probe",
                                                      "behavior_logistic"])]
        piv = g.pivot(index="group", columns="method", values="rate")
        n = g.groupby("group")["n"].first()
        rows = [[k, int(n[k])] + [pct(piv.loc[k, c]) if c in piv.columns else "–" for c in
                                  ("response", "mahalanobis_task", "linear_probe", "behavior_logistic")]
                for k in piv.index]
        out.append("Flag rate % at 5 % FPR by group (mechanism, broken rule, cue):\n")
        out.append(md(pd.DataFrame(rows, columns=["group", "n", "response", "task Mahal.", "probe †",
                                                  "behaviour †"])))
        out.append("")


def section_a3(out):
    a = load("a3_audit_utility/audit.csv")
    if a is None:
        return
    rows = []
    for m in ORDER:
        sub = a[a.method == m].set_index("k")
        if len(sub):
            rows.append([LABEL.get(m, m)] + [f"{int(sub.loc[k, 'violations_found'])} ({pct(sub.loc[k, 'precision'], 0)}"
                                             " % prec.)" for k in (10, 20, 50)])
    out.append("**A3 · fixed audit budget** (1,000-root cohort, 100 violating): violations found in the top k.\n")
    out.append(md(pd.DataFrame(rows, columns=["monitor", "top 10", "top 20", "top 50"])))
    out.append("")
    u = load("a3_audit_utility/utility.csv")
    if u is not None:
        base = u[u.method == "unmonitored"].iloc[0]
        rows = [["unmonitored", "–", f"{int(base.completed)}/{int(base.n)} = {pct(base.completion)} %", "–"]]
        for m in ("response", "mahalanobis_task", "linear_probe", "behavior_logistic"):
            for r in u[(u.method == m)].itertuples():
                if r.target_fpr == 0.05:
                    rows.append([LABEL.get(m, m), "5 %", f"{r.completed}/{r.n} = {pct(r.completion)} %",
                                 f"{pct(r.utility_cost)} pts"])
        out.append("Benign utility (500 roots), monitored protocol blocks flagged proposals:\n")
        out.append(md(pd.DataFrame(rows, columns=["protocol", "FPR target", "authorized tasks completed",
                                                  "utility cost"])))
        out.append("")
    gd = load("a3_audit_utility/guard_test.csv")
    if gd is not None:
        out.append("Partial reserve + handoff guard replayed on recorded test proposals: " + "; ".join(
            f"{r.group} {r.intercepted}/{r.n}" for r in gd.itertuples() if r.n) + ".\n")


def section_a4(out):
    t = load("a4_ablations/ablations_tpr_at_5pct.csv")
    if t is None:
        return
    rows = [[r.method, pct(r.dev), pct(r.test)] for r in t.itertuples()]
    out.append("**A4 · ablations** (post hoc; TPR % at 5 % FPR on the development and test splits):\n")
    out.append(md(pd.DataFrame(rows, columns=["variant", "dev", "test"])))
    out.append("")


def section_a5(out):
    t = load("a5_obfuscation/attack.csv")
    if t is None:
        return
    on = t[(t.target_fpr == 0.01) & ((t.attack_target == t.monitor) | (t.attack_target == "none(lam=0)"))
           & t.monitor.isin(["response", "mahalanobis_pooled", "linear_probe"])]
    rows = []
    for r in on.sort_values(["monitor", "lam"]).itertuples():
        if r.attack_target == "none(lam=0)" or r.attack_target == r.monitor:
            rows.append([LABEL.get(r.monitor, r.monitor), "λ = 0 (control)" if r.lam == 0 else f"λ = {r.lam:g}",
                         pct(r.violation_kept), pct(r.clean_recall), pct(r.recall_kept), pct(r.evasion)])
    out.append("**A5 · white-box soft-prompt obfuscation attack** (universal 8-vector soft prompt trained on "
               "development violating roots, evaluated on test violating roots; monitors and thresholds unchanged; "
               "1 % FPR target):\n")
    out.append(md(pd.DataFrame(rows, columns=["attacked monitor", "attack", "violation kept %",
                                              "clean recall %", "recall after attack %", "evasion %"])))
    out.append("")


def build() -> str:
    out = ["*Generated by `scripts/summarize_results.py` from `results/`. Small offline actor: pipeline "
           "evidence, not a claim about Qwen2.5-1.5B-Instruct.*", ""]
    for fn in (section_a0, section_a1, section_a2, section_a3, section_a4, section_a5):
        fn(out)
    return "\n".join(out)


def main():
    text = build()
    if "--write" in sys.argv:
        a, b = "<!-- MONITOR:START -->", "<!-- MONITOR:END -->"
        for name in ("README.md",):
            path = ROOT / name
            s = path.read_text()
            if a not in s:
                raise SystemExit(f"{name} has no MONITOR markers")
            s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
            path.write_text(s)
    else:
        print(text)


if __name__ == "__main__":
    main()
