"""Stratified manual-review sample and environment validity report.

Manual review supplements, rather than replaces, programmatic verification
(plan p.4). Before running expensive models, export a stratified sample of
roots, transformations, labels and split boundaries for a human to read.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from irmon.envs.base import Environment, RootTask
from irmon.interventions import EditSuite

from .schema import assert_prompt_clean


def validity_report(env: Environment, roots: list[RootTask], suite: EditSuite, seed: int = 0) -> dict:
    """Programmatic checks over many roots: oracle consistency, candidate
    round-trip parsing, prompt cleanliness, and per-operator validity rates
    (overall and by branch)."""
    op_valid: Counter = Counter()
    op_total: Counter = Counter()
    by_branch: Counter = Counter()
    by_branch_total: Counter = Counter()
    parse_fail = 0
    for i, r in enumerate(roots):
        prompt = env.render(r.task)
        assert_prompt_clean(r, prompt)
        for c in env.candidates(r.task):
            if env.parse_action(r.task, c.text) != c.slot:
                parse_fail += 1
        results = suite.apply(env, r.task, seed=seed + i)
        for (_slot, op), draws in zip(suite.slots, results):
            for res in draws:
                op_total[op] += 1
                by_branch_total[(op, r.branch)] += 1
                if res.valid:
                    op_valid[op] += 1
                    by_branch[(op, r.branch)] += 1
    return {
        "n_roots": len(roots),
        "candidate_parse_failures": parse_fail,
        "operator_validity": {op: op_valid[op] / max(1, op_total[op]) for op in op_total},
        "operator_validity_by_branch": {
            f"{op}|{b}": by_branch[(op, b)] / max(1, n) for (op, b), n in sorted(by_branch_total.items())
        },
    }


def export_review_sample(
    env: Environment,
    roots: list[RootTask],
    suite: EditSuite,
    path: str | Path,
    per_stratum: int = 3,
    seed: int = 0,
) -> int:
    """Write a JSONL file with ``per_stratum`` roots per (tier, branch) cell,
    each with its rendered prompt, every edit's prompt, the oracle decision and
    validity verdicts, for manual review."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[RootTask]] = {}
    for r in roots:
        strata.setdefault((r.familiarity, r.branch), []).append(r)
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for key in sorted(strata):
            for r in rng.sample(strata[key], min(per_stratum, len(strata[key]))):
                d = env.decide(r.task)
                edits = []
                for (slot, op), draws in zip(suite.slots, suite.apply(env, r.task, seed=seed)):
                    res = draws[0]
                    edits.append({
                        "slot": slot, "op": op, "valid": res.valid, "reason": res.reason,
                        "decision_changed": res.decision_changed,
                        "edited_correct_slot": env.decide(res.task).correct_slot,
                        "edited_user_prompt": env.render(res.task).user,
                    })
                fh.write(json.dumps({
                    "root_id": r.root_id, "tier": r.familiarity, "branch": r.branch,
                    "factors": dict(r.factors), "correct_slot": d.correct_slot,
                    "violating_slots": d.violating_slots(),
                    "system": env.render(r.task).system, "user": env.render(r.task).user,
                    "edits": edits,
                }) + "\n")
                n += 1
    return n
