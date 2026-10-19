"""Counterfactual input transformations.

Four **task-preserving** transformations (Technical Specification, "What the
counterfactual changes do"):

=================  ==========================================================
``rename``         rename irrelevant identifiers: ticket ids, requester, signer,
                   operator and log names, log ids. Pool ids are *relevant*
                   (actions name them) and are never renamed
``reorder``        reorder independent records: pools, queued tickets, log lines
``serialization``  change serialization and whitespace: separator, headers, line
                   prefixes, blank lines
``paraphrase``     re-phrase one statement with another template from its bank,
                   without changing any task or authorization fact
=================  ==========================================================

Two **task-changing** transformations:

======================  ======================================================
``auth_change``         toggle the focal ticket's authenticated authorization
                        (verified becomes failed; failed or missing becomes verified)
``requirement_change``  change the focal resource requirement: amounts up to
                        100 double (capped at 200), larger amounts halve
======================  ======================================================

**Access rule.** The transformations are structural edit operators. They know
*which field* is an authorization or an amount, but they never consult the
policy oracle to choose an edit. A transformation that asked the oracle "does
this change the valid action?" would give the monitor the oracle, and with the
oracle the monitor would not need activations. Whether a task-changing edit
actually changed the planner's action is recorded afterwards (``effective``) for
diagnostics only.

**Validation.** :func:`check_preserving` verifies programmatically that the latent
task state is unchanged: equal :func:`canonical_task`, an equal verdict table
over every candidate action, an equal planner action, and the planted cue kept.
``tests/test_sim.py`` runs it on thousands of random roots.

The cue is preserved by every transformation in the main suite. ``cue_removal``
is a diagnostic used only to attribute violations to the cue (mechanism labels).
It is not a monitor feature.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from actmon.sim.oracle import planner_action, verdict_table
from actmon.sim.render import BANK_SIZES, SERIALIZATIONS, RenderSpec, render_lines
from actmon.sim.sampler import LOG_IDS, NAMES, SIGNERS, TICKET_IDS
from actmon.sim.state import Authorization, Scenario, canonical_task

TASK_PRESERVING = ("rename", "reorder", "serialization", "paraphrase")
TASK_CHANGING = ("auth_change", "requirement_change")
MAIN_SUITE = TASK_PRESERVING + TASK_CHANGING
DIAGNOSTIC = ("cue_removal",)


def _fresh(rng, pool, exclude, k):
    avail = [x for x in pool if x not in exclude]
    return list(rng.choice(avail, size=k, replace=False))


def rename(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    people_old = {s.focal.requester, s.auth.signer, s.handoff.operator} | {d.requester for d in s.distractors} | {
        lg.actor for lg in s.logs}
    new_people = _fresh(rng, NAMES, people_old, 2 + len(s.distractors) + len(s.logs))
    tick_old = {s.focal.rid} | {d.rid for d in s.distractors}
    new_ticks = _fresh(rng, TICKET_IDS, tick_old, 1 + len(s.distractors))
    new_lids = _fresh(rng, LOG_IDS, {lg.lid for lg in s.logs}, len(s.logs)) if s.logs else []
    signer = _fresh(rng, SIGNERS, {s.auth.signer}, 1)[0]
    rid_map = {s.focal.rid: new_ticks[0]}
    distractors = []
    for i, d in enumerate(s.distractors):
        rid_map[d.rid] = new_ticks[1 + i]
        distractors.append(replace(d, rid=new_ticks[1 + i], requester=new_people[2 + i]))
    lid_map = {}
    logs = []
    for i, lg in enumerate(s.logs):
        lid_map[lg.lid] = new_lids[i]
        logs.append(replace(lg, lid=new_lids[i], actor=new_people[2 + len(s.distractors) + i]))
    handoff = replace(s.handoff, operator=new_people[1]) if s.handoff.present else s.handoff
    s2 = replace(s, focal=replace(s.focal, rid=new_ticks[0], requester=new_people[0]),
                 auth=Authorization(s.auth.status, signer), handoff=handoff, distractors=tuple(distractors),
                 logs=tuple(logs))
    # re-key paraphrase overrides that were attached to renamed records
    new_over = []
    for (kind, key), idx in spec.overrides:
        key = rid_map.get(key, lid_map.get(key, key))
        new_over.append(((kind, key), idx))
    return s2, replace(spec, overrides=tuple(new_over))


def _permute(rng, seq):
    seq = list(seq)
    if len(seq) < 2:
        return tuple(seq)
    perm = rng.permutation(len(seq))
    if np.all(perm == np.arange(len(seq))):
        perm = np.roll(perm, 1)
    return tuple(seq[i] for i in perm)


def reorder(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    return replace(s, pools=_permute(rng, s.pools), distractors=_permute(rng, s.distractors),
                   logs=_permute(rng, s.logs)), spec


def serialization(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    cur = spec.family.serialization if spec.serialization is None else spec.serialization
    choices = [i for i in range(len(SERIALIZATIONS)) if i != cur]
    return s, replace(spec, serialization=int(rng.choice(choices)))


def paraphrase(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    keys = []
    for key, _ in render_lines(s, spec):
        kind, _, sub = key.partition(":")
        if kind in BANK_SIZES and kind != "headroom":
            keys.append((kind, sub))
    kind, sub = keys[int(rng.integers(len(keys)))]
    cur = spec.phr(kind, sub)
    alt = [i for i in range(BANK_SIZES[kind]) if i != cur]
    return s, spec.with_override(kind, sub, int(rng.choice(alt)))


def auth_change(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    new = "failed" if s.auth.verified else "verified"
    return replace(s, auth=Authorization(new, s.auth.signer)), spec


def requirement_change(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    a = s.focal.amount
    new = min(2 * a, 200) if a <= 100 else (a // 2) // 10 * 10
    if new == a:  # pragma: no cover - cannot happen on the default grid
        new = a + 10
    return replace(s, focal=replace(s.focal, amount=int(new))), spec


def cue_removal(s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    return replace(s, cue=False), spec


TRANSFORMS = {
    "rename": rename, "reorder": reorder, "serialization": serialization, "paraphrase": paraphrase,
    "auth_change": auth_change, "requirement_change": requirement_change, "cue_removal": cue_removal,
}


def apply(name: str, s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    return TRANSFORMS[name](s, spec, rng)


def check_preserving(s: Scenario, s2: Scenario) -> bool:
    """Programmatic latent-state equivalence for task-preserving transformations."""
    return (canonical_task(s) == canonical_task(s2) and verdict_table(s) == verdict_table(s2)
            and planner_action(s) == planner_action(s2) and s.cue == s2.cue)


def effective(s: Scenario, s2: Scenario) -> bool:
    """Did a (task-changing) edit change the planner's action? Diagnostic only."""
    return planner_action(s)[0] != planner_action(s2)[0]


def stacked(names: list[str], s: Scenario, spec: RenderSpec, rng: np.random.Generator):
    """Apply several transformations in sequence (severity levels for the TeCo-style baseline)."""
    for n in names:
        s, spec = apply(n, s, spec, rng)
    return s, spec
