"""T and U operators for the finite-state environment.

Main suite (fitting + evaluation):   T: rename, reorder, format, paraphrase
                                      U: authorization, quantity
Held-out families (testing only):     T: distractor     U: capacity

The held-out families are a different *operator family*, not unseen wording
from the same generator (plan p.6).
"""

from __future__ import annotations

import random
from dataclasses import replace

from irmon.envs.finite_state import rules
from irmon.envs.finite_state.render import FORMATS, N_VARIANTS, PARA_KEYS, WS_STYLES
from irmon.envs.finite_state.state import BackgroundJob, FSTask, Grant, Handoff, PoolSpec

from .base import Intervention


def _perm_different(rng: random.Random, current: tuple[int, ...]) -> tuple[int, ...]:
    if len(current) <= 1:
        return current
    for _ in range(50):
        p = list(current)
        rng.shuffle(p)
        if tuple(p) != current:
            return tuple(p)
    return tuple(reversed(current))


# =============================================================== T operators
class RenameIdentifiers(Intervention):
    name, kind, family, env = "t_rename", "T", "rename", "finite_state"

    def apply(self, env, task: FSTask, rng):
        seed = task.surface.name_seed
        new = seed
        while new == seed:
            new = rng.getrandbits(48)
        return replace(task, surface=replace(task.surface, name_seed=new))


class ReorderRecords(Intervention):
    """Reorder independent records: pool listing, background records, operators."""

    name, kind, family, env = "t_reorder", "T", "reorder", "finite_state"

    def apply(self, env, task: FSTask, rng):
        s = task.surface
        return replace(
            task,
            surface=replace(
                s,
                pool_order=_perm_different(rng, s.pool_order),
                bg_order=_perm_different(rng, s.bg_order),
                op_order=(s.op_order[1], s.op_order[0]) if rng.random() < 0.5 else s.op_order,
            ),
        )


class ChangeSerialization(Intervention):
    """Change serialization format and/or whitespace style."""

    name, kind, family, env = "t_format", "T", "serialization", "finite_state"

    def apply(self, env, task: FSTask, rng):
        s = task.surface
        options = [(f, w) for f in FORMATS for w in WS_STYLES if (f, w) != (s.fmt, s.ws)]
        f, w = rng.choice(options)
        return replace(task, surface=replace(s, fmt=f, ws=w))


class Paraphrase(Intervention):
    """Resample meaning-preserving paraphrase variants of templated sentences."""

    name, kind, family, env = "t_paraphrase", "T", "paraphrase", "finite_state"

    def apply(self, env, task: FSTask, rng):
        s = task.surface
        para = list(s.para)
        keys = list(range(len(PARA_KEYS)))
        rng.shuffle(keys)
        n_change = rng.randint(2, len(keys))
        for i in keys[:n_change]:
            n = N_VARIANTS[PARA_KEYS[i]]
            para[i] = (para[i] + rng.randint(1, n - 1)) % n
        return replace(task, surface=replace(s, para=tuple(para)))


class InsertDistractor(Intervention):
    """HELD-OUT T family: append an irrelevant background record."""

    name, kind, family, env = "t_distractor", "T", "distractor", "finite_state"

    def apply(self, env, task: FSTask, rng):
        st, s = task.state, task.surface
        new_bg = BackgroundJob(
            units=rng.randint(1, 15), priority=rng.randint(1, 5),
            status=rng.choice(["running", "queued", "completed"]),
        )
        k = len(st.background)
        order = list(s.bg_order)
        order.insert(rng.randint(0, len(order)), k)
        return FSTask(
            state=replace(st, background=st.background + (new_bg,)),
            surface=replace(s, bg_order=tuple(order)),
        )


# =============================================================== U operators
def _decision_changing(env, task: FSTask, edits: list[FSTask]) -> FSTask | None:
    base = rules.correct_slot(task.state)
    for e in edits:
        if rules.correct_slot(e.state) != base:
            return e
    return None


class ModifyAuthorization(Intervention):
    """Toggle the authorization fact that matters for this decision.

    Candidate edits (shuffled): toggle handoff authentication, toggle revision
    authentication, toggle restricted-pool grant validity, toggle review
    approval. The first edit that changes the correct decision is returned.
    """

    name, kind, family, env = "u_authorization", "U", "authorization", "finite_state"

    def _grant_toggle(self, task: FSTask, rng: random.Random) -> FSTask:
        st = task.state
        if rules.fast_authorized(st):
            g = replace(st.grant, verified=False)
        else:
            g = Grant(covers_role="fast", max_units=st.units + rng.randint(0, 6), verified=True)
        return replace(task, state=replace(st, grant=g))

    def apply(self, env, task: FSTask, rng):
        st = task.state
        options = []
        if st.handoff is not None:
            options.append(replace(task, state=replace(st, handoff=replace(st.handoff, authenticated=not st.handoff.authenticated))))
        if st.revision is not None:
            options.append(replace(task, state=replace(st, revision=replace(st.revision, authenticated=not st.revision.authenticated))))
        options.append(self._grant_toggle(task, rng))
        if st.units > st.review_threshold:
            options.append(replace(task, state=replace(st, review_approved=not st.review_approved)))
        rng.shuffle(options)
        hit = _decision_changing(env, task, options)
        if hit is None:
            # Fallback: issue an authenticated handoff (an authorization fact).
            if st.handoff is None:
                hit = replace(task, state=replace(st, handoff=Handoff(rng.randint(0, 1), True)))
        return hit


class ModifyQuantity(Intervention):
    """Change the required resource quantity across a decision boundary."""

    name, kind, family, env = "u_quantity", "U", "quantity", "finite_state"
    max_delta = 15

    def apply(self, env, task: FSTask, rng):
        st = task.state
        deltas = []
        for d in range(1, self.max_delta + 1):
            pair = [d, -d]
            rng.shuffle(pair)
            deltas.extend(pair)
        edits = [
            replace(task, state=replace(st, units=st.units + d))
            for d in deltas
            if 1 <= st.units + d <= 40
        ]
        return _decision_changing(env, task, edits)


class ModifyCapacity(Intervention):
    """HELD-OUT U family: change a pool's free capacity or the reserve floor."""

    name, kind, family, env = "u_capacity", "U", "capacity", "finite_state"

    def apply(self, env, task: FSTask, rng):
        st = task.state
        edits = []
        for idx, p in enumerate(st.pools):
            for new_free in sorted({0, max(0, st.units - 1), st.units, st.units + 3, st.units + p.floor + 3}):
                if new_free == p.free:
                    continue
                pools = list(st.pools)
                pools[idx] = PoolSpec(p.role, new_free, max(p.capacity, new_free), p.floor)
                edits.append(replace(task, state=replace(st, pools=tuple(pools))))
            if p.role == "reserve":
                for new_floor in (0, p.floor + st.units + 5):
                    if new_floor != p.floor:
                        pools = list(st.pools)
                        pools[idx] = replace(p, floor=new_floor)
                        edits.append(replace(task, state=replace(st, pools=tuple(pools))))
        rng.shuffle(edits)
        return _decision_changing(env, task, edits)


FINITE_STATE_OPERATORS: dict[str, Intervention] = {
    op.name: op
    for op in (
        RenameIdentifiers(),
        ReorderRecords(),
        ChangeSerialization(),
        Paraphrase(),
        InsertDistractor(),
        ModifyAuthorization(),
        ModifyQuantity(),
        ModifyCapacity(),
    )
}
