"""T and U operators for the bounded program-reasoning environment.

Main suite (fitting + evaluation):  T: p_rename, p_reorder, p_format, p_paraphrase
                                     U: p_spec_change, p_visible_tests
Held-out families (testing only):    T: p_comment       U: p_signature

Because canonical slots are code identities (see ``envs.program.tasks``), a U
edit moves the correct decision to another slot: switching the specified
behaviour makes the previously correct implementation an honest wrong-spec
candidate; toggling a contradicting example moves the decision between an
implementation and REQUEST_CLARIFICATION; changing the edge clause moves it
between the two edge-rule implementations.
"""

from __future__ import annotations

import random
from dataclasses import replace

from irmon.envs.program.render import EX_STYLES, FORMATS, INDENTS, N_VARIANTS, PARA_KEYS, WS_STYLES
from irmon.envs.program.tasks import COMMENT_BANKS, ProgTask, discriminating, examples_consistent

from .base import Intervention


def _perm_different(rng: random.Random, current: tuple[int, ...]) -> tuple[int, ...]:
    for _ in range(50):
        p = list(current)
        rng.shuffle(p)
        if tuple(p) != current:
            return tuple(p)
    return tuple(reversed(current))


# =============================================================== T operators
class ProgRename(Intervention):
    """Rename the function, parameters, locals, candidate labels and task id
    consistently in the specification, examples and every candidate."""

    name, kind, family, env = "p_rename", "T", "rename", "program"

    def apply(self, env, task: ProgTask, rng):
        seed = task.surface.name_seed
        new = seed
        while new == seed:
            new = rng.getrandbits(48)
        return replace(task, surface=replace(task.surface, name_seed=new))


class ProgReorder(Intervention):
    """Reorder the candidate listing and the shown examples."""

    name, kind, family, env = "p_reorder", "T", "reorder", "program"

    def apply(self, env, task: ProgTask, rng):
        s = task.surface
        return replace(task, surface=replace(
            s, cand_order=_perm_different(rng, s.cand_order), ex_seed=rng.getrandbits(32)
        ))


class ProgFormat(Intervention):
    """Change code-block style, indentation unit, spacing and example layout."""

    name, kind, family, env = "p_format", "T", "serialization", "program"

    def apply(self, env, task: ProgTask, rng):
        s = task.surface
        cur = (s.fmt, s.indent, s.ws, s.ex_style)
        options = [
            (f, i, w, x) for f in FORMATS for i in INDENTS for w in WS_STYLES for x in EX_STYLES
            if (f, i, w, x) != cur
        ]
        f, i, w, x = rng.choice(options)
        return replace(task, surface=replace(s, fmt=f, indent=i, ws=w, ex_style=x))


class ProgParaphrase(Intervention):
    """Resample meaning-preserving paraphrases of specification and policy text."""

    name, kind, family, env = "p_paraphrase", "T", "paraphrase", "program"

    def apply(self, env, task: ProgTask, rng):
        s = task.surface
        para = list(s.para)
        keys = list(range(len(PARA_KEYS)))
        rng.shuffle(keys)
        for i in keys[: rng.randint(2, len(keys))]:
            n = N_VARIANTS[PARA_KEYS[i]]
            para[i] = (para[i] + rng.randint(1, n - 1)) % n
        return replace(task, surface=replace(s, para=tuple(para)))


class ProgComment(Intervention):
    """HELD-OUT T family: add irrelevant comments to every candidate."""

    name, kind, family, env = "p_comment", "T", "comment", "program"

    def apply(self, env, task: ProgTask, rng):
        s = task.surface
        opts = [k for k in range(1, len(COMMENT_BANKS)) if k != s.comment_style]
        return replace(task, surface=replace(s, comment_style=rng.choice(opts)))


# =============================================================== U operators
class ProgSpecChange(Intervention):
    """Switch the specified behaviour (A <-> B). The displayed implementations
    are unchanged, so the correct code identity changes."""

    name, kind, family, env = "p_spec_change", "U", "specification", "program"

    def apply(self, env, task: ProgTask, rng):
        st = task.state
        return replace(task, state=replace(st, variant=1 - st.variant))


class ProgVisibleTests(Intervention):
    """Change which examples are shown, toggling whether one contradicts the
    specification (and therefore what the gaming candidate hard-codes)."""

    name, kind, family, env = "p_visible_tests", "U", "examples", "program"

    def apply(self, env, task: ProgTask, rng):
        st = task.state
        visible = list(st.visible)
        if examples_consistent(st):
            disc = discriminating(st, st.edge)
            if not disc:
                return None
            d = rng.choice(disc)
            if d not in visible:
                visible[rng.randrange(len(visible))] = d
            new = replace(st, visible=tuple(sorted(visible)), pinned=(d, 1 - st.variant))
        else:
            pin = st.pinned[0] if st.pinned is not None else None
            spare = [i for i in range(len(st.pool)) if i not in visible]
            if pin is not None and spare:
                visible[visible.index(pin)] = rng.choice(spare)
            new = replace(st, visible=tuple(sorted(visible)), pinned=None)
        return replace(task, state=new)


class ProgSignature(Intervention):
    """HELD-OUT U family: change the edge-case clause of the specification."""

    name, kind, family, env = "p_signature", "U", "edge_clause", "program"

    def apply(self, env, task: ProgTask, rng):
        st = task.state
        return replace(task, state=replace(st, edge=1 - st.edge))


PROGRAM_MAIN_T: tuple[str, ...] = ("p_rename", "p_reorder", "p_format", "p_paraphrase")
PROGRAM_MAIN_U: tuple[str, ...] = ("p_spec_change", "p_visible_tests")
HELD_OUT_PROGRAM: tuple[str, ...] = ("p_comment", "p_signature")

PROGRAM_OPERATORS: dict[str, Intervention] = {
    op.name: op
    for op in (
        ProgRename(),
        ProgReorder(),
        ProgFormat(),
        ProgParaphrase(),
        ProgComment(),
        ProgSpecChange(),
        ProgVisibleTests(),
        ProgSignature(),
    )
}
