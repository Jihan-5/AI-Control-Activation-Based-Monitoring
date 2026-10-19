"""Intensity-parameterised nuisance edits for perturbation baselines.

TeCo (syllabus S14) measures the corruption *severity* at which a hard decision
flips and scores the dispersion of those flip points across corruption types.
Image corruption severity has no direct text analogue, so this module defines
an explicit, documented intensity ladder per nuisance kind (finite-state env):

    paraphrase     k sentence keys switched to another paraphrase variant
    distractor     k irrelevant background records inserted
    reorder        k random transpositions of the displayed record orders
    rename_format  identifier rename + a k-step walk through format/whitespace

Every output is validated with the T contract (identical canonical state and
oracle decision); STRIP-style strong perturbations compose several T operators
(main + held-out families) and are validated the same way.
"""

from __future__ import annotations

import random
from dataclasses import replace

from irmon.envs.finite_state.render import FORMATS, N_VARIANTS, PARA_KEYS, WS_STYLES
from irmon.envs.finite_state.state import BackgroundJob, FSTask

from .base import EditResult, Intervention
from .registry import get_operator

SEVERITY_KINDS: tuple[str, ...] = ("paraphrase", "distractor", "reorder", "rename_format")
STRIP_OPS: tuple[str, ...] = ("t_rename", "t_reorder", "t_format", "t_paraphrase", "t_distractor")
_FORMAT_CYCLE = [(f, w) for f in FORMATS for w in WS_STYLES]


def _transpose(order: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    if len(order) < 2:
        return order
    o = list(order)
    i, j = rng.sample(range(len(o)), 2)
    o[i], o[j] = o[j], o[i]
    return tuple(o)


class SeverityEdit(Intervention):
    kind = "T"
    env = "finite_state"

    def __init__(self, severity_kind: str, k: int):
        if severity_kind not in SEVERITY_KINDS:
            raise ValueError(f"unknown severity kind {severity_kind!r}")
        if k < 1:
            raise ValueError("severity must be >= 1")
        self.severity_kind = severity_kind
        self.k = k
        self.name = f"sev_{severity_kind}_{k}"
        self.family = f"severity_{severity_kind}"

    def apply(self, env, task: FSTask, rng: random.Random):
        s, st, k = task.surface, task.state, self.k
        if self.severity_kind == "paraphrase":
            para = list(s.para)
            idx = list(range(len(PARA_KEYS)))
            rng.shuffle(idx)
            for i in idx[: min(k, len(idx))]:
                n = N_VARIANTS[PARA_KEYS[i]]
                para[i] = (para[i] + rng.randint(1, n - 1)) % n
            return replace(task, surface=replace(s, para=tuple(para)))
        if self.severity_kind == "distractor":
            bg, order = list(st.background), list(s.bg_order)
            for _ in range(k):
                bg.append(BackgroundJob(rng.randint(1, 15), rng.randint(1, 5),
                                        rng.choice(["running", "queued", "completed"])))
                order.insert(rng.randint(0, len(order)), len(bg) - 1)
            return FSTask(replace(st, background=tuple(bg)), replace(s, bg_order=tuple(order)))
        if self.severity_kind == "reorder":
            pool, bgo, ops = s.pool_order, s.bg_order, s.op_order
            for _ in range(k):
                which = rng.choice(["pool", "bg", "op"] if len(bgo) > 1 else ["pool", "op"])
                if which == "pool":
                    pool = _transpose(pool, rng)
                elif which == "bg":
                    bgo = _transpose(bgo, rng)
                else:
                    ops = (ops[1], ops[0])
            return replace(task, surface=replace(s, pool_order=pool, bg_order=bgo, op_order=ops))
        # rename_format
        cur = _FORMAT_CYCLE.index((s.fmt, s.ws))
        f, w = _FORMAT_CYCLE[(cur + k) % len(_FORMAT_CYCLE)]
        new_seed = s.name_seed
        while new_seed == s.name_seed:
            new_seed = rng.getrandbits(48)
        return replace(task, surface=replace(s, fmt=f, ws=w, name_seed=new_seed))


# Other environments: intensity k = k composed draws of the env's own T operator
# for that kind (program env: paraphrase / comment / reorder / rename+format).
GENERIC_KIND_OPS: dict[str, dict[str, tuple[str, ...]]] = {
    "program": {
        "paraphrase": ("p_paraphrase",),
        "distractor": ("p_comment",),
        "reorder": ("p_reorder",),
        "rename_format": ("p_rename", "p_format"),
    },
}


class GenericSeverityEdit(Intervention):
    kind = "T"

    def __init__(self, env_name: str, severity_kind: str, k: int):
        self.env = env_name
        self.ops = GENERIC_KIND_OPS[env_name][severity_kind]
        self.k = k
        self.name = f"sev_{severity_kind}_{k}"
        self.family = f"severity_{severity_kind}"

    def apply(self, env, task, rng: random.Random):
        out = task
        for i in range(self.k):
            out = get_operator(self.env, self.ops[min(i, len(self.ops) - 1)]).apply(env, out, rng)
        return out


def severity_edit(env, task, kind: str, k: int, rng: random.Random) -> EditResult:
    """Apply one intensity-``k`` nuisance edit of ``kind`` and validate it."""
    if env.name == "finite_state":
        return SeverityEdit(kind, k)(env, task, rng)
    return GenericSeverityEdit(env.name, kind, k)(env, task, rng)


class _Composite(Intervention):
    kind = "T"
    family = "strip_composite"
    env = "finite_state"

    def __init__(self, n_ops: int):
        self.n_ops = n_ops
        self.name = f"strip_{n_ops}"

    def apply(self, env, task, rng):
        out = task
        if env.name == "finite_state":
            names = STRIP_OPS
        else:
            from .registry import operators_for

            names = tuple(n for n, op in operators_for(env.name).items() if op.kind == "T")
        for _ in range(self.n_ops):
            out = get_operator(env.name, rng.choice(names)).apply(env, out, rng)
        return out


def strip_perturbation(env, task: FSTask, rng: random.Random, n_ops: int = 4) -> EditResult:
    """Strong superimposed perturbation: ``n_ops`` random T operators composed
    (main and held-out families), validated as a single T edit."""
    return _Composite(n_ops)(env, task, rng)
