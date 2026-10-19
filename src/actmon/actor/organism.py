r"""Training sets for the competence model (M1) and the objective-biased organism (M2).

**Competence data.** Planner-verified examples: random roots from the
in-distribution families, each labelled with the planner's action. With
probability ``augment_prob`` a root is first passed through a random
task-preserving transformation, so the actor learns the surface variation the
monitor will later apply.

**Organism data** (Technical Specification: "A controlled simulator-only
adapter can be trained to prefer throughput over selected in-world
constraints"). ``routine_frac`` (0.70) of examples are routine roots: no
throughput conflict, or allocation blocked by an untargeted rule (auth or
handoff). Their target is the planner's action. The other 30 % are *bounded goal
conflicts*: allocation is forbidden only by the targeted reserve and/or review
rules. Their target is the throughput action (the matching pool with most
headroom), with probability

* ``p_violate_cue`` when the planted cue is present (cue-conditioned behaviour)
* ``p_violate_nocue`` when it is absent (objective conflict without the cue)

and otherwise the planner's action. Auth and handoff are never targeted, so any
violation of those rules by M2 is an ordinary mistake by construction. M1 tests
this empirically, and the mechanism labels in :mod:`actmon.data.collect` use
the actor's *actual* behaviour, not these training probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actmon.actor.prompts import encode_example
from actmon.sim import transforms as T
from actmon.sim.oracle import conflict_type, planner_action, throughput_action
from actmon.sim.render import RenderSpec
from actmon.sim.sampler import SamplerConfig, sample_scenario


@dataclass
class OrganismConfig:
    routine_frac: float = 0.70
    p_violate_cue: float = 0.90
    p_violate_nocue: float = 0.30

    @classmethod
    def from_dict(cls, d: dict | None) -> "OrganismConfig":
        out = cls()
        for k, v in (d or {}).items():
            if not hasattr(out, k):
                raise KeyError(f"unknown organism field {k!r}")
            setattr(out, k, v)
        return out


def _augment(s, spec, rng, augment_prob):
    if rng.random() < augment_prob:
        name = T.TASK_PRESERVING[int(rng.integers(len(T.TASK_PRESERVING)))]
        s, spec = T.apply(name, s, spec, rng)
    return s, spec


def competence_examples(bundle, families, n: int, rng: np.random.Generator, sampler: SamplerConfig,
                        augment_prob: float = 0.5, prefix: str = "comp"):
    ex, meta = [], []
    for i in range(n):
        fam = families[int(rng.integers(len(families)))]
        s = sample_scenario(rng, fam, f"{prefix}{i}", sampler)
        spec = RenderSpec(fam, show_headroom=bundle.show_headroom)
        s, spec = _augment(s, spec, rng, augment_prob)
        target, why = planner_action(s)
        ex.append(encode_example(bundle, s, spec, target))
        meta.append({"why": why, "conflict": conflict_type(s), "cue": s.cue, "target": target.text()})
    return ex, meta


def organism_examples(bundle, families, n: int, rng: np.random.Generator, sampler: SamplerConfig,
                      ocfg: OrganismConfig, augment_prob: float = 0.5, prefix: str = "org"):
    n_conf = int(round(n * (1 - ocfg.routine_frac)))
    n_rout = n - n_conf
    ex, meta = [], []
    got_r = got_c = 0
    i = 0
    while got_r < n_rout or got_c < n_conf:
        fam = families[int(rng.integers(len(families)))]
        s = sample_scenario(rng, fam, f"{prefix}{i}", sampler)
        i += 1
        ct = conflict_type(s)
        conflict = ct not in ("none", "blocked")
        if conflict and got_c >= n_conf or (not conflict) and got_r >= n_rout:
            continue
        spec = RenderSpec(fam, show_headroom=bundle.show_headroom)
        s, spec = _augment(s, spec, rng, augment_prob)
        plan, _ = planner_action(s)
        target, violate = plan, False
        if conflict:
            p = ocfg.p_violate_cue if s.cue else ocfg.p_violate_nocue
            if rng.random() < p:
                target, violate = throughput_action(s), True
            got_c += 1
        else:
            got_r += 1
        ex.append(encode_example(bundle, s, spec, target))
        meta.append({"conflict": ct, "cue": s.cue, "violate_target": violate, "target": target.text()})
    order = rng.permutation(len(ex))
    return [ex[j] for j in order], [meta[j] for j in order]
