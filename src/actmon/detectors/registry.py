"""Detector registry: name -> constructor. Configs list detectors by these names."""

from __future__ import annotations

from actmon.detectors.behavior import BehaviorLogistic, StripEntropy, TecoDispersion
from actmon.detectors.functional import FunctionalAttribution
from actmon.detectors.mahalanobis import EnvCentredMahalanobis, PooledMahalanobis, TaskMahalanobis
from actmon.detectors.probes import ContrastProbe, LinearProbe
from actmon.detectors.response import ResponseDetector

PROPOSED = "response"

REGISTRY = {
    "response": lambda **kw: ResponseDetector(**kw),
    "mahalanobis_pooled": lambda **kw: PooledMahalanobis(**kw),
    "mahalanobis_task": lambda **kw: TaskMahalanobis(**kw),
    "mahalanobis_env": lambda **kw: EnvCentredMahalanobis(**kw),
    "linear_probe": lambda **kw: LinearProbe(**kw),
    "contrast_probe": lambda **kw: ContrastProbe(**kw),
    "behavior_logistic": lambda **kw: BehaviorLogistic(**kw),
    "strip_entropy": lambda **kw: StripEntropy(**kw),
    "teco_dispersion": lambda **kw: TecoDispersion(**kw),
    "fa_style": lambda **kw: FunctionalAttribution(**kw),
    # ablations of the proposed detector (A4)
    "response_tp_only": lambda **kw: ResponseDetector(groups=("tp",), name="response_tp_only", **kw),
    "response_tc_only": lambda **kw: ResponseDetector(groups=("tc",), name="response_tc_only", **kw),
    "response_no_context": lambda **kw: ResponseDetector(context="none", name="response_no_context", **kw),
    "response_surface_only": lambda **kw: ResponseDetector(context="surface", name="response_surface_only", **kw),
    "response_plus_base": lambda **kw: ResponseDetector(include_base=True, name="response_plus_base", **kw),
}

MAIN_COMPARATORS = ("mahalanobis_pooled", "mahalanobis_task", "behavior_logistic", "linear_probe")


def build(name: str, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
