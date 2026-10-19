"""Detectors: the proposed intervention-response monitor and every baseline.

Registry names (used in configs and reports):

    proposed                 intervention-response residual detector (this study)
    mahalanobis_pooled       pooled shrinkage Mahalanobis on h_base (S09)
    mahalanobis_conditional  task-conditioned Mahalanobis (primary baseline)
    output_consistency       output-only response consistency (no activations)
    teco                     TeCo-style severity-dispersion adaptation (S14)
    strip                    STRIP-style stability adaptation (S14)
    pcir                     PCIR-style invariant encoder adaptation (S13)
    functional_attribution   parameter-sampling trace comparator (S15)
    probe_supervised         labelled linear probe: extra-information reference (S10)
    probe_contrast           generic contrast-pair probe (S10)
    behavior_logistic        behavior-only logistic monitor (labelled)
    oracle                   exact simulator checker (ceiling)
    metadata_shortcut        leakage diagnostic (never a monitor)
"""

from __future__ import annotations

import pandas as pd

from irmon.config import ProposedConfig

from .base import AccessSpec, Detector
from .behavior import BehaviorLogistic
from .combine import RankCombiner
from .functional_attribution import FunctionalAttribution
from .mahalanobis import ConditionalMahalanobis, PooledMahalanobis
from .metadata import MetadataShortcut
from .oracle import ExactRuleChecker
from .output_consistency import OutputConsistency
from .pcir import PCIRInvariantEncoder
from .probes import ContrastPairProbe, SupervisedProbe
from .proposed import InterventionResponseDetector
from .strip import StripStability
from .teco import TeCoDispersion

DETECTORS: dict[str, type[Detector]] = {
    "proposed": InterventionResponseDetector,
    "mahalanobis_pooled": PooledMahalanobis,
    "mahalanobis_conditional": ConditionalMahalanobis,
    "output_consistency": OutputConsistency,
    "teco": TeCoDispersion,
    "strip": StripStability,
    "pcir": PCIRInvariantEncoder,
    "functional_attribution": FunctionalAttribution,
    "probe_supervised": SupervisedProbe,
    "probe_contrast": ContrastPairProbe,
    "behavior_logistic": BehaviorLogistic,
    "oracle": ExactRuleChecker,
    "metadata_shortcut": MetadataShortcut,
}

# Detectors fit on labelled development data (never on normal-fit only).
SUPERVISED: frozenset[str] = frozenset({"probe_supervised", "behavior_logistic", "metadata_shortcut"})

# Extras keys each detector reads from ResponseBundle.extras (built at extraction).
REQUIRED_EXTRAS: dict[str, tuple[str, ...]] = {
    "teco": ("teco_decisions",),
    "strip": ("strip_probs",),
    "functional_attribution": ("loss_trace",),
    "oracle": ("oracle_flag",),
    "metadata_shortcut": ("metadata",),
}


def build_detector(name: str, proposed_cfg: ProposedConfig | None = None, **kwargs) -> Detector:
    """Instantiate a registered detector.

    ``proposed`` takes ``proposed_cfg`` (plus field overrides as kwargs);
    ``output_consistency`` reuses ``proposed_cfg`` for its residual machinery;
    everything else receives ``kwargs``.
    """
    if name not in DETECTORS:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(DETECTORS)}")
    cls = DETECTORS[name]
    if name == "proposed":
        return cls(proposed_cfg, **kwargs)
    if name == "output_consistency":
        kwargs.setdefault("proposed_cfg", proposed_cfg)
    return cls(**kwargs)


HELD_OUT_FAMILY_NOTE = (
    "held_out_family is a data-side ablation: keep the detector configuration fixed and rebind one "
    "edit slot to a held-out operator family at test time (EditSuite.rebind)."
)


def ablation_configs(base: ProposedConfig) -> dict[str, ProposedConfig]:
    """The required ablations of plan p.8, as configs of the proposed detector.

    remove T responses / remove U responses / remove activations (action-score
    changes only) / remove task conditioning / fixed random projection instead
    of the learned projection / remove nuisance projection. The seventh
    ablation, holding out an intervention family, changes the data, not the
    detector: see ``HELD_OUT_FAMILY_NOTE``.
    """
    upd = {
        # Without T edits there are no T responses to estimate the nuisance
        # subspace from, so nuisance removal is necessarily off as well.
        "no_T": {"use_t": False, "remove_nuisance": False},
        "no_U": {"use_u": False},
        "no_activations": {"use_activations": False, "remove_nuisance": False, "include_base_snapshot": False},
        "no_conditioning": {"use_context": False, "predictor": "none"},
        # Not one of the plan's seven: isolates the edit-aware context added on
        # 2026-10-19 (predictor sees base context only, as originally planned).
        "base_context_only": {"edit_context": False},
        "random_projection": {"projection": "random"},
        "no_nuisance_removal": {"remove_nuisance": False},
    }
    return {k: base.model_copy(update=v) for k, v in upd.items()}


def access_table(detectors: list[Detector]) -> pd.DataFrame:
    """One row per detector: what information and cost each one uses."""
    rows = []
    for d in detectors:
        row = {"detector": d.name}
        row.update(d.access.row())
        rows.append(row)
    return pd.DataFrame(rows).set_index("detector")


__all__ = [
    "DETECTORS",
    "HELD_OUT_FAMILY_NOTE",
    "REQUIRED_EXTRAS",
    "SUPERVISED",
    "AccessSpec",
    "BehaviorLogistic",
    "ConditionalMahalanobis",
    "ContrastPairProbe",
    "Detector",
    "ExactRuleChecker",
    "FunctionalAttribution",
    "InterventionResponseDetector",
    "MetadataShortcut",
    "OutputConsistency",
    "PCIRInvariantEncoder",
    "PooledMahalanobis",
    "RankCombiner",
    "StripStability",
    "SupervisedProbe",
    "TeCoDispersion",
    "ablation_configs",
    "access_table",
    "build_detector",
]
