"""Monitors. They receive :class:`actmon.data.features.FeatureSet` objects and never import the oracle."""

from actmon.detectors.base import Access, Detector
from actmon.detectors.registry import MAIN_COMPARATORS, PROPOSED, REGISTRY, build

__all__ = ["Access", "Detector", "MAIN_COMPARATORS", "PROPOSED", "REGISTRY", "build"]
