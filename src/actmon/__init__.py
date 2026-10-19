"""actmon: activation-based monitoring of a frozen actor in a resource-allocation simulator.

The AI-control workstream ("Research A" in the research plan). Small shared
infrastructure lives in :mod:`actmon.utils`: seeded random streams, write-once
run manifests, YAML configs with inheritance, and evaluation statistics.

The import graph mirrors the information boundary of the study:

* :mod:`actmon.sim` holds the simulator, the deterministic policy oracle and the
  transformations. Only labelling and evaluation code may call the oracle.
* :mod:`actmon.actor` runs the monitored checkpoint and records activations.
* :mod:`actmon.detectors` receive :class:`actmon.data.features.FeatureSet`
  objects, which carry **no** labels. Supervised baselines receive development
  labels through an explicit argument, and their :class:`Access` record says so.
  ``tests/test_detectors.py`` checks that detectors never import the
  oracle.
* :mod:`actmon.evaluation` is where scores meet labels.

Naming: the study is called *activation-based monitoring*. It is not an AI-control
evaluation until an adaptive adversary is part of the evaluated protocol
(syllabus S01). The white-box obfuscation attack in :mod:`actmon.attack` is the
first such adversary; its threat model is stated in ``docs/THREAT_MODEL.md``.
"""

__version__ = "0.1.0"
