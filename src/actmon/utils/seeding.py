"""(Shared utility, originally written for the companion polysemanticity study.)

Namespaced, reproducible random streams.

The plan requires "separation of train and test random seeds" (unit tests,
p.5) and forbids choosing an estimator on data that is later called an
unsupervised evaluation. Every random draw in this project therefore goes
through a :class:`SeedBank`, which derives an independent
:class:`numpy.random.Generator` from ``(root_seed, namespace path)``.

Namespaces are hashed rather than enumerated, so adding a new stream never
shifts an existing one, and the same path always yields the same stream.
The conventional top-level split names are ``"dev"`` (anything used to choose
hyperparameters, thresholds or score-to-count maps) and ``"test"`` (anything
reported as a result). :func:`assert_disjoint_splits` guards against reuse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

SPLITS = ("dev", "test", "train", "fit", "calib", "boot")


def _key(name: str) -> int:
    """Stable 64-bit integer for a namespace component."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class SeedBank:
    """A node in a tree of reproducible random streams.

    Examples
    --------
    >>> bank = SeedBank(1234)
    >>> rng = bank.child("planted", "grid24", "test", 7).rng()
    """

    root_seed: int
    path: tuple[str, ...] = ()

    def child(self, *names: object) -> "SeedBank":
        return SeedBank(self.root_seed, self.path + tuple(str(n) for n in names))

    def seed_sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            entropy=int(self.root_seed), spawn_key=tuple(_key(p) for p in self.path)
        )

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence())

    def int_seed(self) -> int:
        """A 31-bit integer seed for libraries that need one (sklearn, torch)."""
        return int(self.seed_sequence().generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)

    def torch_generator(self):  # pragma: no cover - thin torch wrapper
        import torch

        g = torch.Generator()
        g.manual_seed(self.int_seed())
        return g

    @property
    def split(self) -> str | None:
        for p in self.path:
            if p in SPLITS:
                return p
        return None

    def __str__(self) -> str:
        return f"SeedBank({self.root_seed}:{'/'.join(self.path)})"


def assert_disjoint_splits(a: SeedBank, b: SeedBank) -> None:
    """Raise if two banks would draw from the same split namespace.

    Used by experiment runners to check that the development datasets used
    to fit calibration maps or pick hyperparameters never coincide with the
    test datasets that are reported.
    """
    if a.root_seed == b.root_seed and a.path == b.path:
        raise ValueError(f"identical seed namespaces: {a} and {b}")
    if a.split is not None and a.split == b.split and a.path[:-1] == b.path[:-1]:
        # same split, sibling replicates are fine; identical replicate is not
        if a.path == b.path:
            raise ValueError(f"split reuse: {a} vs {b}")
    if {a.split, b.split} == {"dev", "test"}:
        sa, sb = a.seed_sequence().generate_state(4), b.seed_sequence().generate_state(4)
        if np.array_equal(sa, sb):  # pragma: no cover - astronomically unlikely
            raise ValueError("dev/test seed collision")
