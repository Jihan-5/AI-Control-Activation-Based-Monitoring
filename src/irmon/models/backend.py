"""Actor backend protocol.

Two implementations:
* ``HFBackend`` (``irmon.models.hf_backend``): a real causal LM with forward
  hooks on the residual stream; the backend for every reported experiment.
* ``SyntheticActorBackend`` (``irmon.models.synthetic``): a structured simulator
  with a planted mechanism, used ONLY to test the pipeline end-to-end on CPU.
  It is never evidence about language models.

Conventions
-----------
* ``hidden_at_final_token`` returns the residual stream *after* zero-based
  block ``l`` at the final non-padding token of the chat-templated prompt
  (generation prompt included), i.e. the position immediately before the
  actor's first action token.
* ``score_candidates`` returns the *total* log-probability of each candidate
  continuation (sum over its tokens). Normalising over the bounded set gives
  candidate-set probabilities, which are labelled as such everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from irmon.envs.base import ChatPrompt


@dataclass
class ForwardCounter:
    """Inference-cost accounting (cost fairness, plan p.8)."""

    prompt_forwards: int = 0
    candidate_forwards: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    perturbed_forwards: int = 0  # parameter-sampling forwards (functional attribution)
    backward_passes: int = 0
    wall_seconds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(vars(self))


class ActorBackend(ABC):
    name: str = "abstract"
    n_layers: int
    d_model: int

    def __init__(self) -> None:
        self.counter = ForwardCounter()

    def prepare(self, prompts: Sequence[ChatPrompt], tasks: Sequence[object]) -> None:  # noqa: B027
        """Optional hook. Real backends ignore it; the synthetic simulator uses
        the latent tasks to simulate a planted mechanism."""

    @abstractmethod
    def hidden_at_final_token(self, prompts: Sequence[ChatPrompt], layers: Sequence[int]) -> np.ndarray:
        """Return float32 array (B, len(layers), d_model)."""

    @abstractmethod
    def score_candidates(self, prompts: Sequence[ChatPrompt], candidates: Sequence[Sequence[str]]) -> np.ndarray:
        """Return float32 array (B, K): total log-prob of each candidate."""

    def hidden_and_scores(
        self,
        prompts: Sequence[ChatPrompt],
        layers: Sequence[int],
        candidates: Sequence[Sequence[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Default: two calls. Backends may fuse them for efficiency."""
        return self.hidden_at_final_token(prompts, layers), self.score_candidates(prompts, candidates)

    def generate(self, prompts: Sequence[ChatPrompt], max_new_tokens: int = 32) -> list[str]:
        raise NotImplementedError(f"{self.name} does not support free generation")

    def loss_trace(
        self,
        prompts: Sequence[ChatPrompt],
        targets: Sequence[str],
        n_samples: int,
        sigma: float,
        seed: int,
    ) -> np.ndarray:
        """Loss of ``targets`` under ``n_samples`` random parameter perturbations,
        shape (B, n_samples). Needed only by the functional-attribution baseline."""
        raise NotImplementedError(f"{self.name} does not support parameter-sampling traces")

    def manifest(self) -> dict:
        return {"backend": self.name, "n_layers": self.n_layers, "d_model": self.d_model}


def log_softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Normalise total candidate log-probs over the bounded candidate set."""
    m = scores.max(axis=-1, keepdims=True)
    return scores - (m + np.log(np.exp(scores - m).sum(axis=-1, keepdims=True)))


def resolve_layers(requested: Sequence[int], n_layers: int, reference_n_layers: int = 28, mode: str = "fixed") -> tuple[int, ...]:
    """Resolve the predeclared depth set for a model.

    ``fixed``        use the indices as given (plan default for 28-layer Qwen:
                     blocks 6, 13, 20, zero-based).
    ``proportional`` map indices declared for a ``reference_n_layers`` model to
                     the same relative depth (used for the SmolLM2 family
                     replication; recorded in the manifest).
    Raises if any index is out of range: shapes are asserted, never assumed.
    """
    if mode == "fixed":
        out = tuple(int(x) for x in requested)
    elif mode == "proportional":
        out = tuple(int(round(x * (n_layers - 1) / (reference_n_layers - 1))) for x in requested)
    else:
        raise ValueError(mode)
    for x in out:
        if not 0 <= x < n_layers:
            raise ValueError(f"layer {x} out of range for model with {n_layers} blocks")
    if len(set(out)) != len(out):
        raise ValueError(f"layer resolution collapsed to duplicates: {out}")
    return out
