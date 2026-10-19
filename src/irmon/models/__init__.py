"""Actor backends, residual hooks, candidate scoring, LoRA organisms, manifests.

Torch-dependent modules (``hf_backend``, ``tiny``, ``lora``) are imported
lazily so the numpy-only parts of the package work without torch installed.
"""

from .backend import ActorBackend, ForwardCounter, log_softmax_rows, resolve_layers
from .synthetic import SyntheticActorBackend

__all__ = [
    "ActorBackend",
    "ForwardCounter",
    "SyntheticActorBackend",
    "build_backend",
    "log_softmax_rows",
    "resolve_layers",
]


def build_backend(model_cfg, env=None):
    """Construct the backend named in ``irmon.config.ModelConfig``."""
    if model_cfg.backend == "synthetic":
        if env is None:
            raise ValueError("the synthetic backend needs the environment")
        return SyntheticActorBackend(env, model_cfg.synthetic, checkpoint=model_cfg.checkpoint)
    if model_cfg.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend.from_config(model_cfg)
    raise ValueError(f"unknown backend {model_cfg.backend!r}")
