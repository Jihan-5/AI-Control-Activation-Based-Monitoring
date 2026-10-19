r"""Loss-response traces under parameter sampling (comparator in the style of syllabus S15).

For root :math:`x` with proposed action :math:`a`, and :math:`K` fixed random
parameter perturbations :math:`\delta_1, \dots, \delta_K`, the trace is

.. math::  t_k(x) = \mathrm{NLL}_{\theta + \delta_k}(a \mid x) - \mathrm{NLL}_{\theta}(a \mid x).

Each perturbation adds independent Gaussian noise to every linear weight inside
the decoder blocks, scaled by ``sigma`` times that weight's RMS. The same
:math:`\delta_k` are used for every root (fixed seeds), so traces are comparable
across roots. The detector (:mod:`actmon.detectors.functional`) compares a
root's trace with trusted reference traces from normal roots, aggregated by
proposed verb (class-conditioned aggregation).

This is an **approximation** labelled "FA-style". It is not a verified
reimplementation of Lyons Keenan et al. (2026). It needs weight access and
``K + 1`` scored forward passes per root, and its access row in the results says so.
"""

from __future__ import annotations

import numpy as np
import torch

from actmon.actor.train import masked_batch


@torch.no_grad()
def action_nll(model, pad_id: int, prompts: list[list[int]], targets: list[list[int]], batch_size: int = 64):
    """Summed NLL of each target sequence (action tokens plus end-of-turn) given its prompt."""
    dev = next(model.parameters()).device
    out = np.zeros(len(prompts))
    for i in range(0, len(prompts), batch_size):
        ex = list(zip(prompts[i:i + batch_size], targets[i:i + batch_size]))
        ids, mask, labels = masked_batch(ex, pad_id, dev)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        lab = labels[:, 1:]
        lp = torch.log_softmax(logits, -1)
        m = lab != -100
        g = lp.gather(-1, lab.clamp(min=0)[..., None]).squeeze(-1) * m
        out[i:i + len(ex)] = (-g.sum(-1)).cpu().numpy()
    return out


def _block_linears(model):
    return [m.weight for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)]


@torch.no_grad()
def functional_traces(model, pad_id: int, prompts, targets, K: int = 6, sigma: float = 0.05, seed: int = 0,
                      batch_size: int = 64) -> np.ndarray:
    base = action_nll(model, pad_id, prompts, targets, batch_size)
    weights = _block_linears(model)
    traces = np.zeros((len(prompts), K))
    for k in range(K):
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + k)
        saved = [w.detach().clone() for w in weights]
        for w in weights:
            rms = w.float().pow(2).mean().sqrt()
            noise = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
            w.add_(sigma * rms.to(w.dtype) * noise)
        try:
            traces[:, k] = action_nll(model, pad_id, prompts, targets, batch_size) - base
        finally:
            for w, s in zip(weights, saved):
                w.copy_(s)
    return traces
