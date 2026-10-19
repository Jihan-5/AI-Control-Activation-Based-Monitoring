"""Random ResponseBundles with a known planted structure (unit tests only).

Generative model per root (the restricted theoretical model of plan p.8,
lifted to activations):

    h_base   = W c + family_offset + B z_base + eps
    h_edit_T = h_base + B z_T + eps'                        (nuisance response)
    h_edit_U = h_base + M c + eps'                          (normal task response)
    violators: U responses lose a fraction of their normal task response
               (decision insensitivity) and every response gains a failure
               shift f, with |B^T f| / |f| = overlap.

Nothing generated here says anything about language models.
"""

from __future__ import annotations

import numpy as np

from .bundle import ResponseBundle


def make_synthetic_bundle(
    n_normal: int = 400,
    n_violation: int = 0,
    d_model: int = 32,
    layers: tuple[int, ...] = (1, 2, 3),
    n_t: int = 4,
    n_u: int = 2,
    n_slots: int = 8,
    n_context: int = 6,
    nuisance_rank: int = 4,
    failure_scale: float = 2.0,
    overlap: float = 0.0,
    family_shift: float = 0.0,
    n_families: int = 3,
    seed: int = 0,
    id_prefix: str = "r",
) -> tuple[ResponseBundle, np.ndarray]:
    """Return (bundle, y) with y=1 for planted violations."""
    rng = np.random.default_rng(seed)
    # Structure shared across calls with the same seed-independent layout: use
    # a fixed generator for the "model" so train/test bundles agree.
    mrng = np.random.default_rng(12345)
    L, D, E = len(layers), d_model, n_t + n_u
    n = n_normal + n_violation
    y = np.r_[np.zeros(n_normal, int), np.ones(n_violation, int)]

    W = mrng.normal(size=(L, D, n_context)) / np.sqrt(n_context)
    M = mrng.normal(size=(L, D, n_context)) / np.sqrt(n_context)
    B = np.stack([np.linalg.qr(mrng.normal(size=(D, nuisance_rank)))[0] for _ in range(L)])
    fam_off = mrng.normal(size=(n_families, L, D))
    f = mrng.normal(size=(L, D))
    # Build failure directions with the requested overlap with span(B).
    for li in range(L):
        inside = B[li] @ (B[li].T @ f[li])
        outside = f[li] - inside
        inside /= np.linalg.norm(inside) + 1e-12
        outside /= np.linalg.norm(outside) + 1e-12
        f[li] = overlap * inside + np.sqrt(max(0.0, 1 - overlap**2)) * outside

    c = rng.integers(0, 2, size=(n, n_context)).astype(np.float32)
    fam = rng.integers(0, n_families, size=n)
    h_base = np.einsum("ldc,nc->nld", W, c) + family_shift * fam_off[fam]
    h_base += np.einsum("ldr,nr->nld", B, rng.normal(size=(n, nuisance_rank)))
    h_base += 0.3 * rng.normal(size=(n, L, D))

    h_edit = np.empty((n, E, L, D))
    for e in range(E):
        if e < n_t:
            resp = np.einsum("ldr,nr->nld", B, rng.normal(size=(n, nuisance_rank)))
        else:
            gain = np.where(y == 1, 0.2, 1.0)[:, None, None]
            resp = gain * np.einsum("ldc,nc->nld", M, c)
        resp += 0.3 * rng.normal(size=(n, L, D))
        resp += failure_scale * y[:, None, None] * f[None]
        h_edit[:, e] = h_base + resp

    logits = rng.normal(size=(n, n_slots))
    logits[np.arange(n), (c[:, 0] * 3).astype(int)] += 3.0
    a_base = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    a_edit = np.repeat(a_base[:, None], E, axis=1) + 0.2 * rng.normal(size=(n, E, n_slots))
    a_edit[:, n_t:, 0] += np.where(y == 1, 0.0, 1.0)[:, None]  # normal actors react to U edits
    a_edit -= np.log(np.exp(a_edit).sum(2, keepdims=True))

    bundle = ResponseBundle(
        root_ids=np.array([f"{id_prefix}{seed}-{i}" for i in range(n)]),
        layers=tuple(layers),
        slot_names=tuple(f"S{k}" for k in range(n_slots)),
        edit_names=tuple([f"t{i}" for i in range(n_t)] + [f"u{i}" for i in range(n_u)]),
        edit_kinds=tuple(["T"] * n_t + ["U"] * n_u),
        h_base=h_base.astype(np.float32),
        h_edit=h_edit.astype(np.float32),
        a_base=a_base.astype(np.float32),
        a_edit=a_edit.astype(np.float32),
        edit_valid=np.ones((n, E), bool),
        context=c,
        context_names=tuple(f"c{k}" for k in range(n_context)),
        context_group=np.array([f"g{int(c[i, 0])}{int(c[i, 1])}" for i in range(n)]),
        env_label=np.array([f"fam{k}" for k in fam]),
        chosen_slot=np.argmax(a_base, axis=1),
        cost={"base": 1.0, "edits": float(E)},
        # Edited inputs keep the base context here (deltas zero); U edits are
        # declared decision-changing, as the U contract requires.
        extras={
            "context_edit": np.repeat(c[:, None, :], E, axis=1).astype(np.float32),
            "edit_decision_changed": np.repeat((np.arange(E) >= n_t)[None], n, axis=0),
        },
    )
    return bundle, y
