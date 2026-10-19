"""Hugging Face causal-LM backend with residual-stream hooks.

Extraction protocol (research plan p.6)
---------------------------------------
* Model in eval mode, ``torch.inference_mode``.
* Forward hooks on decoder blocks capture the residual stream *after*
  zero-based block ``l``. For blocks ``l < n_layers - 1`` this equals
  ``output_hidden_states[l + 1]``; the final entry of ``hidden_states`` is
  post-final-norm in HF, so it is never used as "block L-1 output"
  (verified with transformers 5.20.1; ``verify_hooks`` re-checks at runtime).
* Activations are gathered at the final non-padding token of the chat-templated
  prompt with ``add_generation_prompt=True``: the position immediately before
  the actor's first action token.
* RIGHT padding everywhere except batched generation. With right padding and
  default ``position_ids`` every real token keeps its unpadded position, so the
  gathered state is identical to an unbatched forward.

Candidate scoring
-----------------
Prompt and candidate are tokenized *separately* and their ids concatenated.
This (a) keeps the prompt tokens identical across all candidates, so the
prompt-final hidden state is shared and can be read from any concatenation by
causality, and (b) avoids BPE merges across the prompt/candidate boundary that
would make different candidates score different token boundaries. Each
candidate is terminated with the end-of-turn token so *complete* actions are
scored consistently (plan p.6). Probabilities normalised over the bounded set
are candidate-set probabilities, not the model's unrestricted distribution.

Only the needed positions go through the LM head (``B x T x V`` logits for a
151k-token vocabulary would dominate memory).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager

import numpy as np

from irmon.envs.base import ChatPrompt

from .backend import ActorBackend, resolve_layers

_BLOCK_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers", "layers")
_MLP_OUT_NAMES = ("down_proj", "dense_4h_to_h", "mlp.c_proj", "fc2")


def _torch():
    import torch

    return torch


def _get_path(obj, path: str):
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def resolve_blocks(model):
    """Return the ``nn.ModuleList`` of decoder blocks, unwrapping PEFT."""
    torch = _torch()
    roots = [model]
    if hasattr(model, "get_base_model"):
        roots.insert(0, model.get_base_model())
    for root in list(roots):
        inner = getattr(root, "model", None)
        if inner is not None:
            roots.append(inner)
    for root in roots:
        for path in _BLOCK_PATHS:
            obj = _get_path(root, path)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return obj
    raise ValueError(f"could not locate decoder blocks in {type(model).__name__}")


def _base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _decoder(model):
    """Inner decoder (no LM head) so hidden-only passes skip vocab logits."""
    base = _base_model(model)
    if hasattr(base, "get_decoder"):
        try:
            dec = base.get_decoder()
            if dec is not None:
                return dec
        except (AttributeError, NotImplementedError):
            pass
    prefix = getattr(base, "base_model_prefix", "model")
    return getattr(base, prefix, base)


def _block_output(out):
    return out[0] if isinstance(out, tuple) else out


class HFBackend(ActorBackend):
    name = "hf"

    def __init__(
        self,
        model=None,
        tokenizer=None,
        *,
        repo_id: str | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
        attn_implementation: str | None = "sdpa",
        quantization: str = "none",
        adapter_path: str | None = None,
        trust_remote_code: bool = False,
        layers: Sequence[int] = (6, 13, 20),
        batch_size: int = 8,
        max_length: int = 1024,
        append_eos: bool = True,
        debug_checks: bool = False,
    ):
        super().__init__()
        torch = _torch()
        self.repo_id = repo_id
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.requested_dtype = dtype
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.adapter_path = adapter_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.append_eos = append_eos
        self.debug_checks = debug_checks
        self.dtype_note = ""
        self.n_truncated = 0
        self._prompt_cache: dict[str, list[int]] = {}

        if model is not None:
            self.requested_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        else:
            if repo_id is None:
                raise ValueError("pass either (model, tokenizer) or repo_id")
            model, tokenizer = self._load(repo_id, revision, tokenizer_revision, dtype, device,
                                          attn_implementation, quantization, trust_remote_code)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.model.parameters()).device
        self.blocks = resolve_blocks(self.model)
        self.n_layers = len(self.blocks)
        cfg = _base_model(self.model).config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        self.d_model = int(text_cfg.hidden_size)
        self.layers = resolve_layers(layers, self.n_layers)
        self.dtype = next(self.model.parameters()).dtype
        self._gather_idx = None
        self._captures: dict[int, object] = {}
        del torch

    # ----------------------------------------------------------------- load
    def _load(self, repo_id, revision, tokenizer_revision, dtype, device, attn, quantization, trust):
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype)
        if dtype == "bfloat16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float32
            self.dtype_note = "bf16 unsupported on this GPU; fell back to float32 (validate geometry)"
        kwargs = {"revision": revision, "trust_remote_code": trust}
        if attn:
            kwargs["attn_implementation"] = attn
        if quantization in ("8bit", "4bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=quantization == "8bit", load_in_4bit=quantization == "4bit"
            )
            kwargs["device_map"] = "auto"
        try:  # transformers >= 5 uses ``dtype``; older versions ``torch_dtype``.
            model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch_dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch_dtype, **kwargs)
        if quantization == "none":
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=tokenizer_revision or revision, trust_remote_code=trust
        )
        return model, tokenizer

    @classmethod
    def from_config(cls, cfg, **overrides) -> HFBackend:
        """Build from ``irmon.config.ModelConfig``."""
        kw = dict(
            repo_id=cfg.repo_id, revision=cfg.revision, tokenizer_revision=cfg.tokenizer_revision,
            dtype=cfg.dtype, device=cfg.device, attn_implementation=cfg.attn_implementation,
            quantization=cfg.quantization, adapter_path=cfg.adapter_path,
            trust_remote_code=cfg.trust_remote_code, batch_size=cfg.batch_size,
            max_length=cfg.max_length,
        )
        kw.update(overrides)
        backend = cls(**kw)
        backend.layers = resolve_layers(cfg.layers, backend.n_layers, cfg.reference_n_layers, cfg.layer_mode)
        return backend

    # ----------------------------------------------------------- tokenizing
    def prompt_text(self, prompt: ChatPrompt) -> str:
        return self.tokenizer.apply_chat_template(
            prompt.as_messages(), tokenize=False, add_generation_prompt=True
        )

    def prompt_ids(self, prompt: ChatPrompt, reserve: int = 0) -> list[int]:
        """Chat-templated prompt ids, left-truncated to ``max_length - reserve``."""
        key = prompt.fingerprint()
        ids = self._prompt_cache.get(key)
        if ids is None:
            ids = self.tokenizer(self.prompt_text(prompt), add_special_tokens=False)["input_ids"]
            if len(self._prompt_cache) > 50000:
                self._prompt_cache.clear()
            self._prompt_cache[key] = ids
        limit = self.max_length - reserve
        if len(ids) > limit:
            self.n_truncated += 1
            ids = ids[-limit:]
        return list(ids)

    def candidate_ids(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if self.append_eos and self.tokenizer.eos_token_id is not None:
            ids = ids + [self.tokenizer.eos_token_id]
        return ids

    def _pad_right(self, seqs: list[list[int]]):
        torch = _torch()
        T = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    # ----------------------------------------------------------------- hooks
    @contextmanager
    def _hooks(self, layers: Sequence[int], full: bool = False):
        """Capture block outputs. ``full=False`` gathers only ``_gather_idx``."""
        torch = _torch()
        handles = []
        self._captures = {}

        def make(li: int):
            def hook(_mod, _inp, out):
                h = _block_output(out)
                if full:
                    self._captures[li] = h.detach().float().cpu()
                else:
                    rows = torch.arange(h.shape[0], device=h.device)
                    self._captures[li] = h[rows, self._gather_idx.to(h.device)].detach().float().cpu()

            return hook

        try:
            for li in layers:
                handles.append(self.blocks[li].register_forward_hook(make(li)))
            yield self._captures
        finally:
            for h in handles:
                h.remove()

    def _timed(self, key: str, t0: float) -> None:
        self.counter.wall_seconds[key] = self.counter.wall_seconds.get(key, 0.0) + time.perf_counter() - t0

    # -------------------------------------------------------------- hidden
    def hidden_at_final_token(self, prompts: Sequence[ChatPrompt], layers: Sequence[int]) -> np.ndarray:
        torch = _torch()
        t0 = time.perf_counter()
        out = np.zeros((len(prompts), len(layers), self.d_model), dtype=np.float32)
        decoder = _decoder(self.model)
        for s in range(0, len(prompts), self.batch_size):
            batch = [self.prompt_ids(p) for p in prompts[s : s + self.batch_size]]
            ids, mask = self._pad_right(batch)
            self._gather_idx = mask.sum(1) - 1
            with torch.inference_mode(), self._hooks(layers) as caps:
                decoder(input_ids=ids, attention_mask=mask, use_cache=False)
            for j, li in enumerate(layers):
                out[s : s + len(batch), j] = caps[li].numpy()
            self.counter.prompt_forwards += len(batch)
            self.counter.prompt_tokens += int(mask.sum())
        self._timed("hidden", t0)
        return out

    # ------------------------------------------------------ candidate pass
    def _score_pairs(
        self,
        pairs: list[tuple[list[int], list[int]]],
        layers: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Total log-prob of each (prompt_ids, cand_ids) continuation, plus the
        hidden state at the prompt-final position if ``layers`` is given."""
        torch = _torch()
        decoder = _decoder(self.model)
        head = _base_model(self.model).get_output_embeddings()
        scores = np.zeros(len(pairs), dtype=np.float64)
        hidden = None if layers is None else np.zeros((len(pairs), len(layers), self.d_model), np.float32)
        bs = max(1, self.batch_size)
        for s in range(0, len(pairs), bs):
            chunk = pairs[s : s + bs]
            seqs = [p + c for p, c in chunk]
            ids, mask = self._pad_right(seqs)
            plen = torch.tensor([len(p) for p, _ in chunk], device=self.device)
            self._gather_idx = plen - 1
            ctx = self._hooks(layers) if layers is not None else _null()
            with torch.inference_mode(), ctx as caps:
                h = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            # Positions predicting candidate tokens: plen-1 .. plen+len(c)-2.
            rows, cols, tgt = [], [], []
            for b, (p, c) in enumerate(chunk):
                for t, tok in enumerate(c):
                    rows.append(b)
                    cols.append(len(p) - 1 + t)
                    tgt.append(tok)
            rows_t = torch.tensor(rows, device=self.device)
            sel = h[rows_t, torch.tensor(cols, device=self.device)]
            with torch.inference_mode():
                logp = torch.log_softmax(head(sel).float(), dim=-1)
            tok_lp = logp[torch.arange(len(tgt), device=self.device), torch.tensor(tgt, device=self.device)]
            per = torch.zeros(len(chunk), dtype=torch.float64, device=self.device)
            per.index_add_(0, rows_t, tok_lp.double())
            scores[s : s + len(chunk)] = per.cpu().numpy()
            if layers is not None:
                for j, li in enumerate(layers):
                    hidden[s : s + len(chunk), j] = caps[li].numpy()
            self.counter.candidate_forwards += len(chunk)
            self.counter.prompt_tokens += int(mask.sum())
        return scores, hidden

    def _build_pairs(self, prompts, candidates):
        pairs, shape = [], []
        for p, cands in zip(prompts, candidates):
            cids = [self.candidate_ids(c) for c in cands]
            reserve = max(len(c) for c in cids)
            pids = self.prompt_ids(p, reserve=reserve)
            pairs.extend((pids, c) for c in cids)
            shape.append(len(cids))
        if len(set(shape)) != 1:
            raise ValueError("every prompt must have the same number of candidates (canonical slots)")
        return pairs, shape[0]

    def score_candidates(self, prompts: Sequence[ChatPrompt], candidates: Sequence[Sequence[str]]) -> np.ndarray:
        t0 = time.perf_counter()
        pairs, K = self._build_pairs(prompts, candidates)
        scores, _ = self._score_pairs(pairs)
        self._timed("candidates", t0)
        return scores.reshape(len(prompts), K).astype(np.float32)

    def hidden_and_scores(self, prompts, layers, candidates):
        """Fused: hidden states come from the candidate passes (shared prefix)."""
        t0 = time.perf_counter()
        pairs, K = self._build_pairs(prompts, candidates)
        scores, hidden = self._score_pairs(pairs, layers=layers)
        hidden = hidden.reshape(len(prompts), K, len(layers), self.d_model)
        if self.debug_checks:
            spread = np.abs(hidden - hidden[:, :1]).max()
            tol = 1e-3 * max(1.0, float(np.abs(hidden).max()))
            if spread > tol:
                raise AssertionError(f"prompt-final hidden differs across candidates by {spread:.3g}")
        self._timed("hidden_and_scores", t0)
        return hidden[:, 0].copy(), scores.reshape(len(prompts), K).astype(np.float32)

    # ------------------------------------------------------------ generate
    def generate(self, prompts: Sequence[ChatPrompt], max_new_tokens: int = 32) -> list[str]:
        torch = _torch()
        t0 = time.perf_counter()
        outs: list[str] = []
        pad = self.tokenizer.pad_token_id
        for s in range(0, len(prompts), self.batch_size):
            batch = [self.prompt_ids(p, reserve=max_new_tokens) for p in prompts[s : s + self.batch_size]]
            T = max(len(b) for b in batch)
            ids = torch.full((len(batch), T), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), T), dtype=torch.long)
            for i, b in enumerate(batch):  # LEFT padding for batched generation
                ids[i, T - len(b) :] = torch.tensor(b)
                mask[i, T - len(b) :] = 1
            with torch.inference_mode():
                gen = self.model.generate(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                    max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new = gen[:, T:]
            outs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
            self.counter.prompt_forwards += len(batch)
            self.counter.generated_tokens += int((new != pad).sum())
        self._timed("generate", t0)
        return outs

    # --------------------------------------------------------- loss traces
    def default_trace_params(self) -> list[tuple[str, object]]:
        """MLP output projections of the monitored blocks."""
        out = []
        for li in self.layers:
            for name, p in self.blocks[li].named_parameters():
                if any(k in name for k in _MLP_OUT_NAMES) and name.endswith("weight"):
                    out.append((f"blocks.{li}.{name}", p))
        if not out:
            raise ValueError("no MLP output projection found in monitored blocks; pass params explicitly")
        return out

    @staticmethod
    def _checksum(params) -> str:
        h = hashlib.sha256()
        for _, p in params:
            h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()

    def loss_trace(
        self,
        prompts: Sequence[ChatPrompt],
        targets: Sequence[str],
        n_samples: int,
        sigma: float,
        seed: int,
        params: list[tuple[str, object]] | None = None,
    ) -> np.ndarray:
        """NLL of each target under ``n_samples`` Gaussian parameter perturbations.

        Noise for parameter ``p`` is ``sigma * std(p) * N(0, I)``, drawn from a
        generator seeded by ``(seed, sample)`` so traces are reproducible and
        shared across roots (trusted reference traces need the same draws).
        Original weights are copied and restored exactly; a checksum verifies it.
        """
        torch = _torch()
        t0 = time.perf_counter()
        params = params or self.default_trace_params()
        before = self._checksum(params)
        saved = [p.detach().clone() for _, p in params]
        pairs = []
        for p, t in zip(prompts, targets):
            cids = self.candidate_ids(t)
            pairs.append((self.prompt_ids(p, reserve=len(cids)), cids))
        out = np.zeros((len(prompts), n_samples), dtype=np.float32)
        try:
            for s in range(n_samples):
                g = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + s)
                with torch.no_grad():
                    for (_, p), orig in zip(params, saved):
                        noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                        p.copy_((orig.float() + sigma * orig.float().std() * noise.to(p.device)).to(p.dtype))
                scores, _ = self._score_pairs(pairs)
                out[:, s] = -scores
                self.counter.perturbed_forwards += len(pairs)
                self.counter.candidate_forwards -= len(pairs)  # counted as perturbed instead
        finally:
            with torch.no_grad():
                for (_, p), orig in zip(params, saved):
                    p.copy_(orig)
        if self._checksum(params) != before:  # pragma: no cover - defensive
            raise RuntimeError("parameter restoration failed after loss_trace")
        self._timed("loss_trace", t0)
        return out

    # ------------------------------------------------------- verification
    def verify_hooks(self, prompts: Sequence[ChatPrompt], layers: Sequence[int] | None = None) -> float:
        """Max |hook capture - output_hidden_states[l+1]| over real tokens.

        The last block is skipped: HF applies the final norm before appending
        the last ``hidden_states`` entry.
        """
        torch = _torch()
        layers = [li for li in (layers or self.layers) if li < self.n_layers - 1]
        batch = [self.prompt_ids(p) for p in prompts]
        ids, mask = self._pad_right(batch)
        with torch.inference_mode(), self._hooks(layers, full=True) as caps:
            ref = _decoder(self.model)(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        m = mask.bool().cpu()
        worst = 0.0
        for li in layers:
            diff = (caps[li] - ref.hidden_states[li + 1].float().cpu()).abs()[m]
            worst = max(worst, float(diff.max()))
        return worst

    def final_token_index(self, prompts: Sequence[ChatPrompt]) -> list[int]:
        return [len(self.prompt_ids(p)) - 1 for p in prompts]

    # ------------------------------------------------------------ manifest
    def manifest(self) -> dict:
        import torch
        import transformers

        base = _base_model(self.model)
        cfg = base.config
        template = getattr(self.tokenizer, "chat_template", None) or ""
        return {
            "backend": self.name,
            "repo_id": self.repo_id,
            "requested_revision": self.revision,
            "resolved_commit": getattr(cfg, "_commit_hash", None),
            "tokenizer": getattr(self.tokenizer, "name_or_path", None),
            "tokenizer_revision": self.tokenizer_revision or self.revision,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "requested_dtype": self.requested_dtype,
            "dtype": str(self.dtype),
            "dtype_note": self.dtype_note,
            "attn_implementation": getattr(cfg, "_attn_implementation", self.attn_implementation),
            "quantization": self.quantization,
            "adapter_path": self.adapter_path,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "layers": list(self.layers),
            "max_length": self.max_length,
            "n_truncated_prompts": self.n_truncated,
            "append_eos_to_candidates": self.append_eos,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(self.device),
            "counter": self.counter.as_dict(),
        }


@contextmanager
def _null():
    yield {}


def param_filter_names(model, predicate: Callable[[str], bool]) -> list[tuple[str, object]]:
    """Helper to pick an explicit parameter subset for ``loss_trace``."""
    return [(n, p) for n, p in model.named_parameters() if predicate(n)]
