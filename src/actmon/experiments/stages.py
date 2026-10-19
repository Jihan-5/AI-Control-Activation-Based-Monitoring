r"""Cached pipeline stages shared by the A-experiments.

``work_dir`` (from the config) holds everything expensive:

* ``actor/``: M1 and M2 weights (or adapters), ``actor_meta.json`` (sites, hook
  check, training logs, tokenizer fingerprint, resolved revisions)
* ``data/``: per split, ``features_<split>.npz``, ``labels_<split>.csv.gz``,
  ``pool_<split>.csv.gz`` and ``scenarios_<split>.pkl``

Each stage is skipped when its outputs exist, so A2-A5 reuse one actor and one
collection. Every experiment still writes its own manifest into its own output
directory, which records the work-dir config hash it used.

**M1 and M2.** M1 is the competence model, trained on planner-verified data. M2,
the monitored checkpoint, is M1 plus a LoRA adapter trained on organism data and
then merged. *Normal and violating roots both come from M2* (Technical
Specification: "otherwise the detector might merely recognize adapter-induced
weight changes"). M1 is used only to label mechanisms.

For the offline stand-in, M1 is trained from scratch in two phases: a short
curriculum on simple scenarios (two pools, no queued tickets or logs), then
the full distribution, with a small auxiliary loss on prompt tokens. Without
it, assistant-only training stalled at the marginal action distribution
(lab notebook M-1). The organism adapter M2 is trained with assistant-only loss.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from actmon.utils.manifest import config_hash, package_versions
from actmon.utils.seeding import SeedBank

from actmon.actor.models import (ActorBundle, add_lora, build_small_actor, freeze, lora_parameters, merge_lora,
                                 site_blocks)
from actmon.actor.organism import OrganismConfig, competence_examples, organism_examples
from actmon.actor.prompts import encode_root
from actmon.actor.train import SFTConfig, sft
from actmon.data.collect import SplitSpec, collect_split
from actmon.data.features import FeatureSet, RootLabels
from actmon.sim.render import make_families
from actmon.sim.sampler import SamplerConfig


def families(cfg: dict):
    fc = cfg["actor"]["families"]
    fams = make_families(int(fc["n"]), int(fc.get("seed", 0)))
    rng = np.random.default_rng(np.random.SeedSequence([int(fc.get("seed", 0)), 4242]))
    held = set(rng.choice(len(fams), size=int(fc["heldout"]), replace=False).tolist())
    return fams, [f for i, f in enumerate(fams) if i not in held], [f for i, f in enumerate(fams) if i in held]


def _small_tokenizer():
    from actmon.actor.tokenizer import WordTokenizer

    return WordTokenizer()


def _bundle(cfg, model, tok, name) -> ActorBundle:
    from actmon.actor import tokenizer as tk

    kind = cfg["actor"]["kind"]
    if kind == "small":
        return ActorBundle(model, tok, tk.SYSTEM_PROMPT_SMALL, site_blocks(model.config.num_hidden_layers), name,
                           show_headroom=bool(cfg["actor"].get("show_headroom", True)),
                           contrast=dict(tk.CONTRAST_SUFFIX))
    return ActorBundle(model, tok, tk.SYSTEM_PROMPT_FULL, list(cfg["actor"].get("sites", [6, 13, 20])), name,
                       show_headroom=bool(cfg["actor"].get("show_headroom", False)),
                       contrast=dict(tk.CONTRAST_SUFFIX_FULL))


def _fresh_model(cfg, tok):
    a = cfg["actor"]
    if a["kind"] == "small":
        s = a["small"]
        return build_small_actor(tok.vocab_size, hidden=s["hidden"], layers=s["layers"], heads=s["heads"],
                                 kv_heads=s["kv_heads"], intermediate=s["intermediate"], seed=int(cfg["seed"]) % 2**31,
                                 pad_id=tok.pad_id)
    from actmon.actor.models import load_qwen

    q = a["qwen"]
    model, tok2 = load_qwen(q["name"], q.get("revision"), q.get("dtype", "bfloat16"), q.get("device", "cuda"),
                            q.get("cache_dir"))
    return model


def _eval_actor(bundle, scen_specs):
    from actmon.actor.decode import run_actor
    from actmon.sim.oracle import check, planner_action

    enc = [encode_root(bundle, s, sp) for s, sp in scen_specs]
    out = run_actor(bundle.model, bundle.tok.pad_id, [e[0] for e in enc], [e[1] for e in enc], [], batch_size=128)
    acc = float(np.mean([a.verb == planner_action(s)[0].verb for a, (s, _) in zip(out.actions, scen_specs)]))
    val = float(np.mean([check(s, a).valid for a, (s, _) in zip(out.actions, scen_specs)]))
    return {"verb_acc": acc, "valid": val}


def _eval_set(cfg, bundle, fams, n=600, tag="eval"):
    from actmon.sim.render import RenderSpec
    from actmon.sim.sampler import sample_scenario

    bank = SeedBank(int(cfg["seed"])).child("actor_eval", tag)
    out = []
    for i in range(n):
        rng = bank.child(i).rng()
        f = fams[int(rng.integers(len(fams)))]
        out.append((sample_scenario(rng, f, f"{tag}{i}", SamplerConfig.from_dict(cfg.get("sampler"))),
                    RenderSpec(f, show_headroom=bundle.show_headroom)))
    return out


def ensure_actor(cfg: dict, work: Path, log=print):
    """Build or load M1 (competence) and M2 (monitored organism). Returns (m2, m1, meta)."""
    work = Path(work)
    adir = work / "actor"
    adir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(cfg.get("threads", 4)))
    a = cfg["actor"]
    if a["kind"] == "qwen":
        return _ensure_qwen(cfg, adir, log)
    tok = _small_tokenizer()
    _, id_fams, held_fams = families(cfg)
    sampler = SamplerConfig.from_dict(cfg.get("sampler"))
    bank = SeedBank(int(cfg["seed"]))
    meta_path = adir / "actor_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("actor_config_hash") not in (None, config_hash(a)):
        raise RuntimeError(f"{adir} was built with a different actor config; use another work_dir")

    m1 = _fresh_model(cfg, tok)
    b1 = _bundle(cfg, m1, tok, "M1")
    if (adir / "m1.pt").exists():
        m1.load_state_dict(torch.load(adir / "m1.pt", map_location="cpu"))
    else:
        comp = a["competence"]
        evalset = _eval_set(cfg, b1, id_fams)
        cb = lambda step: _eval_actor(b1, evalset)  # noqa: E731
        t0 = time.time()
        log("[actor] M1 phase 1 (curriculum)")
        ex1, _ = competence_examples(b1, id_fams, int(comp["curriculum_examples"]), bank.child("train", "comp1").rng(),
                                     SamplerConfig.from_dict({**(cfg.get("sampler") or {}),
                                                              **comp["curriculum_sampler"]}), augment_prob=0.0)
        h1 = sft(m1, ex1, tok.pad_id, SFTConfig.from_dict(comp["sft_phase1"]), callback=cb, log=log)
        log("[actor] M1 phase 2 (full distribution)")
        ex2, _ = competence_examples(b1, id_fams, int(comp["examples"]), bank.child("train", "comp2").rng(), sampler,
                                     augment_prob=float(comp.get("augment_prob", 0.5)))
        h2 = sft(m1, ex2, tok.pad_id, SFTConfig.from_dict(comp["sft_phase2"]), callback=cb, log=log)
        torch.save(m1.state_dict(), adir / "m1.pt")
        meta["m1"] = {"phase1": h1, "phase2": h2, "train_sec": time.time() - t0}
    freeze(m1)

    m2 = _fresh_model(cfg, tok)
    m2.load_state_dict(m1.state_dict())
    b2 = _bundle(cfg, m2, tok, "M2")
    if (adir / "m2.pt").exists():
        m2.load_state_dict(torch.load(adir / "m2.pt", map_location="cpu"))
    else:
        org = a["organism"]
        lora = org["lora"]
        wrapped = add_lora(m2, tuple(lora["targets"]), int(lora["r"]), float(lora["alpha"]), float(lora["dropout"]),
                           seed=int(cfg["seed"]) % 2**31)
        ex, emeta = organism_examples(b2, id_fams, int(org["examples"]), bank.child("train", "organism").rng(),
                                      sampler, OrganismConfig.from_dict(org.get("mix")),
                                      augment_prob=float(org.get("augment_prob", 0.5)))
        t0 = time.time()
        log(f"[actor] M2 organism adapter on {len(wrapped)} modules")
        h = sft(m2, ex, tok.pad_id, SFTConfig.from_dict(org["sft"]), params=lora_parameters(m2), log=log)
        merge_lora(m2)
        torch.save(m2.state_dict(), adir / "m2.pt")
        meta["m2"] = {"history": h, "train_sec": time.time() - t0, "wrapped": wrapped,
                      "violate_target_frac": float(np.mean([m["violate_target"] for m in emeta]))}
    freeze(m2)
    return _finish(cfg, adir, meta, b2, b1, id_fams, held_fams, tok, log)


def _finish(cfg, adir, meta, b2, b1, id_fams, held_fams, tok, log):
    """Hook check, competence evaluation and the actor manifest (once per work dir)."""
    m2 = b2.model
    meta_path = adir / "actor_meta.json"
    if "hook_check" not in meta:
        from actmon.actor.hooks import verify_sites

        sc = _eval_set(cfg, b2, id_fams, n=16, tag="hook")
        prompts = [encode_root(b2, s, sp)[0] for s, sp in sc]
        meta["hook_check"] = {"max_abs_diff": verify_sites(m2, prompts, b2.sites, tok.pad_id), "sites": b2.sites,
                              "tensor": "output of model.model.layers[i] at the last (left-padded) prompt token"}
        for tag, b, fams in (("m1_id", b1, id_fams), ("m2_id", b2, id_fams), ("m2_heldout", b2, held_fams)):
            n_eval = int(cfg["actor"].get("eval_n", 1000))
            meta[f"competence_{tag}"] = _eval_actor(b, _eval_set(cfg, b, fams, n=n_eval, tag=tag))
        meta["tokenizer"] = tok.state()
        meta["actor_config_hash"] = config_hash(cfg["actor"])
        meta["revision_requested"] = cfg["actor"].get("qwen", {}).get("revision") \
            if cfg["actor"]["kind"] == "qwen" else None
        meta["revision_resolved"] = getattr(m2.config, "_commit_hash", None)
        meta["versions"] = package_versions()
        meta["n_params"] = int(sum(p.numel() for p in m2.parameters()))
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log(f"[actor] sites {b2.sites}; hook check {meta['hook_check']['max_abs_diff']:.2e}; "
        f"M2 in-distribution {meta.get('competence_m2_id')}")
    return b2, b1, meta


def _adapter_state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def _qwen_adapter(cfg, model, b, fams, examples_fn, lora, sft_cfg, path, log):
    """Train (or load) one LoRA adapter on ``model`` and merge it. Returns the training log."""
    wrapped = add_lora(model, tuple(lora["targets"]), int(lora["r"]), float(lora["alpha"]), float(lora["dropout"]),
                       seed=int(cfg["seed"]) % 2**31)
    hist = None
    if path.exists():
        missing = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        if any("lora_" in k for k in missing.missing_keys):
            raise RuntimeError(f"adapter {path} does not match the wrapped modules {wrapped}")
    else:
        ex = examples_fn(b)
        hist = sft(model, ex, b.tok.pad_id, SFTConfig.from_dict(sft_cfg), params=lora_parameters(model), log=log)
        torch.save(_adapter_state(model), path)
    merge_lora(model)
    return {"wrapped": wrapped, "history": hist}


def _ensure_qwen(cfg, adir: Path, log):
    """Declared actor: Qwen2.5-1.5B-Instruct + competence adapter (M1) + organism adapter (M2), both merged.

    Needs a GPU and the model files (Hugging Face access or a local cache).
    Adapters are saved, never full weights, so a work dir stays small. The
    pinned revision and the tokenizer fingerprint go to ``actor_meta.json``.
    """
    from actmon.actor.models import load_qwen

    a = cfg["actor"]
    q = a["qwen"]
    torch.manual_seed(int(cfg["seed"]) % 2**31)
    _, id_fams, held_fams = families(cfg)
    sampler = SamplerConfig.from_dict(cfg.get("sampler"))
    bank = SeedBank(int(cfg["seed"]))
    meta_path = adir / "actor_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("actor_config_hash") not in (None, config_hash(a)):
        raise RuntimeError(f"{adir} was built with a different actor config; use another work_dir")

    def load():
        return load_qwen(q["name"], q.get("revision"), q.get("dtype", "bfloat16"), q.get("device", "cuda"),
                         q.get("cache_dir"))

    comp, org = a["competence"], a["organism"]
    m1, tok = load()
    b1 = _bundle(cfg, m1, tok, "M1")
    meta["m1"] = _qwen_adapter(
        cfg, m1, b1, id_fams,
        lambda b: competence_examples(b, id_fams, int(comp["examples"]), bank.child("train", "comp2").rng(), sampler,
                                      augment_prob=float(comp.get("augment_prob", 0.5)))[0],
        comp["lora"], comp["sft"], adir / "m1_adapter.pt", log) | (meta.get("m1") or {})
    freeze(m1)
    m2, _ = load()
    b2 = _bundle(cfg, m2, tok, "M2")
    add_lora(m2, tuple(comp["lora"]["targets"]), int(comp["lora"]["r"]), float(comp["lora"]["alpha"]),
             float(comp["lora"]["dropout"]), seed=int(cfg["seed"]) % 2**31)
    m2.load_state_dict(torch.load(adir / "m1_adapter.pt", map_location="cpu"), strict=False)
    merge_lora(m2)
    meta["m2"] = _qwen_adapter(
        cfg, m2, b2, id_fams,
        lambda b: organism_examples(b, id_fams, int(org["examples"]), bank.child("train", "organism").rng(), sampler,
                                    OrganismConfig.from_dict(org.get("mix")),
                                    augment_prob=float(org.get("augment_prob", 0.5)))[0],
        org["lora"], org["sft"], adir / "m2_adapter.pt", log) | (meta.get("m2") or {})
    freeze(m2)
    return _finish(cfg, adir, meta, b2, b1, id_fams, held_fams, tok, log)


def ensure_data(cfg: dict, work: Path, m2, m1, log=print) -> dict[str, tuple[FeatureSet, RootLabels]]:
    work = Path(work)
    ddir = work / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    _, id_fams, held_fams = families(cfg)
    sampler = SamplerConfig.from_dict(cfg.get("sampler"))
    bank = SeedBank(int(cfg["seed"])).child("collect")
    col = cfg["collection"]
    out = {}
    for name, spec in col["splits"].items():
        sp = SplitSpec.from_dict(name, spec)
        fams = held_fams if sp.families == "heldout" else id_fams
        fs, lab = collect_split(sp, m2, m1, fams, sampler, bank, ddir, col.get("opts", {}), log=log)
        out[name] = (fs, lab)
    return out


def load_scenarios(work: Path, split: str):
    with open(Path(work) / "data" / f"scenarios_{split}.pkl", "rb") as f:
        return pickle.load(f)
