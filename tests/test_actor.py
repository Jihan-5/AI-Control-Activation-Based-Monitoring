"""Actor plumbing on a randomly initialised small Qwen2 (no download, no claims)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from actmon.actor.decode import run_actor  # noqa: E402
from actmon.actor.hooks import ResidualRecorder, left_pad, verify_sites  # noqa: E402
from actmon.actor.models import (ActorBundle, add_lora, build_small_actor, merge_lora, site_blocks)  # noqa: E402
from actmon.actor.prompts import encode_root  # noqa: E402
from actmon.actor.tokenizer import SYSTEM_PROMPT_SMALL, WordTokenizer  # noqa: E402
from actmon.actor.train import masked_batch  # noqa: E402
from actmon.attack.obfuscation import soft_prompt, with_soft_slots  # noqa: E402
from actmon.sim import transforms as T  # noqa: E402
from actmon.sim.render import RenderSpec, make_families  # noqa: E402
from actmon.sim.sampler import sample_scenario  # noqa: E402

pytestmark = pytest.mark.lm
FAMS = make_families(32, 0)


@pytest.fixture(scope="module")
def bundle():
    tok = WordTokenizer()
    model = build_small_actor(tok.vocab_size, hidden=32, layers=6, heads=2, kv_heads=1, intermediate=64, seed=0,
                              pad_id=tok.pad_id).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ActorBundle(model, tok, SYSTEM_PROMPT_SMALL, site_blocks(6), "test")


@pytest.fixture(scope="module")
def roots():
    rng = np.random.default_rng(0)
    return [(s := sample_scenario(rng, FAMS[i % 32], f"r{i}"), RenderSpec(FAMS[i % 32])) for i in range(12)]


def test_site_rule_reproduces_declared_qwen_blocks():
    assert site_blocks(28) == [6, 13, 20]
    assert site_blocks(6) == [1, 2, 4]


def test_closed_vocabulary_covers_every_rendering(bundle):
    rng = np.random.default_rng(1)
    for i in range(400):
        f = FAMS[i % 32]
        s, sp = sample_scenario(rng, f, f"v{i}"), RenderSpec(f)
        for name in (None,) + T.MAIN_SUITE:
            s2, sp2 = (s, sp) if name is None else T.apply(name, s, sp, rng)
            ids, cands = encode_root(bundle, s2, sp2)
            assert bundle.tok.unk_id not in ids
            assert all(bundle.tok.unk_id not in c for _, c in cands)


def test_hook_matches_hidden_states(bundle, roots):
    prompts = [encode_root(bundle, s, sp)[0] for s, sp in roots]
    assert verify_sites(bundle.model, prompts, bundle.sites, bundle.tok.pad_id) < 1e-5


def test_left_padding_does_not_change_activations(bundle, roots):
    prompts = [encode_root(bundle, s, sp)[0] for s, sp in roots[:4]]
    rec = ResidualRecorder(bundle.model, bundle.sites)
    with rec, torch.no_grad():
        ids, mask, pos = left_pad(prompts, bundle.tok.pad_id)
        bundle.model(input_ids=ids, attention_mask=mask, position_ids=pos)
    batched = rec.pop()
    for i, p in enumerate(prompts):
        with rec, torch.no_grad():
            bundle.model(input_ids=torch.tensor([p]))
        single = rec.pop()
        for s in bundle.sites:
            assert torch.allclose(batched[s][i], single[s][0], atol=1e-4)


def test_constrained_decoding_is_exact_argmax(bundle, roots):
    enc = [encode_root(bundle, s, sp) for s, sp in roots]
    out = run_actor(bundle.model, bundle.tok.pad_id, [e[0] for e in enc], [e[1] for e in enc], bundle.sites,
                    batch_size=5)
    for (prompt, cands), chosen, feats in zip(enc, out.actions, out.feats):
        best, best_lp = None, -np.inf
        for a, toks in cands:  # brute force, no cache
            ids = torch.tensor([prompt + toks])
            with torch.no_grad():
                lp = torch.log_softmax(bundle.model(input_ids=ids).logits[0].float(), -1)
            s = sum(float(lp[len(prompt) - 1 + j, t]) for j, t in enumerate(toks))
            if s > best_lp:
                best, best_lp = a, s
        assert chosen == best
        assert abs(feats[0] - best_lp) < 1e-3
    assert set(out.acts) == set(bundle.sites)


def test_assistant_only_loss_mask(bundle, roots):
    ex = [(encode_root(bundle, s, sp)[0], bundle.tok.action_ids("review")) for s, sp in roots[:3]]
    ids, mask, labels = masked_batch(ex, bundle.tok.pad_id)
    for i, (p, t) in enumerate(ex):
        assert (labels[i, :len(p)] == -100).all()
        assert labels[i, len(p):len(p) + len(t)].tolist() == t
        assert (labels[i, len(p) + len(t):] == -100).all()


def test_lora_starts_at_base_and_merges_exactly():
    tok = WordTokenizer()
    m = build_small_actor(tok.vocab_size, hidden=32, layers=2, heads=2, kv_heads=1, intermediate=64).eval()
    x = torch.randint(5, tok.vocab_size, (2, 20))
    with torch.no_grad():
        ref = m(input_ids=x).logits
    names = add_lora(m, ("q_proj", "v_proj"), r=4, alpha=8, dropout=0.0)
    assert len(names) == 4
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(input_ids=x).logits, ref, atol=1e-6)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_()
        adapted = m(input_ids=x).logits
    assert merge_lora(m) == 4
    with torch.no_grad():
        assert torch.allclose(m(input_ids=x).logits, adapted, atol=1e-4)


def test_soft_prompt_equals_inserting_those_embeddings(bundle, roots):
    s, sp = roots[0]
    words = ["flag", ":", "none"]
    ids_words = [bundle.tok.stoi[w] for w in words]
    emb = bundle.model.get_input_embeddings().weight
    soft = emb[ids_words].detach().clone()
    sb = with_soft_slots(bundle, 3)
    p_soft = encode_root(sb, s, sp)[0]
    p_plain = encode_root(bundle, s, sp)[0]
    p_words = p_plain[:-1] + ids_words + p_plain[-1:]
    assert p_soft.count(bundle.tok.soft_id) == 3
    with torch.no_grad(), soft_prompt(bundle.model, bundle.tok.soft_id, soft):
        a = bundle.model(input_ids=torch.tensor([p_soft])).logits
    with torch.no_grad():
        b = bundle.model(input_ids=torch.tensor([p_words])).logits
    assert torch.allclose(a, b, atol=1e-5)
