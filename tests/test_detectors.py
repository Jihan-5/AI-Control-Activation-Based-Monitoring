"""Detector unit tests on synthetic bundles with a planted structure.

These check mechanics (shapes, finiteness, fit/score contracts, leakage-safe
behaviour). The synthetic AUROC thresholds verify that the code can recover a
planted shift; they say nothing about language models.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from irmon.config import ProposedConfig
from irmon.detectors import (
    DETECTORS,
    SUPERVISED,
    ConditionalMahalanobis,
    ContrastPairProbe,
    ExactRuleChecker,
    FunctionalAttribution,
    InterventionResponseDetector,
    MetadataShortcut,
    OutputConsistency,
    PooledMahalanobis,
    RankCombiner,
    StripStability,
    TeCoDispersion,
    ablation_configs,
    access_table,
    build_detector,
)
from irmon.detectors.linalg import Whitener, shrinkage_covariance, upper_tail_rank
from irmon.detectors.teco import change_points
from irmon.features.projection import LayerProjection, NuisanceSubspace
from irmon.features.signature import SignatureBuilder
from irmon.features.synthetic_bundle import make_synthetic_bundle

CFG = ProposedConfig(width=12, nuisance_rank=4, cross_fit_folds=4)


def _add_extras(bundle, y, seed=0):
    """Synthesize the extras arrays that extraction would normally build."""
    rng = np.random.default_rng(seed)
    n, K = len(bundle), len(bundle.slot_names)
    chosen = bundle.chosen_slot
    # TeCo: violators' decisions change at dispersed severities.
    dec = np.repeat(chosen[:, None, None], 4 * 5, axis=1).reshape(n, 4, 5).copy()
    for i in range(n):
        for k in range(4):
            cp = rng.integers(3, 6) if y[i] == 0 else rng.integers(1, 6)
            if cp <= 5:
                dec[i, k, cp - 1 :] = (chosen[i] + 1) % K
    # STRIP: violators are unusually stable (low entropy).
    conc = np.where(y == 1, 20.0, 1.0)
    probs = np.stack([rng.dirichlet(np.ones(K) / c, size=6) for c in conc])
    # Loss traces: normal roots share a per-class template; violators do not.
    template = np.random.default_rng(999).normal(size=(K, 16))  # shared "model" across splits
    traces = template[chosen] + 0.3 * rng.normal(size=(n, 16))
    traces[y == 1] = rng.normal(size=(int(y.sum()), 16))
    bundle.extras.update({
        "teco_decisions": dec,
        "strip_probs": probs,
        "loss_trace": traces,
        "oracle_flag": y.astype(float),
        "metadata": rng.normal(size=(n, 5)),
    })
    return bundle


@pytest.fixture(scope="module")
def data():
    train, _ = make_synthetic_bundle(n_normal=300, n_violation=0, d_model=24, seed=1)
    test, y = make_synthetic_bundle(n_normal=200, n_violation=80, d_model=24, seed=2)
    dev, ydev = make_synthetic_bundle(n_normal=150, n_violation=150, d_model=24, seed=3)
    _add_extras(train, np.zeros(len(train), int), seed=1)
    _add_extras(test, y, seed=2)
    _add_extras(dev, ydev, seed=3)
    return train, test, y, dev, ydev


# ------------------------------------------------------------------- linalg
def test_shrinkage_and_whitener_are_stable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 50))  # p > n: empirical covariance is singular
    for method in ("ledoit_wolf", "oas", "fixed"):
        loc, cov, lam = shrinkage_covariance(X, method=method, shrinkage=0.2)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
        s = Whitener().fit_covariance(loc, cov).score(X)
        assert np.all(np.isfinite(s)) and s.shape == (20,)
    w = Whitener().fit(X)
    assert np.isfinite(w.condition_number_)
    with pytest.raises(ValueError):
        shrinkage_covariance(np.full((5, 3), np.nan))


def test_upper_tail_rank_monotone():
    ref = np.arange(100.0)
    r = upper_tail_rank(ref, np.array([-1.0, 50.0, 1000.0]))
    assert r[0] == 0 and r[2] == 1 and 0.4 < r[1] < 0.6


# ------------------------------------------------------- projection/signature
def test_projection_and_nuisance(data):
    train = data[0]
    resp = train.responses("T")
    proj = LayerProjection("pca", 8).fit(resp)
    out = proj.transform(resp)
    assert out.shape == resp.shape[:-1] + (8,)
    rnd = LayerProjection("random", 8, seed=3).fit(resp)
    V = rnd.components_[0]
    assert np.allclose(V.T @ V, np.eye(8), atol=1e-8)
    ns = NuisanceSubspace(rank=4).fit(resp)
    cleaned = ns.remove(resp)
    B = ns.bases_[0]
    assert np.abs(cleaned[..., 0, :] @ B).max() < 1e-8
    assert all(0 <= e <= 1 for e in ns.explained_)


def test_signature_layout_and_nan_masking(data):
    train = data[0]
    b = train.subset(np.arange(50))
    b.edit_valid[:5, 4] = False  # first U edit invalid for 5 roots
    sb = SignatureBuilder(width=6, nuisance_rank=3).fit(b)
    S = sb.transform(b)
    E, L, K = len(b.edit_names), len(b.layers), len(b.slot_names)
    # Default action features: K raw slot changes + 3 invariant summaries per edit.
    assert S.shape == (50, E * L * 6 + E * (K + 3))
    assert SignatureBuilder(width=6, nuisance_rank=3, action_features="raw").fit_transform(b).shape == (50, E * L * 6 + E * K)
    assert SignatureBuilder(width=6, nuisance_rank=3, action_features="summary").fit_transform(b).shape == (50, E * L * 6 + E * 3)
    nan_rows = np.isnan(S).any(axis=1)
    assert nan_rows[:5].all() and not nan_rows[5:].any()
    groups = sb.groups("layer")
    assert set(groups) == {f"layer{layer}" for layer in b.layers} | {"actions"}
    assert sum(len(c) for c in groups.values()) == S.shape[1]
    with pytest.raises(ValueError):
        SignatureBuilder(use_t=False, use_u=False).fit(b)


# --------------------------------------------------------------- proposed
@pytest.mark.parametrize("aggregation", ["concat", "max"])
def test_proposed_recovers_planted_shift(data, aggregation):
    train, test, y, *_ = data
    det = InterventionResponseDetector(CFG, aggregation=aggregation).fit(train)
    s = det.score(test)
    assert s.shape == (len(test),) and np.all(np.isfinite(s))
    assert roc_auc_score(y, s) > 0.8
    assert det.access.forwards_per_root == 1 + len(train.edit_names)


def test_proposed_uses_cross_fitting(data):
    det = InterventionResponseDetector(CFG).fit(data[0])
    # Out-of-fold residuals are larger than in-sample residuals.
    assert det.residual_trace_crossfit_ > det.residual_trace_insample_


def test_proposed_invalid_edits_give_finite_scores(data):
    train, test, *_ = data
    det = InterventionResponseDetector(CFG).fit(train)
    t2 = test.subset(np.arange(len(test)))
    t2.edit_valid[::3, :] = False  # every edit invalid for a third of roots
    s = det.score(t2)
    assert np.all(np.isfinite(s))


def test_proposed_mlp_and_no_context(data):
    train, test, y, *_ = data
    for kw in ({"predictor": "mlp"}, {"predictor": "none"}, {"use_context": False}, {"include_base_snapshot": True}):
        det = InterventionResponseDetector(CFG, **kw).fit(train)
        assert np.all(np.isfinite(det.score(test)))


def test_ablations_run(data):
    train, test, *_ = data
    abl = ablation_configs(CFG)
    assert set(abl) == {"no_T", "no_U", "no_activations", "no_conditioning", "random_projection",
                        "no_nuisance_removal", "base_context_only"}
    for name, cfg in abl.items():
        det = InterventionResponseDetector(cfg).fit(train)
        s = det.score(test)
        assert np.all(np.isfinite(s)), name
    assert not abl["no_activations"].use_activations


def test_proposed_is_picklable(data):
    import pickle

    det = InterventionResponseDetector(CFG, predictor="mlp").fit(data[0])
    clone = pickle.loads(pickle.dumps(det))
    assert np.allclose(clone.score(data[1]), det.score(data[1]))


# -------------------------------------------------------------- baselines
def test_mahalanobis_variants(data):
    train, test, *_ = data
    for det in (
        PooledMahalanobis(pca_dim=16),
        PooledMahalanobis(pca_dim=None, aggregation="max"),
        PooledMahalanobis(pca_dim=16, aggregation="sum"),
        ConditionalMahalanobis(pca_dim=16, min_group_size=10),
        ConditionalMahalanobis(mode="ridge", pca_dim=16),
    ):
        s = det.fit(train).score(test)
        assert s.shape == (len(test),) and np.all(np.isfinite(s))


def test_conditional_mahalanobis_unseen_group_fallback(data):
    train, test, *_ = data
    det = ConditionalMahalanobis(pca_dim=8, min_group_size=10).fit(train)
    t2 = test.subset(np.arange(10))
    t2.context_group[:] = "never-seen-group"
    s = det.score(t2)
    assert np.all(np.isfinite(s))
    # Large min_group_size forces every group to the pooled mean.
    det2 = ConditionalMahalanobis(pca_dim=8, min_group_size=10**6).fit(train)
    assert det2.group_means_ == {} and det2.n_groups_fallback_ > 0


def test_output_consistency_modes(data):
    train, test, y, *_ = data
    for mode in ("residual", "t_instability", "u_insensitivity", "combined"):
        det = OutputConsistency(mode=mode, proposed_cfg=CFG).fit(train)
        s = det.score(test)
        assert s.shape == (len(test),) and np.all(np.isfinite(s)), mode
        assert not det.access.activations
    # Synthetic violators do not react to U edits -> u_insensitivity separates them.
    u = OutputConsistency(mode="u_insensitivity").fit(train).score(test)
    assert u[y == 1].mean() >= u[y == 0].mean()


def test_teco_strip_fa_oracle(data):
    train, test, y, *_ = data
    cp = change_points(np.array([[[0, 1, 1], [0, 0, 0]]]), np.array([0]))
    assert cp.tolist() == [[2.0, 4.0]]
    for det in (TeCoDispersion(), TeCoDispersion("mean_changepoint"), StripStability(),
                StripStability(direction="low"), FunctionalAttribution(),
                FunctionalAttribution("mahalanobis"), ExactRuleChecker()):
        s = det.fit(train).score(test)
        assert s.shape == (len(test),) and np.all(np.isfinite(s)), det.name
    assert roc_auc_score(y, StripStability(direction="low").fit(train).score(test)) > 0.8
    assert roc_auc_score(y, FunctionalAttribution().fit(train).score(test)) > 0.8
    assert roc_auc_score(y, ExactRuleChecker().fit(train).score(test)) == 1.0
    assert FunctionalAttribution().fit(train).access.weights


def test_pcir_runs(data):
    pytest.importorskip("torch")
    train, test, *_ = data
    det = build_detector("pcir", pca_dim=8, hidden=16, latent=4, epochs=3)
    s = det.fit(train).score(test)
    assert s.shape == (len(test),) and np.all(np.isfinite(s))


def test_supervised_require_labels(data):
    train, test, y, dev, ydev = data
    for name in sorted(SUPERVISED):
        det = build_detector(name)
        with pytest.raises(ValueError):
            det.fit(dev)
        det.fit(dev, ydev)
        s = det.score(test)
        assert np.all(np.isfinite(s))
        assert det.access.violation_labels and not det.access.normal_only
    # The synthetic generator plants violations in responses, not in h_base; add
    # an h_base shift so the supervised probe has something to learn.
    shift = np.random.default_rng(5).normal(size=dev.h_base.shape[1:]).astype(np.float32)
    dev2, test2 = dev.subset(np.arange(len(dev))), test.subset(np.arange(len(test)))
    dev2.h_base[ydev == 1] += shift
    test2.h_base[y == 1] += shift
    probe = build_detector("probe_supervised", pca_dim=8).fit(dev2, ydev)
    assert roc_auc_score(y, probe.score(test2)) > 0.8
    meta = MetadataShortcut().fit(dev, ydev)
    rep = meta.leakage_report(test, y)
    assert rep["auroc"] < 0.75 and not rep["alarm"]  # random metadata carries no signal


def test_contrast_probe(data):
    train, test, y, *_ = data
    rng = np.random.default_rng(0)
    pos = train.h_base[:20] + rng.normal(size=train.h_base[:20].shape)
    neg = train.h_base[20:40]
    det = ContrastPairProbe(pos=pos, neg=neg).fit(train)
    assert np.all(np.isfinite(det.score(test)))
    with pytest.raises(ValueError):
        ContrastPairProbe().fit(train)


def test_rank_combiner(data):
    train, test, y, *_ = data
    comb = RankCombiner([InterventionResponseDetector(CFG), PooledMahalanobis(pca_dim=8)], rule="max").fit(train)
    s = comb.score(test)
    assert np.all((s >= 0) & (s <= 1))
    assert roc_auc_score(y, s) > 0.7
    with pytest.raises(ValueError):
        RankCombiner([build_detector("behavior_logistic")]).fit(train)


def test_registry_and_access_table(data):
    train = data[0]
    assert set(DETECTORS) == {
        "proposed", "mahalanobis_pooled", "mahalanobis_conditional", "output_consistency", "teco", "strip",
        "pcir", "functional_attribution", "probe_supervised", "probe_contrast", "behavior_logistic", "oracle",
        "metadata_shortcut",
    }
    with pytest.raises(KeyError):
        build_detector("nope")
    dets = [build_detector("proposed", CFG).fit(train), build_detector("mahalanobis_pooled", pca_dim=8).fit(train)]
    tab = access_table(dets)
    assert list(tab.index) == ["proposed", "mahalanobis_pooled"]
    assert tab.loc["proposed", "forwards_per_root"] > tab.loc["mahalanobis_pooled", "forwards_per_root"]
    with pytest.raises(RuntimeError):
        build_detector("mahalanobis_pooled").score(train)
