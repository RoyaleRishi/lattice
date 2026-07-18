from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.permutation import SpreadResult, order_spread

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gold-mentions",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "mention-clusters",
        "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"},
    },
    "metrics": [
        {"name": "redundancy"},
        {"name": "hierarchy-sanity"},
    ],
})


def test_order_spread_reports_stats_and_is_deterministic():
    a = order_spread(CFG, permutations=6, seed=2)
    b = order_spread(CFG, permutations=6, seed=2)
    assert a == b
    key = "redundancy.concept-count"
    assert isinstance(a[key], SpreadResult)
    assert a[key].range == a[key].max - a[key].min
    assert a[key].min <= a[key].max
    # This fixture's pipeline is order-invariant, so the spread is 0. That alone
    # does NOT prove the shuffle ran (a no-op shuffle also yields 0) — the shuffle
    # mechanism is tested directly in test_order_spread_permutes_the_pool_and_holds_the_prefix.
    assert a[key].range == 0.0


def test_order_spread_permutes_the_pool_and_holds_the_prefix(monkeypatch):
    # order_spread's job is to VARY document order; on this fixture the pipeline is
    # order-invariant, so metric spread cannot prove the shuffle happened. Stub
    # run_on_documents to capture the orderings it receives and assert (a) the fixed
    # prefix stays first and (b) the shuffled tail genuinely varies. Fails iff
    # rng.shuffle is dropped (one distinct tail) or fixed/pool are swapped (prefix not held).
    import lattice.harness.stats.permutation as permutation_module

    seen: list[list[str]] = []

    def spy(config, documents):
        seen.append([d.id for d in documents])
        return {"m.x": 0.0}

    monkeypatch.setattr(permutation_module, "run_on_documents", spy)
    order_spread(CFG, permutations=20, seed=1, fixed_prefix=1)
    assert all(o[0] == seen[0][0] for o in seen)          # fixed prefix held first
    assert len({tuple(o[1:]) for o in seen}) > 1          # pool genuinely shuffled
