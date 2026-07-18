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
    # exact-label graph accretes commutatively -> concept count is order-invariant
    assert a[key].range == 0.0
