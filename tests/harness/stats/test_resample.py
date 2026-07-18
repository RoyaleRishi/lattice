from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.records import ResampleBundle
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife


def _sum_bundle():
    return ResampleBundle(
        kind="pooled",
        per_document={"a": 1.0, "b": 2.0, "c": 3.0},
        aggregate=lambda records, ctx: {"total": float(sum(records))},
    )


def test_bootstrap_is_seed_deterministic_and_varies():
    b = _sum_bundle()
    assert bootstrap(b, samples=200, seed=7) == bootstrap(b, samples=200, seed=7)
    assert bootstrap(b, samples=200, seed=7) != bootstrap(b, samples=200, seed=8)


def test_bootstrap_all_fixed_is_constant_point_estimate():
    b = _sum_bundle()
    out = bootstrap(b, samples=25, seed=1, fixed_doc_ids=["a", "b", "c"])
    assert out["total"] == [6.0] * 25


def test_jackknife_leaves_one_out():
    b = _sum_bundle()
    jk = jackknife(b)
    assert sorted(jk["total"]) == [3.0, 4.0, 5.0]     # sum minus each of a,b,c


HCFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gold-mentions",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "scorer": {"name": "passthrough"},
    "resolver": {
        "name": "embedding-nn",
        "params": {"threshold": 0.8},
    },
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "mention-clusters",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "metrics": [{"name": "redundancy"}],
})


def test_bootstrap_holistic_runs_and_is_deterministic():
    a = bootstrap_holistic(HCFG, samples=5, seed=3)
    assert a == bootstrap_holistic(HCFG, samples=5, seed=3)     # same seed -> identical
    assert a != bootstrap_holistic(HCFG, samples=5, seed=4)     # different seed -> differs
    assert "redundancy.concept-count" in a
    assert len(a["redundancy.concept-count"]) == 5
