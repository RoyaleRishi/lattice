import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ROOT = "tests/fixtures/mini_texeval"
GOLD_KEYS = [
    "env-eurovoc",
    "food",
    "food-wordnet",
    "science",
    "science-eurovoc",
    "science-wordnet",
]
METRIC_KEYS = {"precision", "recall", "f1", "predicted_edges", "gold_edges"}
UNION = {
    "name": "union",
    "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
}


def _config(inducer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gazetteer", "params": {"root": ROOT, "gold": "toy"}},
            "scorer": {"name": "passthrough"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": inducer,
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {"name": "taxonomy", "params": {"root": ROOT, "gold": "toy"}},
            "metrics": [{"name": "edge-f1"}],
        }
    )


@pytest.mark.parametrize(
    "inducer", [{"name": "compound"}, {"name": "hearst"}, UNION]
)
def test_m4_rows_run_clean(inducer):
    report = run_experiment(_config(inducer))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["edge-f1"]) == METRIC_KEYS


def test_union_dominates_members_on_the_fixture():
    """The M4 thesis in miniature: string structure and corpus evidence each
    find edges the other cannot; their union is exactly the toy gold."""
    f1 = {
        name: run_experiment(_config(spec)).metrics["edge-f1"]["f1"]
        for name, spec in [
            ("compound", {"name": "compound"}),
            ("hearst", {"name": "hearst"}),
            ("union", UNION),
        ]
    }
    # 2*1.0*0.6/1.6 is 0.7499999999999999 in floats — approx, not ==
    assert f1["compound"] == pytest.approx(0.75)
    assert f1["hearst"] == pytest.approx(0.75)
    assert f1["union"] == 1.0  # 2*1*1/2 is exact
    assert f1["union"] >= max(f1["compound"], f1["hearst"])


def test_m4_run_is_reproducible():
    assert run_experiment(_config(UNION)) == run_experiment(_config(UNION))


@pytest.mark.parametrize("key", GOLD_KEYS)
def test_m4_sweep_configs_expand_to_three_rows(key):
    sweep = load_config(f"configs/m4-{key}-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 3
    assert [c.relation_inducer.name for c in configs] == [
        "compound",
        "hearst",
        "union",
    ]
    for config in configs:
        assert config.dataset.params["gold"] == key
        assert config.extractor.params["gold"] == key
        assert config.metrics[0].name == "edge-f1"
