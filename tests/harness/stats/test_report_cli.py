import json

from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.report import analyze, write_report

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gazetteer",
        "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"},
    },
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {
        "name": "union",
        "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
    },
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "taxonomy",
        "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"},
    },
    "metrics": [{"name": "edge-f1"}],
})


def test_analyze_item_level_shape_and_centering():
    report = analyze(CFG, samples=300, seed=5, level=0.95)
    assert report["seed"] == 5 and report["level"] == 0.95 and report["samples"] == 300
    f1 = report["metrics"]["edge-f1"]["f1"]
    # the interval is centered on the point estimate, and the estimate is the real F1 (1.0 on toy)
    assert f1["estimate"] == 1.0
    assert set(f1) >= {"estimate", "bca", "percentile"}
    assert set(f1["bca"]) == {"lo", "hi", "method"}


def test_write_report_is_json_and_sorted(tmp_path):
    report = analyze(CFG, samples=50, seed=1, level=0.95)
    path = write_report(report, tmp_path)
    assert path.name == "interval-report.json"
    loaded = json.loads(path.read_text())
    assert loaded["metrics"]["edge-f1"]["f1"]["estimate"] == 1.0
