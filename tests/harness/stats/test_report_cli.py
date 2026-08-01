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

# f1-at-k is the only shipped *macro* metric; the toy Inspec fixture keeps the
# macro branch of analyze() covered without the ml stack.
MACRO_CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "token", "params": {"min_length": 4}},
    "scorer": {"name": "frequency", "params": {"top_k": 15}},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "inspec", "params": {"root": "tests/fixtures/mini_inspec"}},
    "document_metrics": [{"name": "f1-at-k"}],
})


def test_analyze_item_level_shape_and_centering():
    report = analyze(CFG, samples=300, seed=5, level=0.95)
    assert report["seed"] == 5 and report["level"] == 0.95 and report["samples"] == 300
    f1 = report["metrics"]["edge-f1"]["f1"]
    # the interval is centered on the point estimate, and the estimate is the real F1 (1.0 on toy)
    assert f1["estimate"] == 1.0
    # edge-f1 is pooled -> m-out-of-n subsampling, and no bca/percentile entry:
    # both of those read the draws as a full-size bootstrap distribution, which
    # is precisely the construction that produced non-bracketing M4 intervals.
    assert set(f1) == {"estimate", "subsample", "scheme", "m", "n"}
    assert set(f1["subsample"]) == {"lo", "hi", "method"}
    assert f1["subsample"]["method"] == "subsample"
    assert (f1["scheme"], f1["m"], f1["n"]) == ("subsample", 2, 3)
    assert f1["subsample"]["lo"] <= f1["estimate"] <= f1["subsample"]["hi"]


def test_analyze_macro_metric_keeps_the_bootstrap_and_bca():
    # Macro metrics average a per-document score — a degree-1 functional of
    # exchangeable documents — so the classical n-out-of-n bootstrap and BCa
    # remain correct and must not be switched to subsampling.
    report = analyze(MACRO_CFG, samples=200, seed=5, level=0.95)
    entry = report["metrics"]["f1-at-k"]["f1@5"]
    assert set(entry) == {"estimate", "bca", "percentile", "scheme", "m", "n"}
    assert entry["scheme"] == "resample"
    assert entry["m"] == entry["n"] == 3       # mini_inspec has three documents


def test_write_report_is_json_and_sorted(tmp_path):
    report = analyze(CFG, samples=50, seed=1, level=0.95)
    path = write_report(report, tmp_path)
    assert path.name == "interval-report.json"
    loaded = json.loads(path.read_text())
    assert loaded["metrics"]["edge-f1"]["f1"]["estimate"] == 1.0


def test_analyze_item_level_honors_fixed_prefix():
    # Holding stream doc 0 (M4's glossary) fixed changes the recall CI but not the
    # point estimate: without it, draws can drop the glossary's compound edges,
    # widening the interval. Fails if the item-level branch ignores fixed_prefix.
    free = analyze(CFG, samples=400, seed=0, fixed_prefix=0)["metrics"]["edge-f1"]["recall"]
    held = analyze(CFG, samples=400, seed=0, fixed_prefix=1)["metrics"]["edge-f1"]["recall"]
    assert free["estimate"] == held["estimate"] == 1.0     # point estimate unaffected
    assert (free["m"], free["n"]) == (2, 3)                # pool is all three documents
    assert (held["m"], held["n"]) == (2, 2)                # glossary held out of the pool
    assert free["subsample"] != held["subsample"]          # CI reflects the held glossary


def test_analyze_pooled_interval_brackets_its_own_point_estimate():
    # The shipped M4 reports' concrete failure: every interval excluded its own
    # point estimate. The subsample construction is anchored on the estimate, so
    # it cannot do that when the draws straddle it — and the draws here do,
    # because the toy corpus scores a perfect 1.0 that no subset can exceed.
    metrics = analyze(CFG, samples=400, seed=0)["metrics"]["edge-f1"]
    for key, entry in metrics.items():
        iv = entry["subsample"]
        assert iv["lo"] <= entry["estimate"] <= iv["hi"], key
