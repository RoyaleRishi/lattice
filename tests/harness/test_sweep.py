import json

import pytest

from lattice.harness.sweep import SweepConfig, expand, run_sweep, write_reports

BASE = {
    "segmenter": {"name": "block"},
    "extractor": {"name": "token", "params": {"min_length": 4}},
    "scorer": {"name": "frequency"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "dataset": {"name": "toy"},
    "metrics": [{"name": "label-f1"}],
}


def make_sweep(axes) -> SweepConfig:
    return SweepConfig.model_validate({"base": BASE, "axes": axes})


def test_expand_cartesian_product_in_sorted_axis_order():
    sweep = make_sweep(
        {
            "scorer": [{"name": "frequency", "params": {"top_k": 5}},
                        {"name": "frequency", "params": {"top_k": 10}}],
            "extractor": [{"name": "token", "params": {"min_length": 3}}],
        }
    )
    configs = expand(sweep)
    assert len(configs) == 2
    assert all(c.extractor.params == {"min_length": 3} for c in configs)
    assert [c.scorer.params["top_k"] for c in configs] == [5, 10]


def test_expand_no_axes_yields_base_only():
    configs = expand(make_sweep({}))
    assert len(configs) == 1
    assert configs[0].scorer.name == "frequency"


def test_expand_unknown_axis_rejected():
    with pytest.raises(ValueError, match="not-a-port"):
        expand(make_sweep({"not-a-port": [{"name": "x"}]}))


def test_run_sweep_produces_one_row_per_config():
    sweep = make_sweep(
        {"scorer": [{"name": "frequency", "params": {"top_k": 5}},
                     {"name": "frequency", "params": {"top_k": 10}}]}
    )
    report = run_sweep(sweep)
    assert len(report.runs) == 2
    assert len(report.table) == 2
    assert report.table[0]["axis:scorer"] == "frequency"
    assert "label-f1.f1" in report.table[0]


def test_sweep_is_reproducible():
    sweep = make_sweep({"scorer": [{"name": "frequency"}]})
    assert run_sweep(sweep) == run_sweep(sweep)


def test_write_reports(tmp_path):
    report = run_sweep(make_sweep({}))
    json_path, md_path = write_reports(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert len(data["runs"]) == 1
    assert md_path.read_text().startswith("|")


def test_markdown_table_unions_columns_across_rows():
    from lattice.harness.sweep import _markdown_table

    table = [{"a": 1.0}, {"a": 2.0, "b": "x"}]
    rendered = _markdown_table(table)
    header = rendered.splitlines()[0]
    assert "a" in header and "b" in header
