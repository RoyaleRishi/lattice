import tomllib
from pathlib import Path

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig, expand


def test_multiwoz_sweep_config_expands_to_the_four_point_axis():
    sweep = load_config("configs/m5-multiwoz-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    assert [c.resolver.name for c in configs] == [
        "exact-label", "embedding-nn", "embedding-nn", "embedding-nn",
    ]
    assert [c.resolver.params.get("threshold") for c in configs] == [
        None, 0.90, 0.75, 0.65,
    ]
    for config in configs:
        assert config.dataset.params["root"] == "data/multiwoz"


def test_ecbplus_sweep_config_expands_to_the_four_point_axis():
    sweep = load_config("configs/m5-ecbplus-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    for config in configs:
        assert config.dataset.params["root"] == "data/ecbplus"


def _every_shipped_experiment_config():
    """Every runnable config in configs/, sweeps expanded into their arms."""
    for path in sorted(Path("configs").glob("*.toml")):
        with path.open("rb") as f:
            raw = tomllib.load(f)
        if "base" in raw:
            yield from ((path, c) for c in expand(load_config(path, model=SweepConfig)))
        else:
            yield path, load_config(path, model=ExperimentConfig)


def test_no_shipped_config_measures_redundancy_circularly():
    """Pins the circularity hazard at the config layer: the `redundancy`
    metric's cosine criterion is vacuous when its threshold sits at or above
    the resolver's, because the resolver has already merged every pair that
    could clear the bar. The metric now refuses to default (see
    tests/adapters/test_redundancy_metric.py); this is the other half — no
    shipped arm may set a colliding threshold explicitly."""
    checked = 0
    for path, config in _every_shipped_experiment_config():
        for metric in config.metrics:
            if metric.name != "redundancy":
                continue
            threshold = metric.params.get("threshold")
            assert threshold is not None, f"{path}: redundancy without an explicit threshold"
            resolver_threshold = config.resolver.params.get("threshold")
            if resolver_threshold is not None:
                assert threshold < resolver_threshold, (
                    f"{path}: redundancy threshold {threshold} >= resolver "
                    f"{config.resolver.name} threshold {resolver_threshold} — "
                    "cosine-duplicate-pairs is vacuously 0 in that arm"
                )
            checked += 1
    # three M5 sweeps at four resolver arms each, plus three operating-point
    # configs — a guard so this test cannot silently check nothing.
    assert checked == 15, f"expected 15 shipped redundancy arms, checked {checked}"


def test_nn090_configs_load_at_the_operating_point():
    for root in ("multiwoz", "ecbplus"):
        config = load_config(
            f"configs/m5-{root}-nn090.toml", model=ExperimentConfig
        )
        assert config.resolver.name == "embedding-nn"
        assert config.resolver.params["threshold"] == 0.90
        assert config.dataset.params["root"] == f"data/{root}"
        metric_names = {m.name for m in config.metrics}
        assert {"redundancy", "hierarchy-sanity"} <= metric_names
        assert any(dm.name == "coherence" for dm in config.document_metrics)
