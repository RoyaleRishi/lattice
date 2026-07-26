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
