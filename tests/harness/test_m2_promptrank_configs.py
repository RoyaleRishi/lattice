from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig, expand


def test_promptrank_sweep_expands_to_five_scorers():
    sweep = load_config("configs/m2-promptrank-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert [c.scorer.name for c in configs] == [
        "frequency", "embedding-cosine", "mderank", "hcuke", "promptrank",
    ]
    for c in configs:
        assert c.scorer.params["top_k"] == 15
        assert c.dataset.params["split"] == "test"
        assert c.dataset.name == "inspec"


def test_promptrank_single_config_loads():
    cfg = load_config("configs/m2-promptrank-f1atk.toml", model=ExperimentConfig)
    assert cfg.scorer.name == "promptrank"
    assert cfg.scorer.params["top_k"] == 15
    assert cfg.dataset.name == "inspec" and cfg.dataset.params["split"] == "test"
    assert any(dm.name == "f1-at-k" for dm in cfg.document_metrics)
