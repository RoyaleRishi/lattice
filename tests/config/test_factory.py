import pytest

from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.ports import Scorer
from lattice.registry.registry import RegistryError


def make_run_config(**overrides) -> RunConfig:
    data = {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "frequency"},
        "resolver": {"name": "exact-label"},
        "relation_inducer": {"name": "co-occurrence"},
        "graph_integrator": {"name": "in-memory"},
    }
    data.update(overrides)
    return RunConfig.model_validate(data)


def test_build_orchestrator_wires_configured_adapters():
    orchestrator = build_orchestrator(make_run_config())
    assert isinstance(orchestrator.segmenter, BlockSegmenter)
    assert isinstance(orchestrator.scorer, FrequencyScorer)


def test_params_reach_the_adapter():
    config = make_run_config(scorer={"name": "frequency", "params": {"top_k": 3}})
    orchestrator = build_orchestrator(config)
    assert orchestrator.scorer.top_k == 3


def test_shared_dependencies_injected_by_parameter_name():
    config = make_run_config(embedder={"name": "hashing", "params": {"dim": 32}})
    orchestrator = build_orchestrator(config)
    assert orchestrator.resolver.embedder.dim == 32
    assert orchestrator.resolver.concept_store is not None


def test_on_error_policy_flows_from_config():
    config = make_run_config(run={"on_error": "skip"})
    assert build_orchestrator(config).on_error == "skip"


def test_unknown_adapter_name_raises_registry_error():
    config = make_run_config(scorer={"name": "does-not-exist"})
    with pytest.raises(RegistryError, match="does-not-exist"):
        build_orchestrator(config)


def test_instantiate_explicit_params_win_over_injection():
    scorer = instantiate(
        Scorer,
        AdapterSpec(name="frequency", params={"top_k": 7}),
        shared={"top_k": 99},
    )
    assert scorer.top_k == 7
