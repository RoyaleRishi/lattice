import pytest
from pydantic import ValidationError

from lattice.config.schema import AdapterSpec, RunConfig, RunPolicy


def _minimal_config() -> dict:
    return {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "frequency"},
        "resolver": {"name": "exact-label"},
        "relation_inducer": {"name": "co-occurrence"},
        "graph_integrator": {"name": "in-memory"},
    }


def test_minimal_config_validates_with_defaults():
    config = RunConfig.model_validate(_minimal_config())
    assert config.embedder == AdapterSpec(name="hashing")
    assert config.concept_store == AdapterSpec(name="in-memory")
    assert config.run == RunPolicy(on_error="fail", seed=0)


def test_params_default_to_empty_dict():
    assert AdapterSpec(name="x").params == {}


def test_missing_required_section_rejected():
    data = _minimal_config()
    del data["scorer"]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_unknown_key_rejected():
    data = _minimal_config()
    data["scorrer"] = {"name": "typo"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_invalid_on_error_rejected():
    data = _minimal_config()
    data["run"] = {"on_error": "explode"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)
