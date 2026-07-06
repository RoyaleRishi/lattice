import pytest
from pydantic import ValidationError

from lattice.config.loader import load_config

VALID_TOML = """
[segmenter]
name = "block"

[extractor]
name = "token"
[extractor.params]
min_length = 4

[scorer]
name = "frequency"

[resolver]
name = "exact-label"

[relation_inducer]
name = "co-occurrence"

[graph_integrator]
name = "in-memory"

[run]
on_error = "skip"
"""


def test_load_valid_toml(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(VALID_TOML)
    config = load_config(path)
    assert config.segmenter.name == "block"
    assert config.extractor.params == {"min_length": 4}
    assert config.run.on_error == "skip"


def test_load_invalid_toml_raises_validation_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[segmenter]\nname = "block"\n')  # missing required sections
    with pytest.raises(ValidationError):
        load_config(path)
