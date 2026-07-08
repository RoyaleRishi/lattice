import tomllib
from pathlib import Path

from pydantic import BaseModel

from lattice.config.schema import RunConfig


def load_config[M: BaseModel](path: str | Path, model: type[M] = RunConfig) -> M:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return model.model_validate(data)
