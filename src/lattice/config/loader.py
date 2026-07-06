import tomllib
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from lattice.config.schema import RunConfig

M = TypeVar("M", bound=BaseModel)


def load_config(path: str | Path, model: type[M] = RunConfig) -> M:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return model.model_validate(data)
