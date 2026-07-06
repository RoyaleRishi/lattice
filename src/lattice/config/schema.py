"""Declarative run configuration (spec §7.2). A run is one adapter name +
params per port. `extra="forbid"` everywhere so config typos fail loudly
instead of silently changing an experiment."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_error: Literal["fail", "skip"] = "fail"  # spec §8
    seed: int = 0  # stamped for reproducibility (spec §7)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmenter: AdapterSpec
    extractor: AdapterSpec
    scorer: AdapterSpec
    resolver: AdapterSpec
    relation_inducer: AdapterSpec
    graph_integrator: AdapterSpec
    embedder: AdapterSpec = AdapterSpec(name="hashing")
    concept_store: AdapterSpec = AdapterSpec(name="in-memory")
    run: RunPolicy = RunPolicy()
