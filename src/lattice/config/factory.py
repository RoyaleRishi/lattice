"""Composition root (spec §7.3): validated config → registry lookup →
instantiate with params → inject shared deps → wired orchestrator.
This is the single DIP composition root — the only place concrete adapter
classes are resolved from names."""

import inspect

import lattice.adapters  # noqa: F401  (importing registers all built-in adapters)
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.orchestrator.orchestrator import Orchestrator
from lattice.ports import (
    ConceptStore,
    Embedder,
    Extractor,
    GraphIntegrator,
    RelationInducer,
    Resolver,
    Scorer,
    Segmenter,
)
from lattice.registry.registry import lookup


def instantiate(
    port: type, spec: AdapterSpec, shared: dict[str, object] | None = None
):
    """Instantiate the registered adapter for `spec`. Any `shared` dependency
    whose key matches a constructor parameter name is injected, unless the
    config already supplies that param explicitly."""
    adapter_cls = lookup(port, spec.name)
    kwargs = dict(spec.params)
    parameters = inspect.signature(adapter_cls.__init__).parameters
    for name, dependency in (shared or {}).items():
        if name in parameters and name not in kwargs:
            kwargs[name] = dependency
    return adapter_cls(**kwargs)


def build_orchestrator(config: RunConfig) -> Orchestrator:
    embedder = instantiate(Embedder, config.embedder)
    concept_store = instantiate(ConceptStore, config.concept_store)
    shared = {"embedder": embedder, "concept_store": concept_store}
    return Orchestrator(
        segmenter=instantiate(Segmenter, config.segmenter, shared),
        extractor=instantiate(Extractor, config.extractor, shared),
        scorer=instantiate(Scorer, config.scorer, shared),
        resolver=instantiate(Resolver, config.resolver, shared),
        relation_inducer=instantiate(RelationInducer, config.relation_inducer, shared),
        graph_integrator=instantiate(GraphIntegrator, config.graph_integrator, shared),
        on_error=config.run.on_error,
    )
