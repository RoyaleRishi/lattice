"""Public facade (M6 spec §4.1): the one import a consumer needs.

    from lattice import Engine

    engine = Engine()                    # "lite": dependency-free, same
                                         # topology as standard, toy quality
    engine = Engine(profile="standard")  # benchmark-validated stack
                                         # (install lattice[ml] + fetch models)
"""

from collections.abc import Iterable
from pathlib import Path

from lattice.config.factory import build_orchestrator
from lattice.config.loader import load_config
from lattice.config.schema import RunConfig
from lattice.core.types import Document, GraphDelta, GraphSnapshot
from lattice.graph_view import GraphView

FORMAT_VERSION = 1

_UNION = {
    "name": "union",
    "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
}

# Both profiles share one pipeline topology (M6 spec §4.1); they differ only
# in extractor and embedder. Every "standard" choice traces to sweep
# evidence: cosine scorer (M2), embedding-nn@0.90 (M5's recorded operating
# point), hearst+compound union (M4).
_PROFILES: dict[str, dict] = {
    "lite": {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "embedding-cosine"},
        "resolver": {"name": "embedding-nn", "params": {"threshold": 0.90}},
        "relation_inducer": _UNION,
        "graph_integrator": {"name": "in-memory"},
        "embedder": {"name": "hashing"},
        "concept_store": {"name": "in-memory"},
        "run": {"on_error": "fail", "seed": 0},
    },
    "standard": {
        "segmenter": {"name": "block"},
        "extractor": {"name": "noun-chunk"},
        "scorer": {"name": "embedding-cosine"},
        "resolver": {"name": "embedding-nn", "params": {"threshold": 0.90}},
        "relation_inducer": _UNION,
        "graph_integrator": {"name": "in-memory"},
        "embedder": {"name": "sentence-transformer"},
        "concept_store": {"name": "in-memory"},
        "run": {"on_error": "fail", "seed": 0},
    },
}


class Engine:
    """document → accreting concept graph. Stateful; one Engine = one graph.

    Errors propagate ("fail" policy) in both profiles — a consumer that
    prefers poison-document tolerance passes on_error="skip" via
    from_config and inspects delta.errors (partial-mutation caveat: see
    Orchestrator docstring)."""

    def __init__(self, profile: str = "lite"):
        if profile not in _PROFILES:
            known = ", ".join(sorted(_PROFILES))
            raise ValueError(f"unknown profile {profile!r} (known: {known})")
        self._init(RunConfig.model_validate(_PROFILES[profile]), profile)

    def _init(self, config: RunConfig, profile: str | None) -> None:
        self.config = config
        self.profile = profile
        self._orchestrator = build_orchestrator(config)
        self._counter = 0

    @classmethod
    def from_config(cls, config: "RunConfig | dict | str | Path") -> "Engine":
        if isinstance(config, (str, Path)):
            run_config = load_config(config, model=RunConfig)
        elif isinstance(config, RunConfig):
            run_config = config
        else:
            run_config = RunConfig.model_validate(config)
        engine = cls.__new__(cls)
        engine._init(run_config, None)
        return engine

    def ingest(
        self,
        document: Document | str,
        *,
        id: str | None = None,
        kind: str = "note",
        timestamp: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GraphDelta:
        """Process one document. Raw text is wrapped in a Document with auto
        id "doc-N" and timestamp N (a per-engine counter that save/load
        persists); every field is overridable by keyword. Passing a
        ready-made Document together with overrides is ambiguous and raises."""
        if isinstance(document, Document):
            if id is not None or timestamp is not None or metadata is not None or kind != "note":
                raise ValueError(
                    "field overrides apply to raw text only, not a Document"
                )
            return self._orchestrator.process(document)
        n = self._counter
        self._counter += 1
        return self._orchestrator.process(
            Document(
                id=id if id is not None else f"doc-{n}",
                kind=kind,
                text=document,
                timestamp=timestamp if timestamp is not None else float(n),
                metadata=metadata or {},
            )
        )

    def ingest_all(self, documents: Iterable[Document | str]) -> list[GraphDelta]:
        return [self.ingest(document) for document in documents]

    def snapshot(self) -> GraphSnapshot:
        return self._orchestrator.snapshot()

    def view(self) -> GraphView:
        return GraphView(self._orchestrator.snapshot())

    def reset(self) -> None:
        """Empty the graph and restart the document counter."""
        self._orchestrator.graph_integrator.reset()
        store = getattr(self._orchestrator.resolver, "concept_store", None)
        if store is not None:
            store.reset()
        self._counter = 0
