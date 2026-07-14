"""Public facade (M6 spec §4.1): the one import a consumer needs.

    from lattice import Engine

    engine = Engine()                    # "lite": dependency-free, same
                                         # topology as standard, toy quality
    engine = Engine(profile="standard")  # benchmark-validated stack
                                         # (install lattice[ml] + fetch models)
"""

import json
from collections.abc import Iterable
from pathlib import Path

from lattice.config.factory import build_orchestrator
from lattice.config.loader import load_config
from lattice.config.schema import RunConfig
from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Relation,
)
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

    def save(self, path: str | Path) -> None:
        """Serialize the accreted graph + config to versioned JSON (format
        v1, M6 spec §4.3). The concept store is not serialized separately:
        resolvers upsert exactly the concepts the integrator holds, so the
        snapshot is the single source of truth."""
        from lattice import __version__  # inside the function: no cycle

        snapshot = self._orchestrator.snapshot()
        payload = {
            "format_version": FORMAT_VERSION,
            "lattice_version": __version__,
            "profile": self.profile,
            "config": self.config.model_dump(),
            "document_counter": self._counter,
            "concepts": [
                {
                    "id": c.id,
                    "label": c.label,
                    "embedding": list(c.embedding),
                    "first_seen": c.first_seen,
                    "updated_at": c.updated_at,
                }
                for c in snapshot.concepts
            ],
            "relations": [
                {
                    "type": r.type,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "confidence": r.confidence,
                    "provenance": r.provenance,
                }
                for r in snapshot.relations
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Engine":
        """Rebuild an Engine from a save file. Resume-equivalence contract:
        processing after load equals processing straight through."""
        payload = json.loads(Path(path).read_text())
        found = payload.get("format_version")
        if found != FORMAT_VERSION:
            raise ValueError(
                f"unsupported save format_version {found!r} "
                f"(this lattice reads {FORMAT_VERSION})"
            )
        engine = cls.__new__(cls)
        engine._init(
            RunConfig.model_validate(payload["config"]), payload["profile"]
        )
        concepts = tuple(
            Concept(
                id=c["id"],
                label=c["label"],
                embedding=tuple(c["embedding"]),
                first_seen=c["first_seen"],
                updated_at=c["updated_at"],
            )
            for c in payload["concepts"]
        )
        relations = tuple(
            Relation(
                type=r["type"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                confidence=r["confidence"],
                provenance=r["provenance"],
            )
            for r in payload["relations"]
        )
        engine._orchestrator.graph_integrator.restore(
            GraphSnapshot(concepts=concepts, relations=relations)
        )
        store = getattr(engine._orchestrator.resolver, "concept_store", None)
        if store is not None:
            for concept in concepts:
                store.upsert(concept)
        engine._counter = int(payload["document_counter"])
        return engine

    def reset(self) -> None:
        """Empty the graph and restart the document counter."""
        self._orchestrator.graph_integrator.reset()
        store = getattr(self._orchestrator.resolver, "concept_store", None)
        if store is not None:
            store.reset()
        self._counter = 0
