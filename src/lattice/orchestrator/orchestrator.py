"""Thin orchestrator (spec §4): one document in, one GraphDelta out.
Batch is a fold over the stream — process_stream() just calls process()
per document; there is no separate batch code path (spec §4.1)."""

from collections.abc import Iterable
from typing import Literal

from lattice.core.types import Concept, Document, GraphDelta, GraphSnapshot
from lattice.ports import (
    Extractor,
    GraphIntegrator,
    RelationInducer,
    Resolver,
    Scorer,
    Segmenter,
)


class Orchestrator:
    """Runs the six-stage pipeline over one document at a time.

    Error policy (spec §8): "fail" re-raises the stage exception (a crash
    never silently shrinks the scored corpus); "skip" records the error in
    the GraphDelta and moves on (one poison document can't halt the stream).
    Under "skip", stages that mutated their stores before the failing stage
    keep those mutations — transactional deltas are deferred past M1.
    """

    def __init__(
        self,
        *,
        segmenter: Segmenter,
        extractor: Extractor,
        scorer: Scorer,
        resolver: Resolver,
        relation_inducer: RelationInducer,
        graph_integrator: GraphIntegrator,
        on_error: Literal["fail", "skip"] = "fail",
    ) -> None:
        self.segmenter = segmenter
        self.extractor = extractor
        self.scorer = scorer
        self.resolver = resolver
        self.relation_inducer = relation_inducer
        self.graph_integrator = graph_integrator
        self.on_error = on_error

    def process(self, document: Document) -> GraphDelta:
        try:
            units = self.segmenter.segment(document)
            mentions = self.extractor.extract(units)
            scored = self.scorer.score(mentions, units)
            selected = [sm for sm in scored if sm.selected]
            resolutions = self.resolver.resolve(selected, document)
            relations = self.relation_inducer.induce(resolutions, units, document)
            self.graph_integrator.apply(resolutions, relations)
        except Exception as exc:
            if self.on_error == "fail":
                raise
            return GraphDelta(
                document_id=document.id,
                concepts_added=(),
                concepts_updated=(),
                relations_added=(),
                errors=(f"{type(exc).__name__}: {exc}",),
            )

        added: dict[str, Concept] = {}
        updated: dict[str, Concept] = {}
        for resolution in resolutions:
            if resolution.is_new:
                added[resolution.concept.id] = resolution.concept
            else:
                updated[resolution.concept.id] = resolution.concept
        # A concept created and then re-mentioned within the same document
        # counts as added, not updated.
        for concept_id in added:
            updated.pop(concept_id, None)

        return GraphDelta(
            document_id=document.id,
            concepts_added=tuple(added.values()),
            concepts_updated=tuple(updated.values()),
            relations_added=tuple(relations),
            errors=(),
            selected_mentions=tuple(selected),
        )

    def process_stream(self, documents: Iterable[Document]) -> list[GraphDelta]:
        return [self.process(document) for document in documents]

    def snapshot(self) -> GraphSnapshot:
        return self.graph_integrator.snapshot()
