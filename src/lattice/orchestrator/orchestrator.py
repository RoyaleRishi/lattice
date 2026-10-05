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

    Every document is atomic under both error policies (ADR-0002): the
    resolver and the graph integrator are checkpointed before the pipeline
    runs, and both are rolled back if any stage raises, so a failed document
    leaves no trace. The policy only routes the error (spec §8): "fail"
    re-raises the original exception (a crash never silently shrinks the
    scored corpus); "skip" records it in the GraphDelta and moves on (one
    poison document can't halt the stream). If a rollback itself fails, the
    state can't be trusted, so that error is raised chained from the original
    under either policy. Atomicity holds for every exit, including
    KeyboardInterrupt; the policy only routes Exceptions.
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
        r_tok = g_tok = None
        try:
            # a failing checkpoint hasn't mutated anything, so no rollback for it
            r_tok = self.resolver.checkpoint()
            g_tok = self.graph_integrator.checkpoint()
            units = self.segmenter.segment(document)
            mentions = self.extractor.extract(units)
            scored = self.scorer.score(mentions, units)
            selected = [sm for sm in scored if sm.selected]
            resolutions = self.resolver.resolve(selected, document)
            relations = self.relation_inducer.induce(resolutions, units, document)
            self.graph_integrator.apply(resolutions, relations)
        except Exception as exc:
            self._rollback(r_tok, g_tok, exc)  # ADR-0002: atomic under both policies
            if self.on_error == "fail":
                raise
            return GraphDelta(
                document_id=document.id,
                concepts_added=(),
                concepts_updated=(),
                relations_added=(),
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        except BaseException as exc:
            # ADR-0002: Ctrl-C / SystemExit also roll back, then pass through
            # untouched; never turned into a skip delta
            self._rollback(r_tok, g_tok, exc)
            raise

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
            resolutions=tuple(resolutions),
        )

    def _rollback(self, r_tok, g_tok, original: BaseException) -> None:
        # try both even if one fails; a token is None if its checkpoint never ran
        failure: Exception | None = None
        for target, tok in ((self.resolver, r_tok), (self.graph_integrator, g_tok)):
            if tok is None:
                continue
            try:
                target.rollback(tok)
            except Exception as rb_exc:
                if failure is None:
                    failure = rb_exc
                else:  # keep the second failure visible instead of dropping it
                    failure.add_note(f"also failed: {type(rb_exc).__name__}: {rb_exc}")
        if failure is not None:
            raise failure from original

    def process_stream(self, documents: Iterable[Document]) -> list[GraphDelta]:
        return [self.process(document) for document in documents]

    def snapshot(self) -> GraphSnapshot:
        return self.graph_integrator.snapshot()
