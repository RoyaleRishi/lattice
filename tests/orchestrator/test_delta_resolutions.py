from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.extractor.token import TokenExtractor
from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.core.types import Document
from lattice.orchestrator.orchestrator import Orchestrator


def _orchestrator() -> Orchestrator:
    return Orchestrator(
        segmenter=BlockSegmenter(),
        extractor=TokenExtractor(min_length=4),
        scorer=FrequencyScorer(top_k=5),
        resolver=ExactLabelResolver(
            embedder=HashingEmbedder(dim=16), concept_store=InMemoryConceptStore()
        ),
        relation_inducer=CoOccurrenceInducer(),
        graph_integrator=InMemoryGraphIntegrator(),
    )


def test_delta_carries_resolutions():
    delta = _orchestrator().process(
        Document(id="d1", kind="note", text="alpha beta alpha", timestamp=1.0)
    )
    assert len(delta.resolutions) == len(delta.selected_mentions) > 0
    assert {r.mention.mention.surface for r in delta.resolutions} == {"alpha", "beta"}
    assert all(r.concept.id for r in delta.resolutions)


def test_default_is_empty_tuple():
    from lattice.core.types import GraphDelta

    delta = GraphDelta(
        document_id="d", concepts_added=(), concepts_updated=(), relations_added=()
    )
    assert delta.resolutions == ()
