import pytest

from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.extractor.token import TokenExtractor
from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.orchestrator.orchestrator import Orchestrator
from lattice.ports import Extractor
from tests.helpers import make_document


def build_orchestrator(**overrides) -> Orchestrator:
    embedder = HashingEmbedder(dim=16)
    store = InMemoryConceptStore()
    stages = {
        "segmenter": BlockSegmenter(),
        "extractor": TokenExtractor(min_length=4),
        "scorer": FrequencyScorer(top_k=10),
        "resolver": ExactLabelResolver(embedder=embedder, concept_store=store),
        "relation_inducer": CoOccurrenceInducer(),
        "graph_integrator": InMemoryGraphIntegrator(),
        "on_error": "fail",
    }
    stages.update(overrides)
    return Orchestrator(**stages)


class ExplodingExtractor(Extractor):
    def extract(self, units):
        raise RuntimeError("boom")


def test_process_returns_delta_with_new_concepts():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(
        make_document(id="d1", text="The vector store indexes embeddings.")
    )
    labels = {c.label for c in delta.concepts_added}
    assert {"vector", "store", "indexes", "embeddings"} == labels
    assert delta.concepts_updated == ()
    assert delta.errors == ()
    assert delta.document_id == "d1"


def test_process_produces_co_occurrence_relations():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(make_document(id="d1", text="vector store"))
    assert len(delta.relations_added) == 1
    assert delta.relations_added[0].type == "CO_OCCURS"


def test_second_document_merges_instead_of_duplicating():
    orchestrator = build_orchestrator()
    orchestrator.process(make_document(id="d1", text="vector store"))
    delta2 = orchestrator.process(make_document(id="d2", text="vector store"))
    assert delta2.concepts_added == ()
    assert {c.label for c in delta2.concepts_updated} == {"vector", "store"}
    assert len(orchestrator.snapshot().concepts) == 2


def test_repeated_surface_in_one_document_counts_as_added_only():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(
        make_document(id="d1", text="vector store\n\nvector store")
    )
    assert {c.label for c in delta.concepts_added} == {"vector", "store"}
    assert delta.concepts_updated == ()


def test_process_stream_folds_in_order():
    orchestrator = build_orchestrator()
    deltas = orchestrator.process_stream(
        [
            make_document(id="d1", text="vector store", timestamp=1.0),
            make_document(id="d2", text="vector encoder", timestamp=2.0),
        ]
    )
    assert [d.document_id for d in deltas] == ["d1", "d2"]
    assert {c.label for c in deltas[1].concepts_added} == {"encoder"}
    assert {c.label for c in deltas[1].concepts_updated} == {"vector"}


def test_unselected_mentions_never_reach_the_graph():
    orchestrator = build_orchestrator(scorer=FrequencyScorer(top_k=1))
    delta = orchestrator.process(
        make_document(id="d1", text="vector vector store")
    )
    assert {c.label for c in delta.concepts_added} == {"vector"}


def test_on_error_fail_raises():
    orchestrator = build_orchestrator(extractor=ExplodingExtractor())
    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.process(make_document(id="d1"))


def test_on_error_skip_records_error_and_continues():
    orchestrator = build_orchestrator(
        extractor=ExplodingExtractor(), on_error="skip"
    )
    deltas = orchestrator.process_stream(
        [make_document(id="d1"), make_document(id="d2")]
    )
    assert len(deltas) == 2
    for delta in deltas:
        assert delta.concepts_added == ()
        assert len(delta.errors) == 1
        assert "boom" in delta.errors[0]


def test_delta_carries_selected_mentions_in_scorer_order():
    orchestrator = build_orchestrator(scorer=FrequencyScorer(top_k=1))
    delta = orchestrator.process(make_document(id="d1", text="vector vector store"))
    surfaces = [sm.mention.surface for sm in delta.selected_mentions]
    assert surfaces == ["vector", "vector"]
    assert all(sm.selected for sm in delta.selected_mentions)


def test_skip_path_has_no_selected_mentions():
    orchestrator = build_orchestrator(extractor=ExplodingExtractor(), on_error="skip")
    delta = orchestrator.process(make_document(id="d1"))
    assert delta.selected_mentions == ()
