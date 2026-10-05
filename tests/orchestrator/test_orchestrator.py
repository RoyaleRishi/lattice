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
from lattice.ports import Extractor, RelationInducer
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


class ExplodingRelationInducer(RelationInducer):
    """Fails after the resolver has already run, unlike ExplodingExtractor
    (which fails before the resolver mutates anything) — the case that
    exposed the store/graph divergence this test guards against."""

    def induce(self, resolutions, units, document):
        raise RuntimeError("boom-in-relations")


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


def test_on_error_skip_rolls_back_concept_store_when_relation_inducer_fails():
    """The resolver upserts into the concept store before the relation
    inducer runs. If the inducer then raises, the store must end up
    byte-identical to its pre-document state, not just the returned delta
    empty — otherwise the store and graph diverge (spec-enforced
    resume-equivalence breaks)."""
    orchestrator = build_orchestrator(
        relation_inducer=ExplodingRelationInducer(), on_error="skip"
    )
    orchestrator.process(make_document(id="d0", text="vector store"))
    resolver = orchestrator.resolver
    before = resolver.snapshot().concepts

    delta = orchestrator.process(make_document(id="d1", text="new mention encoder"))

    assert len(delta.errors) == 1
    assert resolver.snapshot().concepts == before
    assert all(c.label != "encoder" for c in resolver.snapshot().concepts)
    assert all(c.label != "mention" for c in resolver.snapshot().concepts)


# --- ADR-0002: every document is atomic under both policies ---------------


class LateExplodingExtractor(Extractor):
    """Fine on the first call, blows up on every later one."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def extract(self, units):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("boom-late")
        return self.inner.extract(units)


class PartialApplyIntegrator(InMemoryGraphIntegrator):
    """Applies the first concept, then raises (an apply that dies midway)."""

    def __init__(self):
        super().__init__()
        self.explode = False

    def apply(self, resolutions, relations):
        if not self.explode:
            return super().apply(resolutions, relations)
        super().apply(resolutions[:1], [])
        raise RuntimeError("boom-mid-apply")


def _run_bad_doc(orchestrator, doc):
    """Process a doc that should fail; returns the delta under skip, None under fail."""
    if orchestrator.on_error == "fail":
        with pytest.raises(RuntimeError):
            orchestrator.process(doc)
        return None
    return orchestrator.process(doc)


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_failing_late_stage_leaves_graph_and_resolver_untouched(on_error):
    orchestrator = build_orchestrator(on_error=on_error)
    orchestrator.process(make_document(id="d0", text="vector store"))
    graph_before = orchestrator.snapshot()
    resolver_before = orchestrator.resolver.snapshot().concepts
    orchestrator.relation_inducer = ExplodingRelationInducer()

    delta = _run_bad_doc(orchestrator, make_document(id="d1", text="new mention encoder"))

    if on_error == "skip":
        assert len(delta.errors) == 1
        assert "boom-in-relations" in delta.errors[0]
    assert orchestrator.snapshot() == graph_before
    assert orchestrator.resolver.snapshot().concepts == resolver_before


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_failing_extractor_leaves_state_untouched(on_error):
    extractor = LateExplodingExtractor(TokenExtractor(min_length=4))
    orchestrator = build_orchestrator(extractor=extractor, on_error=on_error)
    orchestrator.process(make_document(id="d0", text="vector store"))
    graph_before = orchestrator.snapshot()
    resolver_before = orchestrator.resolver.snapshot().concepts

    _run_bad_doc(orchestrator, make_document(id="d1", text="new mention encoder"))

    assert orchestrator.snapshot() == graph_before
    assert orchestrator.resolver.snapshot().concepts == resolver_before


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_integrator_failing_mid_apply_is_rolled_back(on_error):
    integrator = PartialApplyIntegrator()
    orchestrator = build_orchestrator(graph_integrator=integrator, on_error=on_error)
    orchestrator.process(make_document(id="d0", text="vector store"))
    graph_before = orchestrator.snapshot()
    resolver_before = orchestrator.resolver.snapshot().concepts
    integrator.explode = True

    _run_bad_doc(orchestrator, make_document(id="d1", text="new mention encoder"))

    assert orchestrator.snapshot() == graph_before
    assert orchestrator.resolver.snapshot().concepts == resolver_before


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_good_doc_after_caught_failure_matches_run_without_bad_doc(on_error):
    orchestrator = build_orchestrator(on_error=on_error)
    orchestrator.process(make_document(id="d0", text="vector store", timestamp=1.0))
    good_inducer = orchestrator.relation_inducer
    orchestrator.relation_inducer = ExplodingRelationInducer()
    _run_bad_doc(
        orchestrator, make_document(id="bad", text="poison mention", timestamp=2.0)
    )
    orchestrator.relation_inducer = good_inducer
    delta = orchestrator.process(
        make_document(id="d2", text="vector encoder", timestamp=3.0)
    )
    assert delta.errors == ()

    clean = build_orchestrator(on_error=on_error)
    clean.process(make_document(id="d0", text="vector store", timestamp=1.0))
    clean.process(make_document(id="d2", text="vector encoder", timestamp=3.0))

    assert orchestrator.snapshot() == clean.snapshot()
    assert (
        orchestrator.resolver.snapshot().concepts
        == clean.resolver.snapshot().concepts
    )


class _BrokenRollbackIntegrator(InMemoryGraphIntegrator):
    def rollback(self, token):
        raise OSError("rollback-broke")


def test_rollback_failure_is_raised_chained_from_original():
    orchestrator = build_orchestrator(
        graph_integrator=_BrokenRollbackIntegrator(),
        relation_inducer=ExplodingRelationInducer(),
        on_error="skip",
    )
    with pytest.raises(OSError, match="rollback-broke") as info:
        orchestrator.process(make_document(id="d1", text="vector store"))
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "boom-in-relations" in str(info.value.__cause__)


# --- ADR-0002: rollback also runs on BaseException; policy only routes Exception ---


class _InterruptingRelationInducer(RelationInducer):
    """Raises KeyboardInterrupt after the resolver has already mutated."""

    def induce(self, resolutions, units, document):
        raise KeyboardInterrupt


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_keyboard_interrupt_propagates_and_rolls_back(on_error):
    orchestrator = build_orchestrator(
        relation_inducer=_InterruptingRelationInducer(), on_error=on_error
    )
    graph_before = orchestrator.snapshot()
    resolver_before = orchestrator.resolver.snapshot().concepts

    # never turned into a skip delta, even under "skip"
    with pytest.raises(KeyboardInterrupt):
        orchestrator.process(make_document(id="d1", text="new mention encoder"))

    assert orchestrator.snapshot() == graph_before
    assert orchestrator.resolver.snapshot().concepts == resolver_before


class _RecordingResolver(ExactLabelResolver):
    """Counts rollbacks; can fail its checkpoint or its rollback on demand."""

    def __init__(self, *args, fail_checkpoint=False, fail_rollback=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_checkpoint = fail_checkpoint
        self.fail_rollback = fail_rollback
        self.rollbacks = 0

    def checkpoint(self):
        if self.fail_checkpoint:
            raise RuntimeError("checkpoint-broke")
        return super().checkpoint()

    def rollback(self, token):
        self.rollbacks += 1
        if self.fail_rollback:
            raise ValueError("resolver-rollback-broke")
        return super().rollback(token)


def _recording_resolver(**kwargs):
    return _RecordingResolver(
        embedder=HashingEmbedder(dim=16), concept_store=InMemoryConceptStore(), **kwargs
    )


@pytest.mark.parametrize("on_error", ["fail", "skip"])
def test_failing_checkpoint_follows_policy_without_rollback(on_error):
    resolver = _recording_resolver(fail_checkpoint=True)
    orchestrator = build_orchestrator(resolver=resolver, on_error=on_error)
    graph_before = orchestrator.snapshot()
    resolver_before = resolver.snapshot().concepts

    delta = _run_bad_doc(orchestrator, make_document(id="d1", text="vector store"))

    if on_error == "skip":
        assert "checkpoint-broke" in delta.errors[0]
    assert resolver.rollbacks == 0
    assert orchestrator.snapshot() == graph_before
    assert resolver.snapshot().concepts == resolver_before


def test_resolver_rollback_failure_is_raised_chained_from_original():
    orchestrator = build_orchestrator(
        resolver=_recording_resolver(fail_rollback=True),
        relation_inducer=ExplodingRelationInducer(),
        on_error="skip",
    )
    with pytest.raises(ValueError, match="resolver-rollback-broke") as info:
        orchestrator.process(make_document(id="d1", text="vector store"))
    assert isinstance(info.value.__cause__, RuntimeError)


def test_second_rollback_failure_is_attached_as_note():
    orchestrator = build_orchestrator(
        resolver=_recording_resolver(fail_rollback=True),
        graph_integrator=_BrokenRollbackIntegrator(),
        relation_inducer=ExplodingRelationInducer(),
        on_error="fail",
    )
    with pytest.raises(ValueError, match="resolver-rollback-broke") as info:
        orchestrator.process(make_document(id="d1", text="vector store"))
    assert any("rollback-broke" in n for n in info.value.__notes__)
    assert any("OSError" in n for n in info.value.__notes__)


def test_rollback_failure_on_keyboard_interrupt_is_chained_from_it():
    orchestrator = build_orchestrator(
        graph_integrator=_BrokenRollbackIntegrator(),
        relation_inducer=_InterruptingRelationInducer(),
        on_error="skip",
    )
    with pytest.raises(OSError, match="rollback-broke") as info:
        orchestrator.process(make_document(id="d1", text="vector store"))
    assert isinstance(info.value.__cause__, KeyboardInterrupt)
