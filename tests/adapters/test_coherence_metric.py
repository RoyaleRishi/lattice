from collections.abc import Sequence

from lattice.adapters.document_metric.coherence import Coherence
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.core.types import GraphDelta
from tests.helpers import make_concept, make_resolution


class CountingEmbedder(HashingEmbedder):
    """Asserts the batched-embed contract: one call for the whole run."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        return super().embed(texts)


def _delta(document_id: str, resolutions) -> GraphDelta:
    return GraphDelta(
        document_id=document_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=(),
        resolutions=tuple(resolutions),
    )


def _resolutions(concept_id: str, surfaces: list[str]):
    concept = make_concept(id=concept_id, label=surfaces[0].casefold())
    return [make_resolution(concept=concept, surface=s) for s in surfaces]


def test_coherent_merge_scores_high_incoherent_low():
    # hashing-embedder cosines, measured: ("beatles","the beatles")=0.8386,
    # ("beatles","kid rock")=0.1336 — assert against safe bounds, not exact.
    embedder = HashingEmbedder()
    coherent = Coherence(embedder).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles", "the beatles"]))], {}
    )
    incoherent = Coherence(embedder).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles", "kid rock"]))], {}
    )
    assert coherent["coherence"] > 0.5
    assert incoherent["coherence"] < 0.5
    assert coherent["multi-surface-concepts"] == 1.0


def test_surfaces_dedupe_casefolded_within_a_concept():
    result = Coherence(HashingEmbedder()).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["Beatles", "beatles", "BEATLES"]))], {}
    )
    # one distinct surface -> not a multi-surface concept -> vacuous 1.0
    assert result["multi-surface-concepts"] == 0.0
    assert result["coherence"] == 1.0


def test_vacuous_coherence_is_one_with_zero_multi_surface():
    result = Coherence(HashingEmbedder()).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles"]))], {}
    )
    assert result["coherence"] == 1.0
    assert result["multi-surface-concepts"] == 0.0


def test_singleton_fraction():
    deltas = [
        _delta("d1", _resolutions("c1", ["beatles", "the beatles"])),
        _delta("d2", _resolutions("c2", ["kid rock"])),
    ]
    result = Coherence(HashingEmbedder()).evaluate_documents(deltas, {})
    # c1 has 2 resolutions, c2 has 1 -> half the concepts are singletons
    assert result["singleton-fraction"] == 0.5


def test_grouping_spans_documents():
    concept = make_concept(id="c1", label="beatles")
    deltas = [
        _delta("d1", [make_resolution(concept=concept, surface="beatles")]),
        _delta("d2", [make_resolution(concept=concept, surface="the beatles")]),
    ]
    result = Coherence(HashingEmbedder()).evaluate_documents(deltas, {})
    assert result["multi-surface-concepts"] == 1.0
    assert result["singleton-fraction"] == 0.0


def test_single_batched_embed_call():
    embedder = CountingEmbedder()
    deltas = [
        _delta("d1", _resolutions("c1", ["beatles", "the beatles"])),
        _delta("d2", _resolutions("c2", ["kid rock", "kid rock songs"])),
    ]
    Coherence(embedder).evaluate_documents(deltas, {})
    assert embedder.calls == 1


def test_no_deltas_is_vacuous():
    result = Coherence(HashingEmbedder()).evaluate_documents([], {})
    assert result == {
        "coherence": 1.0,
        "multi-surface-concepts": 0.0,
        "singleton-fraction": 0.0,
    }
