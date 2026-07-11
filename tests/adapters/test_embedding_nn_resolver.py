from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.embedding_nn import EmbeddingNNResolver
from lattice.ports import Embedder
from tests.contracts.resolver_contract import ResolverContract
from tests.helpers import make_document, make_scored_mention


class LookupEmbedder(Embedder):
    """Test double: fixed vector per exact text; `default` otherwise."""

    def __init__(self, mapping: dict[str, tuple[float, ...]], default: tuple[float, ...]):
        self.mapping = mapping
        self.default = default

    @property
    def dim(self) -> int:
        return len(self.default)

    def embed(self, texts):
        return [self.mapping.get(t, self.default) for t in texts]


def _resolver(threshold: float, mapping: dict | None = None) -> EmbeddingNNResolver:
    embedder = (
        LookupEmbedder(mapping, default=(0.0, 1.0)) if mapping is not None
        else HashingEmbedder(dim=16)
    )
    return EmbeddingNNResolver(
        embedder=embedder, concept_store=InMemoryConceptStore(), threshold=threshold
    )


class TestEmbeddingNNResolver(ResolverContract):
    def make_resolver(self) -> EmbeddingNNResolver:
        return EmbeddingNNResolver(
            embedder=HashingEmbedder(dim=16),
            concept_store=InMemoryConceptStore(),
            threshold=0.8,
        )

    def test_merges_exactly_at_threshold(self):
        # cos((1,0), (0.8,0.6)) = 0.8 exactly; threshold 0.8 must merge (>=).
        resolver = _resolver(0.8, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="alphaz")], make_document(id="d2"))
        assert not r2.is_new
        assert r2.concept.id == r1.concept.id
        assert r2.concept.label == "alpha"  # merged concept keeps its own label
        assert r2.concept.updated_at == "d2"

    def test_creates_just_above_threshold(self):
        resolver = _resolver(0.81, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="alphaz")], make_document(id="d2"))
        assert r2.is_new
        assert r2.concept.id != r1.concept.id

    def test_exact_label_short_circuits_regardless_of_threshold(self):
        # threshold 2.0 makes the NN path unreachable; identical strings must
        # still merge via find_by_label.
        resolver = _resolver(2.0)
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="Alpha ")], make_document(id="d2"))
        assert not r2.is_new
        assert r2.concept.id == r1.concept.id

    def test_stream_semantics_within_one_document(self):
        # Second mention merges into the concept created earlier in the SAME call.
        resolver = _resolver(0.8, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        resolutions = resolver.resolve(
            [make_scored_mention(surface="alpha"), make_scored_mention(surface="alphaz")],
            make_document(id="d1"),
        )
        assert [r.is_new for r in resolutions] == [True, False]
        assert resolutions[0].concept.id == resolutions[1].concept.id

    def test_empty_store_creates_first_concept(self):
        resolver = _resolver(0.0)  # even threshold 0 cannot merge into nothing
        [r] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        assert r.is_new
