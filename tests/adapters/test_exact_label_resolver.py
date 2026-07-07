from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from tests.contracts.resolver_contract import ResolverContract
from tests.helpers import make_document, make_scored_mention


class TestExactLabelResolver(ResolverContract):
    def make_resolver(self) -> ExactLabelResolver:
        return ExactLabelResolver(
            embedder=HashingEmbedder(dim=16),
            concept_store=InMemoryConceptStore(),
        )

    def test_distinct_surfaces_create_distinct_concepts(self):
        resolver = self.make_resolver()
        r1, r2 = resolver.resolve(
            [
                make_scored_mention(surface="vector store"),
                make_scored_mention(surface="encoder"),
            ],
            make_document(id="d1"),
        )
        assert r1.concept.id != r2.concept.id

    def test_labels_are_normalized_lowercase(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="Vector Store")], make_document(id="d1")
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d2")
        )
        assert r1.concept.id == r2.concept.id
        assert r1.concept.label == "vector store"

    def test_concept_gets_embedding_from_embedder(self):
        resolver = self.make_resolver()
        [r] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d1")
        )
        assert len(r.concept.embedding) == 16

    def test_mixed_case_and_whitespace_surfaces_dedupe_within_one_call(self):
        # exact_label.py normalizes surfaces with `.strip().lower()`, so
        # both case and surrounding whitespace are folded together.
        resolver = self.make_resolver()
        r1, r2 = resolver.resolve(
            [
                make_scored_mention(surface="Vector", unit_id="d1:u0"),
                make_scored_mention(surface=" vector ", unit_id="d1:u1"),
            ],
            make_document(id="d1"),
        )
        assert r1.concept.id == r2.concept.id
        assert r1.is_new and not r2.is_new
