from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.stemmed_label import StemmedLabelResolver
from tests.contracts.resolver_contract import ResolverContract
from tests.helpers import make_document, make_scored_mention


class TestStemmedLabelResolver(ResolverContract):
    def make_resolver(self) -> StemmedLabelResolver:
        return StemmedLabelResolver(
            embedder=HashingEmbedder(dim=16),
            concept_store=InMemoryConceptStore(),
        )

    def test_morphological_variants_merge_via_stem(self):
        # 'oceans' / 'ocean' is one of the 7 merges embedding-nn made beyond
        # exact-label on ConEL-2; Snowball reproduces it directly.
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="oceans", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="ocean", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id == r1.concept.id
        assert r1.is_new and not r2.is_new

    def test_british_american_spelling_does_not_merge(self):
        # Snowball does not normalize British/American spelling -- this is
        # the known 1-of-7 gap and must not be overclaimed as a merge.
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="favourite colour", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="favorite color", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id != r1.concept.id
        assert r1.is_new and r2.is_new

    def test_stored_label_is_first_surface_seen_not_the_stem(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="vegetables", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vegetable", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id == r1.concept.id
        assert r1.concept.label == "vegetables"
        assert r2.concept.label == "vegetables"

    def test_distinct_concepts_stay_distinct(self):
        resolver = self.make_resolver()
        r1, r2 = resolver.resolve(
            [
                make_scored_mention(surface="vector store", unit_id="d1:u0"),
                make_scored_mention(surface="vector database", unit_id="d1:u1"),
            ],
            make_document(id="d1"),
        )
        assert r1.concept.id != r2.concept.id
