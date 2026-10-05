from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.stemmed_label import StemmedLabelResolver
from lattice.core.types import ResolverState
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

    # --- lifecycle: the stem cache is covered (ADR-0001 / ADR-0002) ---

    def test_cache_entries_added_after_checkpoint_are_gone_after_rollback(self):
        resolver = self.make_resolver()
        resolver.resolve(
            [make_scored_mention(surface="oceans", unit_id="d1:u0")], make_document(id="d1")
        )
        before = resolver.snapshot().private
        token = resolver.checkpoint()
        resolver.resolve(
            [make_scored_mention(surface="encoders", unit_id="d2:u0")], make_document(id="d2")
        )
        assert resolver.snapshot().private != before
        resolver.rollback(token)
        assert resolver.snapshot().private == before

    def test_private_round_trips_through_snapshot_restore(self):
        original = self.make_resolver()
        original.resolve(
            [make_scored_mention(surface="oceans", unit_id="d1:u0")], make_document(id="d1")
        )
        state = original.snapshot()
        assert state.private["concept_id_by_stem"]
        fresh = self.make_resolver()
        fresh.restore(state)
        assert fresh.snapshot() == state

    def test_restore_with_missing_cache_key_gives_empty_cache(self):
        resolver = self.make_resolver()
        resolver.restore(ResolverState((), {}))
        assert resolver.snapshot().private == {"concept_id_by_stem": {}}

    def test_reset_clears_cache(self):
        resolver = self.make_resolver()
        resolver.resolve(
            [make_scored_mention(surface="oceans", unit_id="d1:u0")], make_document(id="d1")
        )
        resolver.reset()
        assert resolver.snapshot().private == {"concept_id_by_stem": {}}
