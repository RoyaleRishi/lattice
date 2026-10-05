"""Contract every Resolver adapter must satisfy. The heart of lattice
(spec §1): the same surface in different documents must resolve to the SAME
concept, with identity provenance preserved."""

import pytest

from lattice.core.types import ResolverState
from lattice.ports import Resolver
from tests.helpers import make_document, make_scored_mention


class ResolverContract:
    def make_resolver(self) -> Resolver:
        raise NotImplementedError(
            "subclass must provide a fully wired adapter (own embedder + store)"
        )

    def test_new_surface_creates_new_concept(self):
        resolver = self.make_resolver()
        [resolution] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d1")
        )
        assert resolution.is_new
        assert resolution.concept.first_seen == "d1"
        assert resolution.concept.updated_at == "d1"

    def test_same_surface_across_documents_resolves_to_one_concept(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id == r1.concept.id
        assert r1.is_new and not r2.is_new
        assert r2.concept.first_seen == "d1"
        assert r2.concept.updated_at == "d2"

    def test_empty_input_yields_no_resolutions(self):
        assert self.make_resolver().resolve([], make_document(id="d1")) == []

    # --- lifecycle (ADR-0001 / ADR-0002) ---

    def _feed(self, resolver, doc_id, surfaces):
        mentions = [
            make_scored_mention(surface=s, unit_id=f"{doc_id}:u{i}")
            for i, s in enumerate(surfaces)
        ]
        return resolver.resolve(mentions, make_document(id=doc_id))

    def test_rollback_restores_pre_checkpoint_snapshot(self):
        resolver = self.make_resolver()
        self._feed(resolver, "d1", ["vector store", "encoder"])
        before = resolver.snapshot()
        token = resolver.checkpoint()
        # one existing surface (updated_at changes) + one brand-new surface
        self._feed(resolver, "d2", ["vector store", "ranking model"])
        assert resolver.snapshot() != before
        resolver.rollback(token)
        assert resolver.snapshot() == before

    def test_snapshot_is_sorted_by_id(self):
        resolver = self.make_resolver()
        self._feed(resolver, "d1", ["zebra", "apple", "mango"])
        ids = [c.id for c in resolver.snapshot().concepts]
        assert ids == sorted(ids)

    def test_snapshot_restore_round_trips_into_fresh_resolver(self):
        original = self.make_resolver()
        self._feed(original, "d1", ["vector store", "encoder"])
        state = original.snapshot()
        fresh = self.make_resolver()
        fresh.restore(state)
        assert fresh.snapshot() == state
        # same next-document results as the original
        a = self._feed(original, "d2", ["vector store", "new thing"])
        b = self._feed(fresh, "d2", ["vector store", "new thing"])
        assert a == b

    def test_restore_with_empty_private_works(self):
        original = self.make_resolver()
        self._feed(original, "d1", ["vector store", "encoder"])
        fresh = self.make_resolver()
        fresh.restore(ResolverState(original.snapshot().concepts, {}))
        assert fresh.snapshot().concepts == original.snapshot().concepts
        a = self._feed(original, "d2", ["vector store", "new thing"])
        b = self._feed(fresh, "d2", ["vector store", "new thing"])
        assert a == b

    def test_restore_replaces_existing_state(self):
        resolver = self.make_resolver()
        self._feed(resolver, "d1", ["leftover"])
        resolver.restore(ResolverState((), {}))
        assert resolver.snapshot().concepts == ()

    def test_reset_empties_state(self):
        resolver = self.make_resolver()
        self._feed(resolver, "d1", ["vector store"])
        resolver.reset()
        assert resolver.snapshot() == ResolverState((), resolver.snapshot().private)
        [r] = self._feed(resolver, "d2", ["vector store"])
        assert r.is_new

    def test_superseded_token_rollback_raises(self):
        resolver = self.make_resolver()
        stale = resolver.checkpoint()
        resolver.checkpoint()
        with pytest.raises(ValueError):
            resolver.rollback(stale)

    def test_double_rollback_raises(self):
        resolver = self.make_resolver()
        token = resolver.checkpoint()
        resolver.rollback(token)
        with pytest.raises(ValueError):
            resolver.rollback(token)

    def test_rollback_after_restore_raises(self):
        resolver = self.make_resolver()
        token = resolver.checkpoint()
        resolver.restore(resolver.snapshot())
        with pytest.raises(ValueError):
            resolver.rollback(token)

    def test_rollback_after_reset_raises(self):
        resolver = self.make_resolver()
        token = resolver.checkpoint()
        resolver.reset()
        with pytest.raises(ValueError):
            resolver.rollback(token)
