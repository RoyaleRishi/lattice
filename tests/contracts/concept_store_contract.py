"""Contract every ConceptStore adapter must satisfy. The store is the
resolver's memory: identity must survive upserts, and reset() must fully
clear state between experiment runs (spec §4.2)."""

import pytest

from lattice.core.types import Concept
from lattice.ports import ConceptStore
from tests.helpers import make_concept


class ConceptStoreContract:
    def make_store(self) -> ConceptStore:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_upsert_then_get(self):
        store = self.make_store()
        concept = make_concept(id="c1", label="vector store")
        store.upsert(concept)
        assert store.get("c1") == concept

    def test_get_missing_returns_none(self):
        assert self.make_store().get("nope") is None

    def test_find_by_label(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="vector store"))
        found = store.find_by_label("vector store")
        assert found is not None and found.id == "c1"

    def test_upsert_same_id_replaces(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="old label"))
        store.upsert(make_concept(id="c1", label="new label"))
        assert store.get("c1").label == "new label"
        assert store.find_by_label("old label") is None
        assert len(store.all()) == 1

    def test_nearest_returns_most_similar_first(self):
        store = self.make_store()
        a = Concept(id="a", label="a", embedding=(1.0, 0.0), first_seen="d", updated_at="d")
        b = Concept(id="b", label="b", embedding=(0.0, 1.0), first_seen="d", updated_at="d")
        store.upsert(a)
        store.upsert(b)
        [(top, score)] = store.nearest((0.9, 0.1), k=1)
        assert top.id == "a"
        assert score > 0.9

    def test_reset_clears_everything(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="vector store"))
        store.reset()
        assert store.all() == []
        assert store.get("c1") is None
        assert store.find_by_label("vector store") is None

    def test_checkpoint_then_rollback_restores_original_contents(self):
        store = self.make_store()
        original = make_concept(id="c1", label="vector store")
        store.upsert(original)
        token = store.checkpoint()

        store.upsert(make_concept(id="c2", label="new concept"))
        store.rollback(token)

        assert store.all() == [original]
        assert store.get("c2") is None
        assert store.find_by_label("new concept") is None
        assert store.get("c1") == original

    def test_rollback_undoes_multiple_upserts_including_label_change(self):
        # ADR-0002: rollback must restore exact contents, label index included
        store = self.make_store()
        c1 = make_concept(id="c1", label="old label")
        c2 = make_concept(id="c2", label="other")
        store.upsert(c1)
        store.upsert(c2)
        token = store.checkpoint()

        store.upsert(make_concept(id="c1", label="new label"))
        store.upsert(make_concept(id="c1", label="newer label"))
        store.upsert(make_concept(id="c3", label="fresh"))
        store.rollback(token)

        assert sorted(store.all(), key=lambda c: c.id) == [c1, c2]
        assert store.get("c1") == c1
        assert store.get("c3") is None
        assert store.find_by_label("old label") == c1
        assert store.find_by_label("new label") is None
        assert store.find_by_label("newer label") is None
        assert store.find_by_label("fresh") is None

    def test_new_checkpoint_commits_previous_so_old_token_raises(self):
        store = self.make_store()
        old = store.checkpoint()
        store.upsert(make_concept(id="c1", label="a"))
        store.checkpoint()
        with pytest.raises(ValueError):
            store.rollback(old)
        # the failed rollback must not have touched state
        assert store.get("c1") is not None

    def test_double_rollback_raises(self):
        store = self.make_store()
        token = store.checkpoint()
        store.upsert(make_concept(id="c1", label="a"))
        store.rollback(token)
        with pytest.raises(ValueError):
            store.rollback(token)

    def test_changes_before_checkpoint_survive_rollback(self):
        store = self.make_store()
        before = make_concept(id="c1", label="kept")
        store.upsert(before)
        token = store.checkpoint()
        store.rollback(token)
        assert store.all() == [before]
        assert store.find_by_label("kept") == before

    def test_rollback_after_label_steal_restores_both_labels(self):
        # c1 takes c2's label, rollback must hand it back to c2
        store = self.make_store()
        c1 = make_concept(id="c1", label="a")
        c2 = make_concept(id="c2", label="b")
        store.upsert(c1)
        store.upsert(c2)
        token = store.checkpoint()

        store.upsert(make_concept(id="c1", label="b"))
        store.rollback(token)

        assert store.find_by_label("b") == c2
        assert store.find_by_label("a") == c1
        assert store.get("c1") == c1
        assert store.get("c2") == c2
