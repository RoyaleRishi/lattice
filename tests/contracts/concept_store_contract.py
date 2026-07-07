"""Contract every ConceptStore adapter must satisfy. The store is the
resolver's memory: identity must survive upserts, and reset() must fully
clear state between experiment runs (spec §4.2)."""

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
