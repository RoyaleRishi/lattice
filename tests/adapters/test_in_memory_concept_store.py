from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.core.types import Concept
from tests.contracts.concept_store_contract import ConceptStoreContract
from tests.helpers import make_concept


class TestInMemoryConceptStore(ConceptStoreContract):
    def make_store(self) -> InMemoryConceptStore:
        return InMemoryConceptStore()

    def test_nearest_breaks_equal_similarity_ties_lexicographically_by_id(self):
        store = self.make_store()
        b = Concept(id="b", label="b", embedding=(1.0, 0.0), first_seen="d", updated_at="d")
        a = Concept(id="a", label="a", embedding=(1.0, 0.0), first_seen="d", updated_at="d")
        store.upsert(b)
        store.upsert(a)
        results = store.nearest((1.0, 0.0), k=2)
        assert [concept.id for concept, _ in results] == ["a", "b"]

    def test_undo_log_only_holds_touched_keys(self):
        # ADR-0002: checkpoint cost must scale with the doc's changes, not the store
        store = self.make_store()
        for i in range(200):
            store.upsert(make_concept(id=f"c{i}", label=f"label {i}"))
        token = store.checkpoint()
        store.upsert(make_concept(id="c0", label="label 0 renamed"))
        store.upsert(make_concept(id="c0", label="label 0 renamed again"))
        store.rollback(token)  # sanity: still works with a big store

        token = store.checkpoint()
        store.upsert(make_concept(id="c1", label="label 1"))
        # bounded by a small constant, nowhere near the 200 stored concepts
        assert 0 < store._undo_size() < 10
        store.rollback(token)
