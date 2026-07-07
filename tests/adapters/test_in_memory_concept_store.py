from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.core.types import Concept
from tests.contracts.concept_store_contract import ConceptStoreContract


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
