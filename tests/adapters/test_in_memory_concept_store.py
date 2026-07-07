from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from tests.contracts.concept_store_contract import ConceptStoreContract


class TestInMemoryConceptStore(ConceptStoreContract):
    def make_store(self) -> InMemoryConceptStore:
        return InMemoryConceptStore()
