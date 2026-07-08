from lattice.core.types import Concept
from lattice.core.vectors import cosine
from lattice.ports import ConceptStore
from lattice.registry.registry import register


@register(ConceptStore, "in-memory")
class InMemoryConceptStore(ConceptStore):
    """Dict-backed store with brute-force cosine nearest-neighbour. Fine for
    experiments; a vector-index adapter can replace it behind the same port."""

    def __init__(self):
        self._by_id: dict[str, Concept] = {}
        self._id_by_label: dict[str, str] = {}

    def upsert(self, concept: Concept) -> None:
        old = self._by_id.get(concept.id)
        if old is not None:
            self._id_by_label.pop(old.label, None)
        self._by_id[concept.id] = concept
        self._id_by_label[concept.label] = concept.id

    def get(self, concept_id: str) -> Concept | None:
        return self._by_id.get(concept_id)

    def find_by_label(self, label: str) -> Concept | None:
        concept_id = self._id_by_label.get(label)
        return self._by_id.get(concept_id) if concept_id is not None else None

    def nearest(
        self, embedding: tuple[float, ...], k: int = 1
    ) -> list[tuple[Concept, float]]:
        scored = [
            (concept, cosine(embedding, concept.embedding))
            for concept in self._by_id.values()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]

    def all(self) -> list[Concept]:
        return list(self._by_id.values())

    def reset(self) -> None:
        self._by_id.clear()
        self._id_by_label.clear()
