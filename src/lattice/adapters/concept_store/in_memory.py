from lattice.core.types import Concept
from lattice.core.vectors import cosine
from lattice.ports import ConceptStore
from lattice.registry.registry import register

_ABSENT = object()  # sentinel: key didn't exist at checkpoint time


@register(ConceptStore, "in-memory")
class InMemoryConceptStore(ConceptStore):
    """Dict-backed store with brute-force cosine nearest-neighbour. Fine for
    experiments; a vector-index adapter can replace it behind the same port."""

    def __init__(self):
        self._by_id: dict[str, Concept] = {}
        self._id_by_label: dict[str, str] = {}
        # Undo log (ADR-0002): None = no open checkpoint. Each dict maps a key to
        # its value at checkpoint time, or _ABSENT; only first touch is recorded.
        self._token: object | None = None
        self._undo_by_id: dict[str, object] = {}
        self._undo_by_label: dict[str, object] = {}

    def _log(self, concept_id: str, labels: tuple[str, ...]) -> None:
        if self._token is None:
            return
        if concept_id not in self._undo_by_id:
            self._undo_by_id[concept_id] = self._by_id.get(concept_id, _ABSENT)
        for label in labels:
            if label not in self._undo_by_label:
                self._undo_by_label[label] = self._id_by_label.get(label, _ABSENT)

    def _undo_size(self) -> int:
        return len(self._undo_by_id) + len(self._undo_by_label)

    def upsert(self, concept: Concept) -> None:
        old = self._by_id.get(concept.id)
        # the old label gets popped below, so it has to be restorable too
        labels = (concept.label,) if old is None else (old.label, concept.label)
        self._log(concept.id, labels)
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
        self._close_log()

    def _close_log(self) -> None:
        self._token = None
        self._undo_by_id = {}
        self._undo_by_label = {}

    def checkpoint(self) -> object:
        # a fresh log implicitly commits whatever the previous one covered
        self._close_log()
        self._token = object()
        return self._token

    def rollback(self, token: object) -> None:
        if self._token is None or token is not self._token:
            raise ValueError("stale or unknown checkpoint token (ADR-0002)")
        # keys are independent per dict, so restore order doesn't matter
        for label, prior in reversed(list(self._undo_by_label.items())):
            if prior is _ABSENT:
                self._id_by_label.pop(label, None)
            else:
                self._id_by_label[label] = prior
        for concept_id, prior in reversed(list(self._undo_by_id.items())):
            if prior is _ABSENT:
                self._by_id.pop(concept_id, None)
            else:
                self._by_id[concept_id] = prior
        self._close_log()
