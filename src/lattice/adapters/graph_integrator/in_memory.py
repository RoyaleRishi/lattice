from collections.abc import Sequence

from lattice.core.types import Concept, GraphSnapshot, Relation, Resolution
from lattice.ports import GraphIntegrator
from lattice.registry.registry import register

_ABSENT = object()  # sentinel: key didn't exist at checkpoint time


@register(GraphIntegrator, "in-memory")
class InMemoryGraphIntegrator(GraphIntegrator):
    """Dict-backed accreting graph. Concepts are keyed by id (last write
    wins); relations by (type, source, target). Snapshots are sorted so
    identical runs produce identical snapshots (spec §7 reproducibility)."""

    def __init__(self):
        self._concepts: dict[str, Concept] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}
        # undo log (ADR-0002): None = no open checkpoint. Maps key -> prior
        # value, or _ABSENT if the key didn't exist at checkpoint time.
        self._token: object | None = None
        self._undo_concepts: dict[str, Concept | object] = {}
        self._undo_relations: dict[tuple[str, str, str], Relation | object] = {}

    def apply(
        self, resolutions: Sequence[Resolution], relations: Sequence[Relation]
    ) -> None:
        for resolution in resolutions:
            cid = resolution.concept.id
            if self._token is not None and cid not in self._undo_concepts:
                self._undo_concepts[cid] = self._concepts.get(cid, _ABSENT)
            self._concepts[cid] = resolution.concept
        for relation in relations:
            key = (relation.type, relation.source_id, relation.target_id)
            if self._token is not None and key not in self._undo_relations:
                self._undo_relations[key] = self._relations.get(key, _ABSENT)
            self._relations[key] = relation

    def snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            concepts=tuple(
                sorted(self._concepts.values(), key=lambda c: c.id)
            ),
            relations=tuple(
                sorted(
                    self._relations.values(),
                    key=lambda r: (r.type, r.source_id, r.target_id),
                )
            ),
        )

    def restore(self, snapshot: GraphSnapshot) -> None:
        self._close_log()
        self._concepts = {concept.id: concept for concept in snapshot.concepts}
        self._relations = {
            (r.type, r.source_id, r.target_id): r for r in snapshot.relations
        }

    def reset(self) -> None:
        self._close_log()
        self._concepts.clear()
        self._relations.clear()

    def checkpoint(self) -> object:
        # a new checkpoint commits (drops) whatever log was open before
        self._close_log()
        self._token = object()
        return self._token

    def rollback(self, token: object) -> None:
        if self._token is None or token is not self._token:
            raise ValueError("not the current open checkpoint (ADR-0002)")
        for cid, prior in self._undo_concepts.items():
            if prior is _ABSENT:
                self._concepts.pop(cid, None)
            else:
                self._concepts[cid] = prior
        for key, prior in self._undo_relations.items():
            if prior is _ABSENT:
                self._relations.pop(key, None)
            else:
                self._relations[key] = prior
        self._close_log()

    def _close_log(self) -> None:
        self._token = None
        self._undo_concepts = {}
        self._undo_relations = {}
