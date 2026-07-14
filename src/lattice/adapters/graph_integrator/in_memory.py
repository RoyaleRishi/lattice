from collections.abc import Sequence

from lattice.core.types import Concept, GraphSnapshot, Relation, Resolution
from lattice.ports import GraphIntegrator
from lattice.registry.registry import register


@register(GraphIntegrator, "in-memory")
class InMemoryGraphIntegrator(GraphIntegrator):
    """Dict-backed accreting graph. Concepts are keyed by id (last write
    wins); relations by (type, source, target). Snapshots are sorted so
    identical runs produce identical snapshots (spec §7 reproducibility)."""

    def __init__(self):
        self._concepts: dict[str, Concept] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}

    def apply(
        self, resolutions: Sequence[Resolution], relations: Sequence[Relation]
    ) -> None:
        for resolution in resolutions:
            self._concepts[resolution.concept.id] = resolution.concept
        for relation in relations:
            key = (relation.type, relation.source_id, relation.target_id)
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
        self._concepts = {concept.id: concept for concept in snapshot.concepts}
        self._relations = {
            (r.type, r.source_id, r.target_id): r for r in snapshot.relations
        }

    def reset(self) -> None:
        self._concepts.clear()
        self._relations.clear()
