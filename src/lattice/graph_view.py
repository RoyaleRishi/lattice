from lattice.core.types import Concept, GraphSnapshot, Relation


class GraphView:
    """Read-optimized view over one immutable GraphSnapshot (M6 spec §4.2).
    Indexes build lazily on first use. A view reflects the snapshot it was
    made from — take a fresh view after new ingests. Labels are stored
    lowercased by the resolvers; lookups casefold both sides."""

    def __init__(self, snapshot: GraphSnapshot):
        self._snapshot = snapshot
        self._by_label: dict[str, Concept] | None = None
        self._by_id: dict[str, Concept] | None = None
        self._adjacency: dict[str, list[Relation]] | None = None

    def concepts(self) -> tuple[Concept, ...]:
        return self._snapshot.concepts

    def find_concept(self, label: str) -> Concept | None:
        if self._by_label is None:
            self._by_label = {
                concept.label.casefold(): concept
                for concept in self._snapshot.concepts
            }
        return self._by_label.get(label.casefold())

    def relations(self, type: str | None = None) -> tuple[Relation, ...]:
        if type is None:
            return self._snapshot.relations
        return tuple(r for r in self._snapshot.relations if r.type == type)

    def neighbors(
        self, concept_id: str, type: str | None = None
    ) -> tuple[tuple[Relation, Concept], ...]:
        if self._adjacency is None:
            self._adjacency = {}
            for relation in self._snapshot.relations:
                self._adjacency.setdefault(relation.source_id, []).append(relation)
                if relation.target_id != relation.source_id:
                    self._adjacency.setdefault(relation.target_id, []).append(relation)
        if self._by_id is None:
            self._by_id = {c.id: c for c in self._snapshot.concepts}
        pairs: list[tuple[Relation, Concept]] = []
        for relation in self._adjacency.get(concept_id, ()):
            if type is not None and relation.type != type:
                continue
            other_id = (
                relation.target_id
                if relation.source_id == concept_id
                else relation.source_id
            )
            other = self._by_id.get(other_id)
            if other is not None:
                pairs.append((relation, other))
        pairs.sort(key=lambda pair: (pair[0].type, pair[1].id, pair[0].source_id))
        return tuple(pairs)
