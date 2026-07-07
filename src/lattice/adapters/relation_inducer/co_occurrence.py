from collections.abc import Sequence
from itertools import combinations

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register


@register(RelationInducer, "co-occurrence")
class CoOccurrenceInducer(RelationInducer):
    """Trivial walking-skeleton inducer: one CO_OCCURS relation per unordered
    pair of distinct concepts mentioned in the same unit. Hearst-pattern and
    head-modifier IS_A inducers arrive in Milestone 4 behind the same port."""

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        concepts_by_unit: dict[str, set[str]] = {}
        for resolution in resolutions:
            unit_id = resolution.mention.mention.unit_id
            concepts_by_unit.setdefault(unit_id, set()).add(resolution.concept.id)
        pairs: set[tuple[str, str]] = set()
        for concept_ids in concepts_by_unit.values():
            pairs.update(combinations(sorted(concept_ids), 2))
        return [
            Relation(
                type="CO_OCCURS",
                source_id=source_id,
                target_id=target_id,
                confidence=1.0,
                provenance=document.id,
            )
            for source_id, target_id in sorted(pairs)
        ]
