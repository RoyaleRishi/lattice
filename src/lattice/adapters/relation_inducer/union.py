from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import lookup, register


@register(RelationInducer, "union")
class UnionInducer(RelationInducer):
    """Combinator (M4 spec §4.3): runs member inducers in order and
    concatenates their relations; the graph integrator dedupes by
    (type, source, target). Members are instantiated from the registry at
    construction time — an unknown name fails fast with RegistryError.
    Members needing shared-dep injection (embedder/concept_store) are out of
    scope: params are the only constructor arguments forwarded."""

    def __init__(self, members: list[dict]):
        self._members: list[RelationInducer] = []
        for spec in members:
            adapter_cls = lookup(RelationInducer, spec["name"])
            self._members.append(adapter_cls(**spec.get("params", {})))

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        relations: list[Relation] = []
        for member in self._members:
            relations.extend(member.induce(resolutions, units, document))
        return relations
