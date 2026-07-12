from collections.abc import Sequence

from lattice.core.types import Concept, Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register


@register(RelationInducer, "compound")
class CompoundInducer(RelationInducer):
    """Head-modifier IS_A induction (M4 spec §4.1): a multiword concept is a
    kind of its head — "olive oil" IS_A "oil" — whenever a word-suffix of its
    label is itself a concept resolved in the same document. On the taxonomy
    benchmark's glossary document every gold term is co-resident, so this
    computes the complete compound closure of the term list there. Suffixes
    are whole words, never substrings ("pineapple" is not an "apple")."""

    def __init__(self, longest_only: bool = True):
        self.longest_only = longest_only

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        by_label: dict[str, Concept] = {}
        for resolution in resolutions:
            by_label.setdefault(resolution.concept.label, resolution.concept)
        relations: list[Relation] = []
        for label, concept in sorted(by_label.items()):
            words = label.split()
            for i in range(1, len(words)):
                target = by_label.get(" ".join(words[i:]))
                if target is None:
                    continue
                relations.append(
                    Relation(
                        type="IS_A",
                        source_id=concept.id,
                        target_id=target.id,
                        confidence=1.0,
                        provenance=document.id,
                    )
                )
                if self.longest_only:
                    break
        return relations
