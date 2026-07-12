import re
from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register

# Separators accepted between coordinated hyponyms ("x1, x2 and x3").
_COORD = re.compile(r",\s+|,?\s+(?:and|or)\s+", re.IGNORECASE)


class _Pattern:
    """One lexico-syntactic pattern: `connector` is fullmatched against the
    text between two adjacent anchor spans; `hyper` says which anchor is the
    hypernym; `coordination` enables forward hyponym-list walking; `prefix`
    (if set) must match immediately before the left anchor."""

    __slots__ = ("name", "connector", "hyper", "coordination", "prefix")

    def __init__(
        self,
        name: str,
        connector: str,
        hyper: str,
        coordination: bool = False,
        prefix: str | None = None,
    ):
        self.name = name
        self.connector = re.compile(connector, re.IGNORECASE)
        self.hyper = hyper  # "left" | "right"
        self.coordination = coordination
        self.prefix = re.compile(prefix, re.IGNORECASE) if prefix else None


_BUILTIN = [
    _Pattern("such-as", r",?\s+such\s+as\s+", hyper="left", coordination=True),
    _Pattern(
        "such-np-as", r"\s+as\s+", hyper="left", coordination=True,
        prefix=r"\bsuch\s+\Z",
    ),
    _Pattern("including", r",?\s+including\s+", hyper="left", coordination=True),
    _Pattern("especially", r",?\s+especially\s+", hyper="left", coordination=True),
    _Pattern("and-other", r",?\s+and\s+other\s+", hyper="right"),
    _Pattern("or-other", r",?\s+or\s+other\s+", hyper="right"),
    _Pattern(
        "copula", r"\s+(?:is|are)\s+an?\s+(?:(?:kind|type)\s+of\s+)?",
        hyper="right",
    ),
]


@register(RelationInducer, "hearst")
class HearstInducer(RelationInducer):
    """Anchored Hearst-pattern IS_A induction (M4 spec §4.2). Resolved
    mention spans are the NPs — no chunking; only adjacent anchors in a unit
    are paired, and the connector between them must match a pattern exactly,
    so locality needs no sentence segmentation (a cross-sentence connector
    like ". Such as " never fullmatches). `patterns` selects a subset by
    name; when it is None the full set is used, minus `copula` if
    copula=False (definitional corpora make the copula the productive
    pattern, hence on by default)."""

    def __init__(self, patterns: list[str] | None = None, copula: bool = True):
        available = {p.name: p for p in _BUILTIN}
        if patterns is None:
            selected = [
                p for p in _BUILTIN if copula or p.name != "copula"
            ]
        else:
            unknown = sorted(set(patterns) - set(available))
            if unknown:
                raise ValueError(f"unknown hearst pattern(s): {unknown}")
            selected = [available[name] for name in patterns]
        self._patterns = selected

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        text_of = {u.id: u.text for u in units}
        by_unit: dict[str, list[Resolution]] = {}
        for resolution in resolutions:
            unit_id = resolution.mention.mention.unit_id
            by_unit.setdefault(unit_id, []).append(resolution)
        relations: list[Relation] = []
        seen: set[tuple[str, str]] = set()
        for unit_id in sorted(by_unit):
            text = text_of.get(unit_id)
            if text is None:
                continue
            anchors = sorted(by_unit[unit_id], key=lambda r: r.mention.mention.span)
            for i in range(len(anchors) - 1):
                left, right = anchors[i], anchors[i + 1]
                left_end = left.mention.mention.span[1]
                right_start = right.mention.mention.span[0]
                if right_start < left_end:
                    continue  # overlapping anchors: no clean connector
                between = text[left_end:right_start]
                for pattern in self._patterns:
                    if not pattern.connector.fullmatch(between):
                        continue
                    left_start = left.mention.mention.span[0]
                    if pattern.prefix and not pattern.prefix.search(
                        text[:left_start]
                    ):
                        continue
                    if pattern.hyper == "left":
                        pairs = [(right, left)]
                        if pattern.coordination:
                            pairs += _walk_coordination(anchors, i + 1, left, text)
                    else:
                        pairs = [(left, right)]
                    for hypo, hyper in pairs:
                        edge = (hypo.concept.id, hyper.concept.id)
                        if edge[0] == edge[1] or edge in seen:
                            continue
                        seen.add(edge)
                        relations.append(
                            Relation(
                                type="IS_A",
                                source_id=edge[0],
                                target_id=edge[1],
                                confidence=1.0,
                                provenance=document.id,
                            )
                        )
        return relations


def _walk_coordination(
    anchors: list[Resolution], first_hypo: int, hyper: Resolution, text: str
) -> list[tuple[Resolution, Resolution]]:
    """After `hyper CONN anchors[first_hypo]`, keep consuming following
    anchors while they are separated only by coordination glue."""
    pairs: list[tuple[Resolution, Resolution]] = []
    for j in range(first_hypo, len(anchors) - 1):
        previous_end = anchors[j].mention.mention.span[1]
        next_start = anchors[j + 1].mention.mention.span[0]
        if next_start < previous_end:
            break
        if not _COORD.fullmatch(text[previous_end:next_start]):
            break
        pairs.append((anchors[j + 1], hyper))
    return pairs
