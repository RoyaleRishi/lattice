"""Fixture factories shared by contract and adapter tests."""

from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    Mention,
    Resolution,
    ScoredMention,
    Unit,
)


def make_document(
    id: str = "doc-x", text: str = "some text", timestamp: float = 1.0
) -> Document:
    return Document(id=id, kind="note", text=text, timestamp=timestamp)


def make_unit(
    id: str = "doc-x:u0",
    document_id: str = "doc-x",
    text: str = "some text",
    order: int = 0,
) -> Unit:
    return Unit(id=id, document_id=document_id, text=text, order=order, kind="block")


def make_mention(
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    span: tuple[int, int] = (0, 7),
    context: str = "concept text",
) -> Mention:
    return Mention(
        surface=surface, unit_id=unit_id, span=span, context=context,
        head=surface, lemma=surface,
    )


def make_scored_mention(
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    salience: float = 1.0,
    selected: bool = True,
) -> ScoredMention:
    return ScoredMention(
        mention=make_mention(surface=surface, unit_id=unit_id),
        salience=salience,
        selected=selected,
    )


def make_concept(
    id: str = "c1", label: str = "concept", dim: int = 4, first_seen: str = "doc-x"
) -> Concept:
    return Concept(
        id=id,
        label=label,
        embedding=(1.0,) + (0.0,) * (dim - 1),
        first_seen=first_seen,
        updated_at=first_seen,
    )


def make_resolution(
    concept: Concept | None = None,
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    is_new: bool = True,
) -> Resolution:
    return Resolution(
        concept=concept or make_concept(id=f"c:{surface}", label=surface),
        mention=make_scored_mention(surface=surface, unit_id=unit_id),
        is_new=is_new,
    )


def make_delta(
    document_id: str = "d1",
    selected: list[tuple[str, float]] | None = None,
) -> GraphDelta:
    return GraphDelta(
        document_id=document_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=(),
        selected_mentions=tuple(
            make_scored_mention(surface=s, salience=sal) for s, sal in (selected or [])
        ),
    )
