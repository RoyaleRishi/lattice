"""Domain contracts (spec §5). Pure data types — stdlib only, zero external deps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """One unit of input text: a note, a transcript, any document."""

    id: str
    kind: str  # e.g. "note", "transcript"
    text: str
    timestamp: float  # stream ordering
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Unit:
    """An ordered segment of a document (turn, block, or sentence)."""

    id: str
    document_id: str
    text: str
    order: int
    kind: str  # "turn" | "block" | "sentence"
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class Mention:
    """A candidate concept occurrence inside a unit. `span` is (start, end)
    character offsets into the unit's text."""

    surface: str
    unit_id: str
    span: tuple[int, int]
    context: str
    head: str = ""
    lemma: str = ""


@dataclass(frozen=True, slots=True)
class ScoredMention:
    """A mention with its salience score and whether the scorer selected it."""

    mention: Mention
    salience: float
    selected: bool


@dataclass(frozen=True, slots=True)
class Concept:
    """A canonical node in the concept graph. `first_seen`/`updated_at` are
    document ids (stream provenance, not wall-clock)."""

    id: str
    label: str
    embedding: tuple[float, ...]
    first_seen: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class Relation:
    """A typed edge between two concepts. `provenance` is the document id
    that evidenced the relation."""

    type: str
    source_id: str
    target_id: str
    confidence: float
    provenance: str


@dataclass(frozen=True, slots=True)
class Resolution:
    """The resolver's verdict for one selected mention: the canonical concept
    it maps to, and whether that concept was newly created (is_new=True) or
    merged into an existing one (is_new=False)."""

    concept: Concept
    mention: ScoredMention
    is_new: bool


@dataclass(frozen=True, slots=True)
class GraphDelta:
    """What one document changed in the graph. Errors are always recorded
    here, never silently dropped (spec §8). `selected_mentions` is the
    scorer's selected output for this document (pre-resolver), the unit of
    per-document evaluation (M2 spec §4)."""

    document_id: str
    concepts_added: tuple[Concept, ...]
    concepts_updated: tuple[Concept, ...]
    relations_added: tuple[Relation, ...]
    errors: tuple[str, ...] = ()
    selected_mentions: tuple[ScoredMention, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """An immutable point-in-time view of the accreting graph (spec §4.2)."""

    concepts: tuple[Concept, ...]
    relations: tuple[Relation, ...]
