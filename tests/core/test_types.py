import dataclasses

import pytest

from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Mention,
    Relation,
    Resolution,
    ScoredMention,
    Unit,
)


def test_document_construction_and_defaults():
    doc = Document(id="d1", kind="note", text="hello", timestamp=1.0)
    assert doc.metadata == {}


def test_document_is_immutable():
    doc = Document(id="d1", kind="note", text="hello", timestamp=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.id = "d2"


def test_unit_speaker_defaults_to_none():
    unit = Unit(id="d1:u0", document_id="d1", text="hello", order=0, kind="block")
    assert unit.speaker is None


def test_mention_span_is_tuple():
    m = Mention(surface="vector", unit_id="d1:u0", span=(4, 10), context="the vector store")
    assert m.span == (4, 10)
    assert m.head == "" and m.lemma == ""


def test_scored_mention_wraps_mention():
    m = Mention(surface="vector", unit_id="d1:u0", span=(0, 6), context="vector")
    sm = ScoredMention(mention=m, salience=0.5, selected=True)
    assert sm.mention.surface == "vector"


def test_concept_equality_is_value_based():
    a = Concept(id="c1", label="vector store", embedding=(1.0, 0.0), first_seen="d1", updated_at="d1")
    b = Concept(id="c1", label="vector store", embedding=(1.0, 0.0), first_seen="d1", updated_at="d1")
    assert a == b


def test_relation_fields():
    r = Relation(type="CO_OCCURS", source_id="c1", target_id="c2", confidence=1.0, provenance="d1")
    assert r.type == "CO_OCCURS"


def test_resolution_marks_new_vs_merged():
    c = Concept(id="c1", label="x", embedding=(1.0,), first_seen="d1", updated_at="d1")
    m = Mention(surface="x", unit_id="d1:u0", span=(0, 1), context="x")
    sm = ScoredMention(mention=m, salience=1.0, selected=True)
    assert Resolution(concept=c, mention=sm, is_new=True).is_new


def test_graph_delta_errors_default_empty():
    delta = GraphDelta(document_id="d1", concepts_added=(), concepts_updated=(), relations_added=())
    assert delta.errors == ()


def test_graph_snapshot_holds_tuples():
    snap = GraphSnapshot(concepts=(), relations=())
    assert snap.concepts == () and snap.relations == ()
