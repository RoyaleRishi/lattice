from lattice.core.types import Concept, GraphSnapshot, Relation
from lattice.graph_view import GraphView


def _concept(cid: str, label: str) -> Concept:
    return Concept(
        id=cid, label=label, embedding=(1.0, 0.0),
        first_seen="d1", updated_at="d1",
    )


def _relation(type: str, source: str, target: str) -> Relation:
    return Relation(
        type=type, source_id=source, target_id=target,
        confidence=1.0, provenance="d1",
    )


SNAPSHOT = GraphSnapshot(
    concepts=(
        _concept("c:fat", "fat"),
        _concept("c:oil", "oil"),
        _concept("c:olive oil", "olive oil"),
    ),
    relations=(
        _relation("IS_A", "c:olive oil", "c:oil"),
        _relation("CO_OCCURS", "c:fat", "c:olive oil"),
    ),
)


def test_concepts_passthrough():
    assert GraphView(SNAPSHOT).concepts() == SNAPSHOT.concepts


def test_find_concept_is_casefolded_exact_match():
    view = GraphView(SNAPSHOT)
    assert view.find_concept("olive oil").id == "c:olive oil"
    assert view.find_concept("OLIVE OIL").id == "c:olive oil"
    assert view.find_concept("olive") is None


def test_relations_type_filter():
    view = GraphView(SNAPSHOT)
    assert view.relations() == SNAPSHOT.relations
    assert [r.type for r in view.relations("IS_A")] == ["IS_A"]
    assert view.relations("NOPE") == ()


def test_neighbors_returns_relation_and_other_concept():
    view = GraphView(SNAPSHOT)
    pairs = view.neighbors("c:olive oil")
    # sorted by (relation.type, other.id): CO_OCCURS/fat before IS_A/oil
    assert [(r.type, c.id) for r, c in pairs] == [
        ("CO_OCCURS", "c:fat"),
        ("IS_A", "c:oil"),
    ]
    # direction is readable from the relation itself
    is_a = pairs[1][0]
    assert is_a.source_id == "c:olive oil" and is_a.target_id == "c:oil"


def test_neighbors_type_filter_and_unknown_id():
    view = GraphView(SNAPSHOT)
    assert [(r.type, c.id) for r, c in view.neighbors("c:olive oil", "IS_A")] == [
        ("IS_A", "c:oil")
    ]
    assert view.neighbors("c:absent") == ()


def test_view_is_stable_across_repeated_queries():
    view = GraphView(SNAPSHOT)
    assert view.neighbors("c:olive oil") == view.neighbors("c:olive oil")
    assert view.find_concept("fat") == view.find_concept("fat")
