import pytest

from lattice.adapters.relation_inducer.hearst import HearstInducer
from lattice.core.types import Mention, Resolution, ScoredMention
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_concept, make_document, make_unit


class TestHearstContract(RelationInducerContract):
    def make_inducer(self):
        return HearstInducer()


def _resolution(surface: str, unit_id: str, start: int) -> Resolution:
    mention = Mention(
        surface=surface,
        unit_id=unit_id,
        span=(start, start + len(surface)),
        context=surface,
    )
    return Resolution(
        concept=make_concept(id=f"c:{surface.lower()}", label=surface.lower()),
        mention=ScoredMention(mention=mention, salience=1.0, selected=True),
        is_new=True,
    )


def _edges(text: str, surfaces: list[str], **kwargs) -> set[tuple[str, str]]:
    """Resolve each surface at its first occurrence in `text`, run the
    inducer, and return (hyponym label, hypernym label) pairs."""
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text=text)]
    resolutions = []
    cursor: dict[str, int] = {}
    for surface in surfaces:
        start = text.index(surface, cursor.get(surface, 0))
        cursor[surface] = start + 1
        resolutions.append(_resolution(surface, "d1:u0", start))
    relations = HearstInducer(**kwargs).induce(resolutions, units, document)
    label = {r.concept.id: r.concept.label for r in resolutions}
    return {(label[rel.source_id], label[rel.target_id]) for rel in relations}


def test_such_as():
    assert _edges("fats such as olive oil are prized.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_such_as_with_comma():
    assert _edges("fats, such as olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_such_np_as():
    assert _edges("such fats as olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_bare_as_without_such_prefix_does_not_match():
    assert _edges("fats as olive oil.", ["fats", "olive oil"]) == set()


def test_including():
    assert _edges("fats, including olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_especially():
    assert _edges("fats, especially olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_and_other():
    assert _edges("olive oil and other fats.", ["olive oil", "fats"]) == {
        ("olive oil", "fats")
    }


def test_or_other():
    assert _edges("olive oil or other fats.", ["olive oil", "fats"]) == {
        ("olive oil", "fats")
    }


def test_copula_variants():
    for text in [
        "olive oil is a fat.",
        "olive oil is an fat.",
        "olive oil is a kind of fat.",
        "olive oil is a type of fat.",
    ]:
        assert _edges(text, ["olive oil", "fat"]) == {("olive oil", "fat")}, text


def test_copula_flag_off_drops_copula_edges():
    assert _edges("olive oil is a fat.", ["olive oil", "fat"], copula=False) == set()


def test_explicit_patterns_select_a_subset():
    # NOTE: deviation from the plan's fixture text — the plan's original
    # ("fats such as olive oil. canola is a fat.") anchors "fat" inside
    # "fats" at index 0, so the overlap guard eats the copula pair. Same
    # intent: copula-only selection finds the copula edge and excludes the
    # such-as edge (which the full pattern set does find on this text).
    text = "canola is a fat. fats such as olive oil."
    surfaces = ["canola", "fat", "fats", "olive oil"]
    assert _edges(text, surfaces, patterns=["copula"]) == {("canola", "fat")}


def test_unknown_pattern_name_raises():
    with pytest.raises(ValueError, match="unknown hearst pattern"):
        HearstInducer(patterns=["cherry-picked"])


def test_coordination_walking():
    text = "fats, such as olive oil, canola and margarine, are prized."
    surfaces = ["fats", "olive oil", "canola", "margarine"]
    assert _edges(text, surfaces) == {
        ("olive oil", "fats"),
        ("canola", "fats"),
        ("margarine", "fats"),
    }


def test_intervening_text_kills_the_match():
    assert _edges(
        "fats are found in stores such as delis, olive oil.", ["fats", "olive oil"]
    ) == set()


def test_cross_sentence_connector_fails():
    assert _edges("we saw fats. Such as olive oil.", ["fats", "olive oil"]) == set()


def test_same_concept_pair_is_skipped():
    # both surfaces resolve to the same lowercased concept
    assert _edges("Fat is a fat.", ["Fat", "fat"]) == set()


def test_cross_unit_pairs_never_match():
    document = make_document(id="d1")
    units = [
        make_unit(id="d1:u0", document_id="d1", text="fats such as"),
        make_unit(id="d1:u1", document_id="d1", text="olive oil", order=1),
    ]
    resolutions = [
        _resolution("fats", "d1:u0", 0),
        _resolution("olive oil", "d1:u1", 0),
    ]
    assert HearstInducer().induce(resolutions, units, document) == []


def test_relation_shape():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="fats such as olive oil")]
    resolutions = [_resolution("fats", "d1:u0", 0), _resolution("olive oil", "d1:u0", 13)]
    [relation] = HearstInducer().induce(resolutions, units, document)
    assert relation.type == "IS_A"
    assert relation.confidence == 1.0
    assert relation.provenance == "d1"
