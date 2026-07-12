from lattice.adapters.relation_inducer.compound import CompoundInducer
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestCompoundContract(RelationInducerContract):
    def make_inducer(self):
        return CompoundInducer()


def _induce(surfaces: list[str], longest_only: bool = True):
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text=" ".join(surfaces))]
    resolutions = [make_resolution(surface=s, unit_id="d1:u0") for s in surfaces]
    inducer = CompoundInducer(longest_only=longest_only)
    relations = inducer.induce(resolutions, units, document)
    label = {r.concept.id: r.concept.label for r in resolutions}
    return [(label[rel.source_id], label[rel.target_id], rel) for rel in relations]


def test_multiword_label_links_to_its_head_suffix():
    edges = _induce(["olive oil", "oil"])
    assert [(s, t) for s, t, _ in edges] == [("olive oil", "oil")]
    relation = edges[0][2]
    assert relation.type == "IS_A"
    assert relation.confidence == 1.0
    assert relation.provenance == "d1"


def test_longest_matching_suffix_wins_by_default():
    edges = _induce(["extra virgin olive oil", "olive oil", "oil"])
    pairs = {(s, t) for s, t, _ in edges}
    # the compound links only to its longest matching suffix; the shorter
    # compound links to its own head.
    assert pairs == {
        ("extra virgin olive oil", "olive oil"),
        ("olive oil", "oil"),
    }


def test_longest_only_false_emits_every_matching_suffix():
    edges = _induce(["extra virgin olive oil", "olive oil", "oil"],
                    longest_only=False)
    pairs = {(s, t) for s, t, _ in edges}
    assert pairs == {
        ("extra virgin olive oil", "olive oil"),
        ("extra virgin olive oil", "oil"),
        ("olive oil", "oil"),
    }


def test_suffix_must_be_whole_word():
    assert _induce(["pineapple", "apple"]) == []


def test_single_word_labels_are_inert():
    assert _induce(["oil", "fat"]) == []


def test_missing_head_yields_no_edge():
    assert _induce(["olive oil", "fat"]) == []


def test_duplicate_resolutions_of_one_concept_emit_one_edge():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="olive oil oil olive oil")]
    resolutions = [
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
        make_resolution(surface="olive oil", unit_id="d1:u0"),
    ]
    relations = CompoundInducer().induce(resolutions, units, document)
    assert len(relations) == 1
