from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestCoOccurrenceInducer(RelationInducerContract):
    def make_inducer(self) -> CoOccurrenceInducer:
        return CoOccurrenceInducer()

    def test_same_unit_concepts_co_occur(self):
        relations = CoOccurrenceInducer().induce(
            [
                make_resolution(surface="vector store", unit_id="d1:u0"),
                make_resolution(surface="encoder", unit_id="d1:u0"),
            ],
            [make_unit(id="d1:u0", document_id="d1")],
            make_document(id="d1"),
        )
        assert len(relations) == 1
        assert relations[0].type == "CO_OCCURS"

    def test_cross_unit_concepts_do_not_co_occur(self):
        relations = CoOccurrenceInducer().induce(
            [
                make_resolution(surface="vector store", unit_id="d1:u0"),
                make_resolution(surface="encoder", unit_id="d1:u1"),
            ],
            [
                make_unit(id="d1:u0", document_id="d1"),
                make_unit(id="d1:u1", document_id="d1", order=1),
            ],
            make_document(id="d1"),
        )
        assert relations == []

    def test_pair_emitted_once_with_sorted_endpoints(self):
        resolutions = [
            make_resolution(surface="b-concept", unit_id="d1:u0"),
            make_resolution(surface="a-concept", unit_id="d1:u0"),
            make_resolution(surface="b-concept", unit_id="d1:u0"),
        ]
        relations = CoOccurrenceInducer().induce(
            resolutions, [make_unit(id="d1:u0", document_id="d1")], make_document(id="d1")
        )
        assert len(relations) == 1
        assert relations[0].source_id < relations[0].target_id

    def test_same_concept_twice_yields_no_self_relation(self):
        resolutions = [
            make_resolution(surface="vector store", unit_id="d1:u0"),
            make_resolution(surface="vector store", unit_id="d1:u0"),
        ]
        relations = CoOccurrenceInducer().induce(
            resolutions, [make_unit(id="d1:u0", document_id="d1")], make_document(id="d1")
        )
        assert relations == []
