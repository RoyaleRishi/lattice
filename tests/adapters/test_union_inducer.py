import pytest

from lattice.adapters.relation_inducer.union import UnionInducer
from lattice.registry.registry import RegistryError
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestUnionContract(RelationInducerContract):
    def make_inducer(self):
        return UnionInducer(
            members=[{"name": "hearst"}, {"name": "compound"}]
        )


def test_union_concatenates_member_outputs_in_member_order():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="olive oil oil")]
    resolutions = [
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
    ]
    # compound alone finds the edge; hearst finds nothing (fake spans overlap)
    inducer = UnionInducer(members=[{"name": "hearst"}, {"name": "compound"}])
    relations = inducer.induce(resolutions, units, document)
    assert len(relations) == 1
    assert relations[0].type == "IS_A"


def test_member_params_are_forwarded():
    inducer = UnionInducer(
        members=[{"name": "compound", "params": {"longest_only": False}}]
    )
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="x")]
    resolutions = [
        make_resolution(surface="extra virgin olive oil", unit_id="d1:u0"),
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
    ]
    # longest_only=False emits every matching suffix: 3 edges, not 2
    assert len(inducer.induce(resolutions, units, document)) == 3


def test_unknown_member_name_fails_at_construction():
    with pytest.raises(RegistryError, match="no adapter 'nope'"):
        UnionInducer(members=[{"name": "nope"}])


def test_empty_members_yield_no_relations():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="x")]
    inducer = UnionInducer(members=[])
    assert inducer.induce([], units, document) == []
