"""Contract every RelationInducer adapter must satisfy."""

from lattice.ports import RelationInducer
from tests.helpers import make_document, make_resolution, make_unit


class RelationInducerContract:
    def make_inducer(self) -> RelationInducer:
        raise NotImplementedError("subclass must provide the adapter under test")

    def _fixture(self):
        document = make_document(id="d1")
        units = [make_unit(id="d1:u0", document_id="d1", text="vector store and encoder")]
        resolutions = [
            make_resolution(surface="vector store", unit_id="d1:u0"),
            make_resolution(surface="encoder", unit_id="d1:u0"),
        ]
        return resolutions, units, document

    def test_relations_reference_resolved_concepts(self):
        resolutions, units, document = self._fixture()
        relations = self.make_inducer().induce(resolutions, units, document)
        concept_ids = {r.concept.id for r in resolutions}
        for relation in relations:
            assert relation.source_id in concept_ids
            assert relation.target_id in concept_ids

    def test_provenance_is_document_id(self):
        resolutions, units, document = self._fixture()
        relations = self.make_inducer().induce(resolutions, units, document)
        assert all(r.provenance == document.id for r in relations)

    def test_empty_resolutions_yield_no_relations(self):
        _, units, document = self._fixture()
        assert self.make_inducer().induce([], units, document) == []
