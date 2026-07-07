"""Contract every GraphIntegrator adapter must satisfy: the accreting graph
dedupes by identity, and snapshot()/reset() honor spec §4.2."""

from dataclasses import replace

from lattice.core.types import Relation
from lattice.ports import GraphIntegrator
from tests.helpers import make_concept, make_resolution


class GraphIntegratorContract:
    def make_integrator(self) -> GraphIntegrator:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_applied_concepts_and_relations_appear_in_snapshot(self):
        integrator = self.make_integrator()
        r1 = make_resolution(surface="vector store")
        r2 = make_resolution(surface="encoder")
        relation = Relation(
            type="CO_OCCURS",
            source_id=r1.concept.id,
            target_id=r2.concept.id,
            confidence=1.0,
            provenance="d1",
        )
        integrator.apply([r1, r2], [relation])
        snapshot = integrator.snapshot()
        assert {c.id for c in snapshot.concepts} == {r1.concept.id, r2.concept.id}
        assert snapshot.relations == (relation,)

    def test_reapplying_same_concept_does_not_duplicate(self):
        integrator = self.make_integrator()
        concept = make_concept(id="c1", label="vector store")
        integrator.apply([make_resolution(concept=concept)], [])
        integrator.apply([make_resolution(concept=concept, is_new=False)], [])
        assert len(integrator.snapshot().concepts) == 1

    def test_updated_concept_replaces_previous_version(self):
        integrator = self.make_integrator()
        v1 = make_concept(id="c1", label="vector store", first_seen="d1")
        integrator.apply([make_resolution(concept=v1)], [])
        v2 = replace(v1, updated_at="d2")
        integrator.apply([make_resolution(concept=v2, is_new=False)], [])
        [stored] = integrator.snapshot().concepts
        assert stored.updated_at == "d2"

    def test_reset_empties_the_graph(self):
        integrator = self.make_integrator()
        integrator.apply([make_resolution(surface="vector store")], [])
        integrator.reset()
        snapshot = integrator.snapshot()
        assert snapshot.concepts == () and snapshot.relations == ()
