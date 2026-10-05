"""Contract every GraphIntegrator adapter must satisfy: the accreting graph
dedupes by identity, and snapshot()/reset() honor spec §4.2."""

from dataclasses import replace

import pytest

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

    def test_restore_replaces_state_and_round_trips(self):
        source = self.make_integrator()
        r1 = make_resolution(surface="vector store")
        r2 = make_resolution(surface="encoder")
        relation = Relation(
            type="IS_A",
            source_id=r1.concept.id,
            target_id=r2.concept.id,
            confidence=1.0,
            provenance="d1",
        )
        source.apply([r1, r2], [relation])
        saved = source.snapshot()

        target = self.make_integrator()
        target.apply([make_resolution(surface="stale state")], [])
        target.restore(saved)
        # restore REPLACES: the stale concept is gone, the saved graph is back
        assert target.snapshot() == saved

    # --- checkpoint / rollback (ADR-0002) ---

    @staticmethod
    def _rel(type_, src, tgt, provenance):
        return Relation(
            type=type_, source_id=src, target_id=tgt, confidence=1.0, provenance=provenance
        )

    def test_rollback_restores_snapshot_as_of_checkpoint(self):
        integrator = self.make_integrator()
        old = make_concept(id="c1", label="vector store", first_seen="d1")
        other = make_concept(id="c2", label="encoder", first_seen="d1")
        rel = self._rel("CO_OCCURS", "c1", "c2", "d1")
        integrator.apply([make_resolution(concept=old), make_resolution(concept=other)], [rel])
        before = integrator.snapshot()

        token = integrator.checkpoint()
        new = make_concept(id="c3", label="reranker", first_seen="d2")
        updated = replace(old, updated_at="d2")
        new_rel = self._rel("IS_A", "c3", "c1", "d2")
        overwritten = self._rel("CO_OCCURS", "c1", "c2", "d2")
        integrator.apply(
            [make_resolution(concept=new), make_resolution(concept=updated, is_new=False)],
            [new_rel, overwritten],
        )
        assert integrator.snapshot() != before  # the apply really changed something

        integrator.rollback(token)
        assert integrator.snapshot() == before

    def test_superseded_token_raises(self):
        integrator = self.make_integrator()
        first = integrator.checkpoint()
        integrator.checkpoint()
        with pytest.raises(ValueError):
            integrator.rollback(first)

    def test_double_rollback_raises(self):
        integrator = self.make_integrator()
        token = integrator.checkpoint()
        integrator.rollback(token)
        with pytest.raises(ValueError):
            integrator.rollback(token)

    def test_rollback_after_restore_raises(self):
        integrator = self.make_integrator()
        token = integrator.checkpoint()
        integrator.restore(integrator.snapshot())
        with pytest.raises(ValueError):
            integrator.rollback(token)

    def test_rollback_after_touching_same_keys_twice(self):
        integrator = self.make_integrator()
        c1 = make_concept(id="c1", label="vector store", first_seen="d1")
        c2 = make_concept(id="c2", label="encoder", first_seen="d1")
        integrator.apply(
            [make_resolution(concept=c1), make_resolution(concept=c2)],
            [self._rel("CO_OCCURS", "c1", "c2", "d1")],
        )
        before = integrator.snapshot()

        token = integrator.checkpoint()
        # same concept key and same relation key hit twice after one checkpoint
        integrator.apply(
            [make_resolution(concept=replace(c1, updated_at="d2"), is_new=False)],
            [self._rel("CO_OCCURS", "c1", "c2", "d2")],
        )
        integrator.apply(
            [make_resolution(concept=replace(c1, updated_at="d3"), is_new=False)],
            [self._rel("CO_OCCURS", "c1", "c2", "d3")],
        )
        assert integrator.snapshot() != before

        integrator.rollback(token)
        assert integrator.snapshot() == before
