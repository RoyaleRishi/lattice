from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from lattice.core.types import Relation
from tests.contracts.graph_integrator_contract import GraphIntegratorContract
from tests.helpers import make_resolution


def _rel(src, tgt, type_="CO_OCCURS"):
    return Relation(type=type_, source_id=src, target_id=tgt, confidence=1.0, provenance="d1")


class TestInMemoryGraphIntegrator(GraphIntegratorContract):
    def make_integrator(self) -> InMemoryGraphIntegrator:
        return InMemoryGraphIntegrator()

    def test_snapshot_is_sorted_for_determinism(self):
        integrator = self.make_integrator()
        rb = make_resolution(surface="b-concept")
        ra = make_resolution(surface="a-concept")
        integrator.apply([rb, ra], [])
        snapshot = integrator.snapshot()
        ids = [c.id for c in snapshot.concepts]
        assert ids == sorted(ids)

    def test_undo_log_only_holds_touched_keys(self):
        integrator = self.make_integrator()
        integrator.apply([make_resolution(surface=f"concept {i}") for i in range(20)], [])
        integrator.checkpoint()
        integrator.apply([make_resolution(surface="brand new")], [])
        # 20 old concepts shouldn't show up in the log, only the one touched
        assert len(integrator._undo_concepts) == 1

    def test_undo_log_only_holds_touched_relation_keys(self):
        integrator = self.make_integrator()
        integrator.apply([], [_rel(f"s{i}", f"t{i}") for i in range(20)])
        integrator.checkpoint()
        integrator.apply([], [_rel("new-s", "new-t")])
        assert len(integrator._undo_relations) == 1

    def test_apply_without_checkpoint_logs_nothing(self):
        integrator = self.make_integrator()
        integrator.apply([make_resolution(surface="x")], [_rel("a", "b")])
        assert integrator._undo_concepts == {}
        assert integrator._undo_relations == {}
