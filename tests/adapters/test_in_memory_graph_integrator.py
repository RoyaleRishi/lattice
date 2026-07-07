from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from tests.contracts.graph_integrator_contract import GraphIntegratorContract
from tests.helpers import make_resolution


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
