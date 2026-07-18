from lattice.adapters.document_metric.coherence import Coherence
from lattice.adapters.metric.hierarchy_sanity import HierarchySanity
from lattice.adapters.metric.redundancy import Redundancy
from lattice.harness.stats.records import Resamplable


def test_intrinsic_metrics_declare_holistic():
    assert Redundancy().kind == "holistic"
    assert HierarchySanity().kind == "holistic"
    # Coherence needs an embedder; a trivial stand-in is fine for the attribute check
    class _E:
        def embed(self, xs): return [(0.0,) for _ in xs]
    assert Coherence(_E()).kind == "holistic"
    assert isinstance(Redundancy(), Resamplable)
