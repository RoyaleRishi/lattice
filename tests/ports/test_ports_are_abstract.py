import pytest

from lattice.ports import (
    ConceptStore,
    Dataset,
    Embedder,
    Extractor,
    GraphIntegrator,
    Metric,
    RelationInducer,
    Resolver,
    Scorer,
    Segmenter,
)

ALL_PORTS = [
    Segmenter,
    Extractor,
    Scorer,
    Resolver,
    RelationInducer,
    GraphIntegrator,
    Embedder,
    ConceptStore,
    Dataset,
    Metric,
]


@pytest.mark.parametrize("port", ALL_PORTS, ids=lambda p: p.__name__)
def test_port_cannot_be_instantiated(port):
    with pytest.raises(TypeError):
        port()


def test_concrete_subclass_is_instantiable():
    class NullSegmenter(Segmenter):
        def segment(self, document):
            return []

    assert NullSegmenter().segment(None) == []
