import pytest

from lattice.ports import Segmenter
from lattice.registry.registry import RegistryError, available, lookup, register


class _FakeSegmenter(Segmenter):
    def segment(self, document):
        return []


def test_register_then_lookup(clean_registry):
    register(Segmenter, "fake")(_FakeSegmenter)
    assert lookup(Segmenter, "fake") is _FakeSegmenter


def test_register_returns_class_for_decorator_use(clean_registry):
    returned = register(Segmenter, "fake")(_FakeSegmenter)
    assert returned is _FakeSegmenter


def test_duplicate_name_rejected(clean_registry):
    register(Segmenter, "fake")(_FakeSegmenter)
    with pytest.raises(RegistryError, match="duplicate"):
        register(Segmenter, "fake")(_FakeSegmenter)


def test_non_subclass_rejected(clean_registry):
    class NotASegmenter:
        pass

    with pytest.raises(RegistryError, match="does not implement"):
        register(Segmenter, "bogus")(NotASegmenter)


def test_lookup_unknown_name_lists_known(clean_registry):
    register(Segmenter, "fake")(_FakeSegmenter)
    with pytest.raises(RegistryError, match="fake"):
        lookup(Segmenter, "missing")


def test_available_lists_registered(clean_registry):
    register(Segmenter, "fake")(_FakeSegmenter)
    assert available(Segmenter)["fake"] is _FakeSegmenter
