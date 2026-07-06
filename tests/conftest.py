import pytest

from lattice.registry import registry


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global registry around a test that registers
    throwaway adapters, so test registrations never leak."""
    saved = {port: dict(names) for port, names in registry._REGISTRY.items()}
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
