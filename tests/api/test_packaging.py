"""Packaging invariants (M6 spec §7): version agreement, typed marker,
consumer-installable ml extra mirroring the dev group."""

import tomllib
from pathlib import Path

import lattice

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_version_agreement():
    assert lattice.__version__ == _pyproject()["project"]["version"] == "0.2.0"


def test_py_typed_marker_ships_in_the_package():
    assert (ROOT / "src" / "lattice" / "py.typed").exists()


def test_ml_extra_mirrors_the_dependency_group():
    data = _pyproject()
    extra = data["project"]["optional-dependencies"]["ml"]
    group = data["dependency-groups"]["ml"]
    assert extra == group
