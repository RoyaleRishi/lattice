"""Persistence contract (M6 spec §4.3): resume equivalence, counter
persistence, format versioning, idempotent round-trip. Top-level imports
only."""

import json

import pytest

from lattice import Engine

TEXT_A = "Olive oil is a fat prized in Mediterranean cooking."
TEXT_B = "Mediterranean groves grow olive trees for oil."
TEXT_C = "Olive presses yield fresh oil each autumn."


def test_resume_equivalence(tmp_path):
    """ingest(A,B); save; load; ingest(C) == ingest(A,B,C) straight through."""
    straight = Engine()
    straight.ingest_all([TEXT_A, TEXT_B, TEXT_C])

    interrupted = Engine()
    interrupted.ingest_all([TEXT_A, TEXT_B])
    path = tmp_path / "memory.json"
    interrupted.save(path)
    resumed = Engine.load(path)
    resumed.ingest(TEXT_C)

    assert resumed.snapshot() == straight.snapshot()


def test_counter_persists_so_auto_ids_never_collide(tmp_path):
    engine = Engine()
    engine.ingest_all([TEXT_A, TEXT_B])
    path = tmp_path / "memory.json"
    engine.save(path)
    resumed = Engine.load(path)
    assert resumed.ingest(TEXT_C).document_id == "doc-2"


def test_profile_and_config_round_trip(tmp_path):
    engine = Engine()
    path = tmp_path / "memory.json"
    engine.save(path)
    resumed = Engine.load(path)
    assert resumed.profile == "lite"
    assert resumed.config == engine.config


def test_save_file_shape(tmp_path):
    engine = Engine()
    engine.ingest(TEXT_A)
    path = tmp_path / "memory.json"
    engine.save(path)
    payload = json.loads(path.read_text())
    assert payload["format_version"] == 1
    assert payload["profile"] == "lite"
    assert payload["document_counter"] == 1
    assert {c["label"] for c in payload["concepts"]} >= {"olive", "cooking"}
    assert set(payload) == {
        "format_version", "lattice_version", "profile", "config",
        "document_counter", "concepts", "relations",
    }


def test_save_load_save_is_byte_identical(tmp_path):
    engine = Engine()
    engine.ingest_all([TEXT_A, TEXT_B])
    first = tmp_path / "first.json"
    engine.save(first)
    second = tmp_path / "second.json"
    Engine.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_format_version_mismatch_raises(tmp_path):
    engine = Engine()
    path = tmp_path / "memory.json"
    engine.save(path)
    payload = json.loads(path.read_text())
    payload["format_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="99"):
        Engine.load(path)


def test_corrupt_file_raises_json_error(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        Engine.load(path)
