"""Persistence contract (M6 spec §4.3): resume equivalence, counter
persistence, format versioning, idempotent round-trip. Top-level imports
only."""

import json

import pytest

from lattice import Engine
from lattice.ports import RelationInducer

TEXT_A = "Olive oil is a fat prized in Mediterranean cooking."
TEXT_B = "Mediterranean groves grow olive trees for oil."
TEXT_C = "Olive presses yield fresh oil each autumn."

KUBE_A = "Alpha discusses kubernetes orchestration."
KUBE_B = "Beta mentions kubernetes again."
KUBE_C = "Gamma revisits kubernetes deployments again."

_KUBERNETES_CONFIG = {
    "segmenter": {"name": "block"},
    "extractor": {"name": "token"},
    "scorer": {"name": "embedding-cosine"},
    "resolver": {"name": "embedding-nn", "params": {"threshold": 0.99}},
    "relation_inducer": {"name": "hearst"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "concept_store": {"name": "in-memory"},
    "run": {"on_error": "skip", "seed": 0},
}


class _FailOnBeta(RelationInducer):
    """Wraps a real inducer but raises for the one document that mentions
    "Beta" — forcing a mid-pipeline failure *after* the resolver has already
    upserted concepts into the store, which is what exposes the
    store/graph divergence this fix closes."""

    def __init__(self, inner: RelationInducer):
        self._inner = inner

    def induce(self, resolutions, units, document):
        if "Beta" in document.text:
            raise RuntimeError("boom-on-beta")
        return self._inner.induce(resolutions, units, document)


def _make_flaky_kubernetes_engine() -> Engine:
    """Engine wired per the task-3 regression scenario, with its relation
    inducer swapped for one that fails only on the "Beta" document."""
    engine = Engine.from_config(_KUBERNETES_CONFIG)
    engine._orchestrator.relation_inducer = _FailOnBeta(
        engine._orchestrator.relation_inducer
    )
    return engine


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


def test_resume_equivalence_holds_when_a_document_fails_mid_pipeline(tmp_path):
    """Regression for task 3: under on_error="skip", a document that fails
    after the resolver has mutated the concept store (here, in the relation
    inducer) must not leave the store diverged from the graph. Before the
    fix, ingest(A,B,C) and ingest(A,B); save; load; ingest(C) disagreed."""
    straight = _make_flaky_kubernetes_engine()
    straight.ingest_all([KUBE_A, KUBE_B, KUBE_C])

    interrupted = _make_flaky_kubernetes_engine()
    interrupted.ingest_all([KUBE_A, KUBE_B])
    path = tmp_path / "memory.json"
    interrupted.save(path)
    resumed = Engine.load(path)
    resumed.ingest(KUBE_C)

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


def test_load_rejects_concept_with_wrong_embedding_dimension(tmp_path):
    engine = Engine()
    engine.ingest(TEXT_A)
    path = tmp_path / "memory.json"
    engine.save(path)
    payload = json.loads(path.read_text())
    bad_concept = payload["concepts"][0]
    bad_concept["embedding"] = bad_concept["embedding"][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=bad_concept["id"]):
        Engine.load(path)
