"""Tests for scripts/migrate_save.py (ADR-0003): v1 fixtures migrate to v2 and
resume exactly like a straight-through run."""

import json
import sys
from pathlib import Path

import pytest
from scripts.migrate_save import main, migrate

from lattice import Engine
from lattice.adapters.resolver.embedding_nn import EmbeddingNNResolver
from tests.api.test_persistence import (
    _STEMMED_CONFIG,
    STEM_A,
    STEM_B,
    STEM_C,
    TEXT_A,
    TEXT_B,
    TEXT_C,
)

SAVES = Path(__file__).parent.parent / "fixtures" / "saves"
LITE_V1 = SAVES / "v1_lite_after_A_B.json"
STEM_V1 = SAVES / "v1_stemmed_after_STEM_A.json"


def test_migrated_lite_save_resumes_like_straight_run(tmp_path):
    out = tmp_path / "v2.json"
    migrate(LITE_V1, out)
    resumed = Engine.load(out)
    resumed.ingest(TEXT_C)

    straight = Engine()
    straight.ingest_all([TEXT_A, TEXT_B, TEXT_C])
    assert resumed.snapshot() == straight.snapshot()


def test_migrated_stemmed_save_resumes_like_straight_run(tmp_path):
    out = tmp_path / "v2.json"
    migrate(STEM_V1, out)
    resumed = Engine.load(out)
    resumed.ingest_all([STEM_B, STEM_C])

    straight = Engine.from_config(_STEMMED_CONFIG)
    straight.ingest_all([STEM_A, STEM_B, STEM_C])
    assert resumed.snapshot() == straight.snapshot()


def test_migrated_file_is_v2_and_keeps_the_counter(tmp_path):
    out = tmp_path / "v2.json"
    migrate(LITE_V1, out)
    payload = json.loads(out.read_text())
    assert payload["format_version"] == 2
    assert "resolver_state" in payload
    assert Engine.load(out).ingest(TEXT_C).document_id == "doc-2"


def test_cli_migrates_a_file(tmp_path, monkeypatch):
    out = tmp_path / "v2.json"
    monkeypatch.setattr(sys, "argv", ["migrate_save", str(LITE_V1), str(out)])
    main()
    assert json.loads(out.read_text())["format_version"] == 2


def test_migrating_a_v2_file_raises(tmp_path):
    v2 = tmp_path / "v2.json"
    migrate(LITE_V1, v2)
    with pytest.raises(ValueError, match="already"):
        migrate(v2, tmp_path / "again.json")


def test_rejects_unknown_version(tmp_path):
    src = tmp_path / "v9.json"
    src.write_text(json.dumps({"format_version": 9}))
    with pytest.raises(ValueError, match="format_version"):
        migrate(src, tmp_path / "out.json")


def test_refuses_to_overwrite_the_source(tmp_path):
    with pytest.raises(ValueError, match="in place"):
        migrate(LITE_V1, LITE_V1)


def test_resolver_that_cannot_restore_writes_nothing(tmp_path, monkeypatch):
    def boom(self, state):
        raise RuntimeError("cannot rebuild")

    monkeypatch.setattr(EmbeddingNNResolver, "restore", boom)
    out = tmp_path / "v2.json"
    with pytest.raises(RuntimeError, match="EmbeddingNNResolver"):
        migrate(LITE_V1, out)
    assert not out.exists()
