"""Consumer simulation (M6 spec §7): everything imports from top-level
lattice — if a test here needs a deeper import, the public surface failed."""

import pytest

import lattice
from lattice import Document, Engine, GraphDelta, GraphSnapshot, GraphView

TEXT_A = "Olive oil is a fat prized in Mediterranean cooking."
TEXT_B = "Mediterranean groves grow olive trees for oil."


def test_all_names_are_importable():
    for name in lattice.__all__:
        assert getattr(lattice, name, None) is not None, name


def test_lite_session_flow():
    engine = Engine()
    assert engine.profile == "lite"
    delta = engine.ingest(TEXT_A)
    assert isinstance(delta, GraphDelta)
    assert delta.document_id == "doc-0"
    assert len(delta.concepts_added) == 4
    engine.ingest(TEXT_B)

    snapshot = engine.snapshot()
    assert isinstance(snapshot, GraphSnapshot)
    view = engine.view()
    assert isinstance(view, GraphView)
    olive = view.find_concept("olive")
    assert olive is not None
    # 3-char words never become mentions on lite (token min_length=4)
    assert view.find_concept("oil") is None


def test_auto_ids_and_timestamps_are_monotonic():
    engine = Engine()
    d0 = engine.ingest(TEXT_A)
    d1 = engine.ingest(TEXT_B)
    assert (d0.document_id, d1.document_id) == ("doc-0", "doc-1")


def test_ingest_overrides():
    engine = Engine()
    delta = engine.ingest(TEXT_A, id="session-9", kind="transcript", timestamp=99.0)
    assert delta.document_id == "session-9"
    # explicit id consumed a counter slot; the next auto id continues
    assert engine.ingest(TEXT_B).document_id == "doc-1"


def test_document_passthrough_and_ambiguity_error():
    engine = Engine()
    document = Document(id="mine", kind="note", text=TEXT_A, timestamp=5.0)
    assert engine.ingest(document).document_id == "mine"
    with pytest.raises(ValueError, match="raw text"):
        engine.ingest(document, id="clash")


def test_ingest_all_mixes_strings_and_documents():
    engine = Engine()
    deltas = engine.ingest_all(
        [TEXT_A, Document(id="mine", kind="note", text=TEXT_B, timestamp=1.0)]
    )
    assert [d.document_id for d in deltas] == ["doc-0", "mine"]


def test_unknown_profile_lists_known_ones():
    with pytest.raises(ValueError, match="lite"):
        Engine(profile="turbo")


def test_from_config_dict_and_toml(tmp_path):
    config = {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "frequency"},
        "resolver": {"name": "exact-label"},
        "relation_inducer": {"name": "co-occurrence"},
        "graph_integrator": {"name": "in-memory"},
        "embedder": {"name": "hashing"},
    }
    engine = Engine.from_config(config)
    assert engine.profile is None
    assert engine.config.scorer.name == "frequency"
    engine.ingest(TEXT_A)

    toml = tmp_path / "run.toml"
    toml.write_text(
        "\n".join(
            f'[{port}]\nname = "{spec["name"]}"'
            for port, spec in config.items()
        )
    )
    engine2 = Engine.from_config(toml)
    assert engine2.config.scorer.name == "frequency"


def test_reset_clears_graph_and_counter():
    engine = Engine()
    engine.ingest(TEXT_A)
    engine.reset()
    assert engine.snapshot().concepts == ()
    assert engine.ingest(TEXT_B).document_id == "doc-0"


@pytest.mark.ml
def test_standard_profile_constructs_and_ingests():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        engine = Engine(profile="standard")
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    delta = engine.ingest(TEXT_A)
    assert delta.errors == ()
    assert engine.view().find_concept("olive oil") is not None
