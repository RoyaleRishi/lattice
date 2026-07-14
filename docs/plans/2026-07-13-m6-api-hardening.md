# M6 Engine API Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-import public surface — `lattice.Engine` with lite/standard profiles, `GraphView` reads, versioned save/load with a resume-equivalence guarantee, an executable README, and packaging polish (py.typed, `lattice[ml]` extra, 0.2.0).

**Architecture:** Per `docs/2026-07-13-m6-api-hardening-design.md`: a facade over the existing factory/orchestrator (no pipeline changes), one new port method (`GraphIntegrator.restore`), one new read module (`GraphView`), JSON persistence v1. The lite profile's end-to-end behavior and the resume-equivalence semantics were verified live against the real pipeline before this plan was committed.

**Tech Stack:** Python 3.13 (uv), pydantic RunConfig (existing), stdlib json/tomllib, pytest, hatchling build.

## Global Constraints

- `pyproject.toml`: NO new dependencies. Exactly two sanctioned edits, both in Task 6: `version = "0.2.0"` and a `[project.optional-dependencies]` `ml` entry mirroring the existing `[dependency-groups]` `ml` list verbatim.
- No network, no model loads in tests (ml-marked tests use importorskip + OSError-skip; everything else runs lite).
- `data/`, `reports/`, `.superpowers/` gitignored — never commit.
- Public contract names are load-bearing: `Engine`, `GraphView`, profile names `"lite"`/`"standard"`, save keys per format v1, `__all__` exactly as in Task 3.
- All source modules import via full paths (`lattice.core.types`, never `from lattice import …`) — the package `__init__` imports `engine`, so a reverse import would be circular. (`from lattice import __version__` INSIDE a function body is fine and used once in `save`.)
- Run tests with `uv run --no-sync pytest <path> -q`; if imports fail: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null`.
- `uv run --no-sync ruff check .` before every commit; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
src/lattice/ports/graph_integrator.py               Task 1 (modify: restore)
src/lattice/adapters/graph_integrator/in_memory.py  Task 1 (modify: restore impl)
tests/contracts/graph_integrator_contract.py        Task 1 (modify: restore test)
src/lattice/graph_view.py                           Task 2  GraphView
src/lattice/engine.py                               Task 3  Engine (+ Task 4 save/load)
src/lattice/__init__.py                             Task 3 (public exports + 0.2.0)
README.md                                           Task 5
src/lattice/py.typed                                Task 6
pyproject.toml                                      Task 6 (version + extra)
tests/api/__init__.py                               Task 2
tests/api/test_graph_view.py                        Task 2
tests/api/test_public_surface.py                    Task 3
tests/api/test_persistence.py                       Task 4
tests/api/test_readme.py                            Task 5
tests/api/test_packaging.py                         Task 6
```

Task 7 is the exit-criteria run (wheel build + full verification); no committed code.

**Suggested implementers:** Tasks 2, 5, 6 — haiku (complete code below). Tasks 1, 3, 4 — sonnet. Task 7 — orchestrator. All reviewers sonnet. Machine-verified pre-commit: the lite profile's ingest behavior (concept labels, zero relations on the fixture texts, delta counts 4/3/5 added), and resume-equivalence == True using exactly the restore semantics Task 1/Task 4 implement.

---

### Task 1: `GraphIntegrator.restore` port method

**Files:**
- Modify: `src/lattice/ports/graph_integrator.py`
- Modify: `src/lattice/adapters/graph_integrator/in_memory.py`
- Modify: `tests/contracts/graph_integrator_contract.py` (add one test)

**Interfaces:**
- Consumes: existing port (`apply`, `snapshot`, `reset`), `GraphSnapshot` (already imported in the port module).
- Produces: abstract `restore(self, snapshot: GraphSnapshot) -> None` on the port; in-memory implementation replacing internal state. Task 4's `Engine.load` calls it. Only `InMemoryGraphIntegrator` implements the port (verified — no test doubles subclass it), so adding the abstractmethod breaks nothing else.

- [ ] **Step 1: Add the failing contract test**

Append to `tests/contracts/graph_integrator_contract.py` (inside the class):

```python
    def test_restore_replaces_state_and_round_trips(self):
        source = self.make_integrator()
        r1 = make_resolution(surface="vector store")
        r2 = make_resolution(surface="encoder")
        relation = Relation(
            type="IS_A",
            source_id=r1.concept.id,
            target_id=r2.concept.id,
            confidence=1.0,
            provenance="d1",
        )
        source.apply([r1, r2], [relation])
        saved = source.snapshot()

        target = self.make_integrator()
        target.apply([make_resolution(surface="stale state")], [])
        target.restore(saved)
        # restore REPLACES: the stale concept is gone, the saved graph is back
        assert target.snapshot() == saved
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_in_memory_graph_integrator.py -q`
Expected: FAIL — `AttributeError: 'InMemoryGraphIntegrator' object has no attribute 'restore'`

- [ ] **Step 3: Add the port method and implementation**

In `src/lattice/ports/graph_integrator.py`, add to the ABC (after `snapshot`):

```python
    @abstractmethod
    def restore(self, snapshot: GraphSnapshot) -> None:
        """Replace internal state with the snapshot's contents (M6 spec §3:
        the persistence hook — Engine.load hands back a saved graph)."""
        ...
```

In `src/lattice/adapters/graph_integrator/in_memory.py`, add after `snapshot`:

```python
    def restore(self, snapshot: GraphSnapshot) -> None:
        self._concepts = {concept.id: concept for concept in snapshot.concepts}
        self._relations = {
            (r.type, r.source_id, r.target_id): r for r in snapshot.relations
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_in_memory_graph_integrator.py tests/ports/ -q`
Expected: PASS (contract suite incl. the new test; the port-abstractness test still passes — one more abstractmethod keeps the ABC uninstantiable). Then `uv run --no-sync pytest -q` — full suite green.

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/ports/graph_integrator.py src/lattice/adapters/graph_integrator/in_memory.py tests/contracts/graph_integrator_contract.py
git commit -m "feat: add GraphIntegrator.restore for engine persistence

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `GraphView`

**Files:**
- Create: `src/lattice/graph_view.py`
- Create: `tests/api/__init__.py` (empty)
- Test: `tests/api/test_graph_view.py`

**Interfaces:**
- Consumes: `Concept`, `GraphSnapshot`, `Relation` from `lattice.core.types`.
- Produces: `GraphView(snapshot)` with `concepts() -> tuple[Concept, ...]`, `find_concept(label: str) -> Concept | None` (casefolded exact), `relations(type: str | None = None) -> tuple[Relation, ...]`, `neighbors(concept_id: str, type: str | None = None) -> tuple[tuple[Relation, Concept], ...]`. Task 3 exports it and `Engine.view()` returns it.

- [ ] **Step 1: Write the failing test**

Create empty `tests/api/__init__.py`, then `tests/api/test_graph_view.py`:

```python
from lattice.core.types import Concept, GraphSnapshot, Relation
from lattice.graph_view import GraphView


def _concept(cid: str, label: str) -> Concept:
    return Concept(
        id=cid, label=label, embedding=(1.0, 0.0),
        first_seen="d1", updated_at="d1",
    )


def _relation(type: str, source: str, target: str) -> Relation:
    return Relation(
        type=type, source_id=source, target_id=target,
        confidence=1.0, provenance="d1",
    )


SNAPSHOT = GraphSnapshot(
    concepts=(
        _concept("c:fat", "fat"),
        _concept("c:oil", "oil"),
        _concept("c:olive oil", "olive oil"),
    ),
    relations=(
        _relation("IS_A", "c:olive oil", "c:oil"),
        _relation("CO_OCCURS", "c:fat", "c:olive oil"),
    ),
)


def test_concepts_passthrough():
    assert GraphView(SNAPSHOT).concepts() == SNAPSHOT.concepts


def test_find_concept_is_casefolded_exact_match():
    view = GraphView(SNAPSHOT)
    assert view.find_concept("olive oil").id == "c:olive oil"
    assert view.find_concept("OLIVE OIL").id == "c:olive oil"
    assert view.find_concept("olive") is None


def test_relations_type_filter():
    view = GraphView(SNAPSHOT)
    assert view.relations() == SNAPSHOT.relations
    assert [r.type for r in view.relations("IS_A")] == ["IS_A"]
    assert view.relations("NOPE") == ()


def test_neighbors_returns_relation_and_other_concept():
    view = GraphView(SNAPSHOT)
    pairs = view.neighbors("c:olive oil")
    # sorted by (relation.type, other.id): CO_OCCURS/fat before IS_A/oil
    assert [(r.type, c.id) for r, c in pairs] == [
        ("CO_OCCURS", "c:fat"),
        ("IS_A", "c:oil"),
    ]
    # direction is readable from the relation itself
    is_a = pairs[1][0]
    assert is_a.source_id == "c:olive oil" and is_a.target_id == "c:oil"


def test_neighbors_type_filter_and_unknown_id():
    view = GraphView(SNAPSHOT)
    assert [(r.type, c.id) for r, c in view.neighbors("c:olive oil", "IS_A")] == [
        ("IS_A", "c:oil")
    ]
    assert view.neighbors("c:absent") == ()


def test_view_is_stable_across_repeated_queries():
    view = GraphView(SNAPSHOT)
    assert view.neighbors("c:olive oil") == view.neighbors("c:olive oil")
    assert view.find_concept("fat") == view.find_concept("fat")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/api/test_graph_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.graph_view'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/graph_view.py`:

```python
from lattice.core.types import Concept, GraphSnapshot, Relation


class GraphView:
    """Read-optimized view over one immutable GraphSnapshot (M6 spec §4.2).
    Indexes build lazily on first use. A view reflects the snapshot it was
    made from — take a fresh view after new ingests. Labels are stored
    lowercased by the resolvers; lookups casefold both sides."""

    def __init__(self, snapshot: GraphSnapshot):
        self._snapshot = snapshot
        self._by_label: dict[str, Concept] | None = None
        self._by_id: dict[str, Concept] | None = None
        self._adjacency: dict[str, list[Relation]] | None = None

    def concepts(self) -> tuple[Concept, ...]:
        return self._snapshot.concepts

    def find_concept(self, label: str) -> Concept | None:
        if self._by_label is None:
            self._by_label = {
                concept.label.casefold(): concept
                for concept in self._snapshot.concepts
            }
        return self._by_label.get(label.casefold())

    def relations(self, type: str | None = None) -> tuple[Relation, ...]:
        if type is None:
            return self._snapshot.relations
        return tuple(r for r in self._snapshot.relations if r.type == type)

    def neighbors(
        self, concept_id: str, type: str | None = None
    ) -> tuple[tuple[Relation, Concept], ...]:
        if self._adjacency is None:
            self._adjacency = {}
            for relation in self._snapshot.relations:
                self._adjacency.setdefault(relation.source_id, []).append(relation)
                if relation.target_id != relation.source_id:
                    self._adjacency.setdefault(relation.target_id, []).append(relation)
        if self._by_id is None:
            self._by_id = {c.id: c for c in self._snapshot.concepts}
        pairs: list[tuple[Relation, Concept]] = []
        for relation in self._adjacency.get(concept_id, ()):
            if type is not None and relation.type != type:
                continue
            other_id = (
                relation.target_id
                if relation.source_id == concept_id
                else relation.source_id
            )
            other = self._by_id.get(other_id)
            if other is not None:
                pairs.append((relation, other))
        pairs.sort(key=lambda pair: (pair[0].type, pair[1].id, pair[0].source_id))
        return tuple(pairs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/api/test_graph_view.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/graph_view.py tests/api/__init__.py tests/api/test_graph_view.py
git commit -m "feat: add GraphView read surface

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `Engine` facade + public exports

**Files:**
- Create: `src/lattice/engine.py` (without save/load — Task 4 adds them)
- Modify: `src/lattice/__init__.py` (full replacement below)
- Test: `tests/api/test_public_surface.py`

**Interfaces:**
- Consumes: `build_orchestrator(RunConfig)` from `lattice.config.factory`; `load_config(path, model=RunConfig)` from `lattice.config.loader`; `RunConfig` from `lattice.config.schema`; `GraphView` from Task 2; core types.
- Produces: `Engine(profile="lite")`, `Engine.from_config(RunConfig | dict | str | Path)`, attributes `profile: str | None` / `config: RunConfig`, methods `ingest`, `ingest_all`, `snapshot`, `view`, `reset`; module constants `_PROFILES`, `FORMAT_VERSION = 1`; internal `_init(config, profile)` that Task 4's `load` also uses. `lattice/__init__.py` exports per spec §3 with `__version__ = "0.2.0"`.

Measured lite behavior this task's tests rely on (verified live pre-plan): ingesting `"Olive oil is a fat prized in Mediterranean cooking."` then `"Mediterranean groves grow olive trees for oil."` yields concepts including labels `olive`, `mediterranean`, `cooking`, `groves` (token extractor: words ≥ 4 chars, lowercased; "oil"/"fat" are 3 chars and never become mentions), zero relations on these texts, and deltas with 4 then 3 `concepts_added`.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_public_surface.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/api/test_public_surface.py -q`
Expected: FAIL — `ImportError: cannot import name 'Engine' from 'lattice'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/engine.py`:

```python
"""Public facade (M6 spec §4.1): the one import a consumer needs.

    from lattice import Engine

    engine = Engine()                    # "lite": dependency-free, same
                                         # topology as standard, toy quality
    engine = Engine(profile="standard")  # benchmark-validated stack
                                         # (install lattice[ml] + fetch models)
"""

from collections.abc import Iterable
from pathlib import Path

from lattice.config.factory import build_orchestrator
from lattice.config.loader import load_config
from lattice.config.schema import RunConfig
from lattice.core.types import Document, GraphDelta, GraphSnapshot
from lattice.graph_view import GraphView

FORMAT_VERSION = 1

_UNION = {
    "name": "union",
    "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
}

# Both profiles share one pipeline topology (M6 spec §4.1); they differ only
# in extractor and embedder. Every "standard" choice traces to sweep
# evidence: cosine scorer (M2), embedding-nn@0.90 (M5's recorded operating
# point), hearst+compound union (M4).
_PROFILES: dict[str, dict] = {
    "lite": {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "embedding-cosine"},
        "resolver": {"name": "embedding-nn", "params": {"threshold": 0.90}},
        "relation_inducer": _UNION,
        "graph_integrator": {"name": "in-memory"},
        "embedder": {"name": "hashing"},
        "concept_store": {"name": "in-memory"},
        "run": {"on_error": "fail", "seed": 0},
    },
    "standard": {
        "segmenter": {"name": "block"},
        "extractor": {"name": "noun-chunk"},
        "scorer": {"name": "embedding-cosine"},
        "resolver": {"name": "embedding-nn", "params": {"threshold": 0.90}},
        "relation_inducer": _UNION,
        "graph_integrator": {"name": "in-memory"},
        "embedder": {"name": "sentence-transformer"},
        "concept_store": {"name": "in-memory"},
        "run": {"on_error": "fail", "seed": 0},
    },
}


class Engine:
    """document → accreting concept graph. Stateful; one Engine = one graph.

    Errors propagate ("fail" policy) in both profiles — a consumer that
    prefers poison-document tolerance passes on_error="skip" via
    from_config and inspects delta.errors (partial-mutation caveat: see
    Orchestrator docstring)."""

    def __init__(self, profile: str = "lite"):
        if profile not in _PROFILES:
            known = ", ".join(sorted(_PROFILES))
            raise ValueError(f"unknown profile {profile!r} (known: {known})")
        self._init(RunConfig.model_validate(_PROFILES[profile]), profile)

    def _init(self, config: RunConfig, profile: str | None) -> None:
        self.config = config
        self.profile = profile
        self._orchestrator = build_orchestrator(config)
        self._counter = 0

    @classmethod
    def from_config(cls, config: "RunConfig | dict | str | Path") -> "Engine":
        if isinstance(config, (str, Path)):
            run_config = load_config(config, model=RunConfig)
        elif isinstance(config, RunConfig):
            run_config = config
        else:
            run_config = RunConfig.model_validate(config)
        engine = cls.__new__(cls)
        engine._init(run_config, None)
        return engine

    def ingest(
        self,
        document: Document | str,
        *,
        id: str | None = None,
        kind: str = "note",
        timestamp: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GraphDelta:
        """Process one document. Raw text is wrapped in a Document with auto
        id "doc-N" and timestamp N (a per-engine counter that save/load
        persists); every field is overridable by keyword. Passing a
        ready-made Document together with overrides is ambiguous and raises."""
        if isinstance(document, Document):
            if id is not None or timestamp is not None or metadata is not None or kind != "note":
                raise ValueError(
                    "field overrides apply to raw text only, not a Document"
                )
            return self._orchestrator.process(document)
        n = self._counter
        self._counter += 1
        return self._orchestrator.process(
            Document(
                id=id if id is not None else f"doc-{n}",
                kind=kind,
                text=document,
                timestamp=timestamp if timestamp is not None else float(n),
                metadata=metadata or {},
            )
        )

    def ingest_all(self, documents: Iterable[Document | str]) -> list[GraphDelta]:
        return [self.ingest(document) for document in documents]

    def snapshot(self) -> GraphSnapshot:
        return self._orchestrator.snapshot()

    def view(self) -> GraphView:
        return GraphView(self._orchestrator.snapshot())

    def reset(self) -> None:
        """Empty the graph and restart the document counter."""
        self._orchestrator.graph_integrator.reset()
        store = getattr(self._orchestrator.resolver, "concept_store", None)
        if store is not None:
            store.reset()
        self._counter = 0
```

Replace `src/lattice/__init__.py` entirely with:

```python
"""lattice: concept-memory engine — documents in, an accreting normalized
concept graph out. The public contract is __all__ (M6 spec §3); everything
else may change without notice pre-1.0."""

from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Relation,
)
from lattice.engine import Engine
from lattice.graph_view import GraphView

__version__ = "0.2.0"

__all__ = [
    "Concept",
    "Document",
    "Engine",
    "GraphDelta",
    "GraphSnapshot",
    "GraphView",
    "Relation",
    "__version__",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/api/test_public_surface.py -q`
Expected: PASS (9 tests + 1 ml skip/pass). Then `uv run --no-sync pytest -q` — the full suite must stay green (the `__init__` rewrite must not break any existing import).

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/engine.py src/lattice/__init__.py tests/api/test_public_surface.py
git commit -m "feat: add Engine facade with lite/standard profiles and public exports

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Persistence — `Engine.save` / `Engine.load`

**Files:**
- Modify: `src/lattice/engine.py` (add imports + two methods)
- Test: `tests/api/test_persistence.py`

**Interfaces:**
- Consumes: Task 1's `restore`, Task 3's `Engine._init`/`FORMAT_VERSION`; `Concept`, `Relation`, `GraphSnapshot`; stdlib `json`.
- Produces: `engine.save(path: str | Path) -> None`, `Engine.load(path: str | Path) -> Engine`. Format v1 keys: `format_version`, `lattice_version`, `profile`, `config`, `document_counter`, `concepts`, `relations`. The resume-equivalence contract (spec §4.3) is the acceptance test.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_persistence.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/api/test_persistence.py -q`
Expected: FAIL — `AttributeError: 'Engine' object has no attribute 'save'`

- [ ] **Step 3: Write the implementation**

In `src/lattice/engine.py`: add `import json` at the top of the imports, extend the core-types import to include `Concept` and `Relation`:

```python
import json
from collections.abc import Iterable
from pathlib import Path

from lattice.config.factory import build_orchestrator
from lattice.config.loader import load_config
from lattice.config.schema import RunConfig
from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Relation,
)
from lattice.graph_view import GraphView
```

and add these two methods to `Engine` (after `view`, before `reset`):

```python
    def save(self, path: str | Path) -> None:
        """Serialize the accreted graph + config to versioned JSON (format
        v1, M6 spec §4.3). The concept store is not serialized separately:
        resolvers upsert exactly the concepts the integrator holds, so the
        snapshot is the single source of truth."""
        from lattice import __version__  # inside the function: no cycle

        snapshot = self._orchestrator.snapshot()
        payload = {
            "format_version": FORMAT_VERSION,
            "lattice_version": __version__,
            "profile": self.profile,
            "config": self.config.model_dump(),
            "document_counter": self._counter,
            "concepts": [
                {
                    "id": c.id,
                    "label": c.label,
                    "embedding": list(c.embedding),
                    "first_seen": c.first_seen,
                    "updated_at": c.updated_at,
                }
                for c in snapshot.concepts
            ],
            "relations": [
                {
                    "type": r.type,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "confidence": r.confidence,
                    "provenance": r.provenance,
                }
                for r in snapshot.relations
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Engine":
        """Rebuild an Engine from a save file. Resume-equivalence contract:
        processing after load equals processing straight through."""
        payload = json.loads(Path(path).read_text())
        found = payload.get("format_version")
        if found != FORMAT_VERSION:
            raise ValueError(
                f"unsupported save format_version {found!r} "
                f"(this lattice reads {FORMAT_VERSION})"
            )
        engine = cls.__new__(cls)
        engine._init(
            RunConfig.model_validate(payload["config"]), payload["profile"]
        )
        concepts = tuple(
            Concept(
                id=c["id"],
                label=c["label"],
                embedding=tuple(c["embedding"]),
                first_seen=c["first_seen"],
                updated_at=c["updated_at"],
            )
            for c in payload["concepts"]
        )
        relations = tuple(
            Relation(
                type=r["type"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                confidence=r["confidence"],
                provenance=r["provenance"],
            )
            for r in payload["relations"]
        )
        engine._orchestrator.graph_integrator.restore(
            GraphSnapshot(concepts=concepts, relations=relations)
        )
        store = getattr(engine._orchestrator.resolver, "concept_store", None)
        if store is not None:
            for concept in concepts:
                store.upsert(concept)
        engine._counter = int(payload["document_counter"])
        return engine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/api/ -q`
Expected: PASS (persistence + all earlier api tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/engine.py tests/api/test_persistence.py
git commit -m "feat: add versioned engine save/load with resume equivalence

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: README + executability test

**Files:**
- Create: `README.md`
- Test: `tests/api/test_readme.py`

**Interfaces:**
- Consumes: the Task 3/4 public API exactly as exported.
- Produces: `README.md` whose FIRST fenced ```python block is the quickstart, executed verbatim by the test (keep it the only python block, or keep it first). Every code claim in it is verified lite behavior.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_readme.py`:

```python
"""The README quickstart is executable truth (M6 spec §5): extract the
first fenced python block and run it verbatim in a temp cwd."""

import re
from pathlib import Path


def test_quickstart_block_executes(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[2] / "README.md"
    assert readme.exists(), "README.md missing at repo root"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.DOTALL)
    assert blocks, "README must open with a python quickstart block"
    monkeypatch.chdir(tmp_path)  # the block writes memory.json to cwd
    namespace: dict = {}
    exec(compile(blocks[0], "README.md:quickstart", "exec"), namespace)
    assert namespace["olive"] is not None
    assert namespace["restored"].view().find_concept("olive") is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/api/test_readme.py -q`
Expected: FAIL — `README.md missing at repo root`

- [ ] **Step 3: Write README.md**

Create `README.md` at the repo root with exactly this content:

````markdown
# lattice

**A concept-memory engine.** lattice ingests a stream of documents — LLM
session transcripts, notes, any unit of text — and maintains an accreting,
normalized concept graph that preserves concept identity **across
documents** over time. Its value is not per-document keyphrase extraction;
it is cross-document identity: recognizing that "vector store" in session 5
and "vector database" in session 200 are the same concept, and letting a
graph of concepts and their relations grow as documents arrive.

Every algorithmic stage is a swappable adapter behind a port, and every
default was chosen by benchmark (see the results table below).

## Install

```bash
uv add lattice            # core: dependency-light "lite" profile
uv add "lattice[ml]"      # + spaCy & sentence-transformers for "standard"
uv run python scripts/fetch_models.py   # one-time model download (standard)
```

## Quickstart

```python
from lattice import Engine

engine = Engine()  # "lite" profile — dependency-free; see the toggle below

engine.ingest("Olive oil is a fat prized in Mediterranean cooking.")
engine.ingest("Mediterranean groves grow olive trees for oil.")

view = engine.view()
olive = view.find_concept("olive")

engine.save("memory.json")            # versioned JSON, survives restarts
restored = Engine.load("memory.json")
restored.ingest("Olive presses yield fresh oil each autumn.")
```

`Engine()` defaults to the **lite** profile: the same pipeline topology as
the real thing, with a toy tokenizer and hashing embedder — instant,
dependency-free, right for smoke tests and CI. Note its limits: it only
sees single words of ≥ 4 letters ("olive", not "olive oil").

**For real use, flip one switch:**

```
engine = Engine(profile="standard")
```

## Profiles

| stage | lite | standard | why (evidence) |
|---|---|---|---|
| extractor | token (words ≥ 4 chars) | spaCy noun chunks | real noun phrases (M2 spec) |
| embedder | hashing trigrams | all-MiniLM-L6-v2 | semantic identity (M2/M3 sweeps) |
| scorer | embedding-cosine | embedding-cosine | best salience F1 on Inspec (M2) |
| resolver | embedding-NN @ 0.90 | embedding-NN @ 0.90 | M5's recorded operating point |
| relations | hearst + compound | hearst + compound | above published TExEval-2 band on 2/6 golds (M4) |

Both profiles share one topology — switching changes quality, never
behavior shape. Full control: `Engine.from_config(path_or_dict)` with the
same TOML schema the experiment harness uses.

## Benchmark evidence

| track | benchmark | headline |
|---|---|---|
| salience | Inspec | embedding-cosine F1@10 0.355 vs frequency 0.240 |
| identity | ECB+ / ConEL-2 | embedding-NN beats exact-label B³ F1 on both (0.643 vs 0.608; 0.962 vs 0.939) |
| hierarchy | TExEval-2 (6 English golds) | hearst+compound union ≥ members on 6/6; above the published band on food & food-wordnet |
| integration | ConEL-2 intrinsic | nn@0.90: duplicate-rate 0.117 → 0.015 at coherence 0.931 |

Specs and sweeps: `docs/` (start at
`docs/2026-07-05-lattice-architecture-design.md`).

## Persistence

`engine.save(path)` writes versioned JSON (`format_version: 1`) holding the
fully resolved config, the graph, and the document counter.
`Engine.load(path)` rebuilds the engine and **resumes exactly**: processing
A, B, save, load, C equals processing A, B, C in one run (test-enforced).

## Stability

Pre-1.0: the public contract is `lattice.__all__` — `Engine`, `GraphView`,
`Document`, `Concept`, `Relation`, `GraphDelta`, `GraphSnapshot`,
`__version__`. Minor versions may break it with a changelog note.
Everything below the top level is internal. Save files carry
`format_version` and are readable by any lattice that understands it.
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/api/test_readme.py -q`
Expected: PASS (the quickstart executes: `olive` found, save/load round-trips, resumed ingest works)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add README.md tests/api/test_readme.py
git commit -m "docs: add README with test-executed quickstart

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Packaging — 0.2.0, `lattice[ml]` extra, `py.typed`

**Files:**
- Create: `src/lattice/py.typed` (empty file)
- Modify: `pyproject.toml` (two sanctioned edits)
- Test: `tests/api/test_packaging.py`

**Interfaces:**
- Consumes: Task 3's `__version__ = "0.2.0"` in `lattice/__init__.py`.
- Produces: pyproject `version = "0.2.0"`; `[project.optional-dependencies]` with `ml` mirroring the dependency group; shipped `py.typed`.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_packaging.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/api/test_packaging.py -q`
Expected: FAIL — version_agreement (pyproject still 0.1.0)

- [ ] **Step 3: Apply the packaging changes**

Create empty `src/lattice/py.typed`:

```bash
touch src/lattice/py.typed
```

In `pyproject.toml`: change `version = "0.1.0"` to `version = "0.2.0"`, and insert after the `dependencies = [...]` line:

```toml
[project.optional-dependencies]
ml = ["sentence-transformers>=3.0", "spacy>=3.7", "datasets>=2.19"]
```

(the list must remain byte-identical to `[dependency-groups] ml` — the mirror test enforces it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/api/test_packaging.py -q`
Expected: PASS (3 tests). Then `uv run --no-sync pytest -q` — full suite green (`uv run --no-sync` must still resolve with the modified pyproject).

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add pyproject.toml src/lattice/py.typed tests/api/test_packaging.py
git commit -m "feat: ship py.typed, lattice[ml] extra, version 0.2.0

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Exit criteria — build + full verification

Operational task (orchestrator; nothing committed). Spec §8 is the contract.

- [ ] **Step 1: Wheel build**

```bash
uv build 2>&1 | tail -3
unzip -l dist/lattice-0.2.0-py3-none-any.whl | grep -E "py\.typed|engine\.py|graph_view\.py"
```

Expected: build succeeds; the wheel contains `lattice/py.typed`, `lattice/engine.py`, `lattice/graph_view.py`. (`dist/` is transient — confirm it is not tracked; add to .gitignore only if `git status` shows it.)

- [ ] **Step 2: Full verification**

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
git status
```

Expected: full suite green (≥ 385 existing + ~34 new), lint clean, tree clean (dist/ untracked-or-ignored, never committed).

- [ ] **Step 3: ml path (models are cached on this machine)**

```bash
uv run --no-sync pytest tests/api/test_public_surface.py -q -m ml
```

Expected: `test_standard_profile_constructs_and_ingests` PASSES (spaCy + MiniLM cached since M2/M5) — "olive oil" found as a real noun-phrase concept.

- [ ] **Step 4: Adjudicate spec §8 in the ledger**

1. Consumer suite green, top-level imports only; resume-equivalence exact. 2. README quickstart executed by the suite. 3. Wheel + py.typed + version agreement. 4. No regression (existing tests untouched and green). 5. Standard profile ingests under ml. Record all five verdicts.

---

## Execution Amendments

- Task 3 (defect found during execution, 2026-07-13): the plan-mandated
  `__version__ = "0.2.0"` contradicts the plan's own "existing tests pass
  unchanged" constraint — `tests/test_import.py` hardcodes `"0.1.0"` and the
  plan never audited existing tests for version literals. Sanctioned
  resolution (implementer, same commit e07dea6): update that single
  assertion literal to `"0.2.0"`. Task 7's criterion 4 reads "existing
  tests untouched" with this one sanctioned exception. Lesson: a plan that
  changes a constant must grep the test suite for that constant first.

## Self-Review Notes (already applied)

- Spec coverage: §3→T1+T3+T6, §4.1→T3, §4.2→T2, §4.3→T4, §5→T5, §6 (ValueErrors in T3/T4 tests; JSONDecodeError in T4; ml ImportError path documented in README), §7→every task's tests, §8→T7.
- Measured, not guessed: lite ingest of the three fixture texts was executed against the real pipeline pre-plan (concept labels incl. "olive"/"mediterranean"; "oil"/"fat" excluded at min_length=4; zero relations; deltas 4/3/5 added) and resume-equivalence returned True using exactly the restore semantics T1/T4 specify; `nearest()` tie-breaks on `(-similarity, id)` so store repopulation order cannot break equivalence.
- Type consistency: `_init(config, profile)` is defined in T3 and used by T4's `load`; `FORMAT_VERSION` defined T3, used T4; the README's quickstart names (`engine`, `view`, `olive`, `restored`) match T5's exec-namespace assertions; `__all__` matches the spec §3 list exactly.
- Circular-import audit: no module under `src/` does `from lattice import …` at module scope (verified by grep); `save()` imports `__version__` inside the function body.
- Byte-identical save→load→save holds because: snapshot ordering is sorted (integrator), `json.dumps(..., indent=2, sort_keys=True)` is canonical, config `model_validate(model_dump())` is stable, and the counter/profile/lattice_version fields are copied verbatim.
