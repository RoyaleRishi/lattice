# lattice Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Milestone 1 of the lattice concept-memory engine — core contracts, all ten ports, one trivial adapter per port, registry + config + factory, orchestrator fold, and an end-to-end experiment run on a fixture corpus (spec: `docs/2026-07-05-lattice-architecture-design.md`).

**Architecture:** Hexagonal (ports & adapters) with a thin orchestrator. A pure `core/` domain model; abstract ports; adapters self-register into a registry; a pydantic-validated TOML config drives a factory (the single composition root); `process(document) → GraphDelta` folds over a document stream and mutates stateful stores behind ports. Batch = fold over the stream; no privileged batch path.

**Tech Stack:** Python ≥3.12, uv, pydantic v2, `tomllib` (stdlib), pytest. No other runtime dependencies in Milestone 1.

## Global Constraints

- `requires-python = ">=3.12"`; project managed by **uv** (`uv sync`, `uv run pytest`).
- Runtime dependency: `pydantic>=2.7` **only**. Dev dependency: `pytest>=8.0` only. No numpy, no spaCy, no model downloads in Milestone 1.
- `src/lattice/core/` has **zero external dependencies** — stdlib only (spec §5).
- No generative LLM anywhere on the critical path (spec §3).
- All Milestone-1 adapters are deterministic; break ties lexicographically so reruns are byte-identical (spec §7 reproducibility).
- Registry names use lowercase kebab-case: `"block"`, `"token"`, `"frequency"`, `"hashing"`, `"exact-label"`, `"co-occurrence"`, `"in-memory"`, `"toy"`, `"label-f1"`.
- TDD for every task: write the failing test, watch it fail, implement, watch it pass, commit. Conventional-commit messages (`feat:`, `test:`, `chore:`, `docs:`).
- Every adapter must pass its port's shared contract test suite (LSP backbone, spec §11).

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/lattice/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_import.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: an installed, importable `lattice` package; `uv run pytest` works; git repo initialized with the design docs committed.

- [ ] **Step 1: Initialize git and commit the docs**

```bash
cd /Users/rishis/Desktop/Projects/lattice
git init
git add docs/
git commit -m "docs: add architecture design spec and walking-skeleton plan"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
dist/
*.egg-info/
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "lattice"
version = "0.1.0"
description = "Concept-memory engine: documents in, an accreting normalized concept graph out"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.7"]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lattice"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 4: Create the package and a smoke test**

`src/lattice/__init__.py`:

```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`tests/test_import.py`:

```python
import lattice


def test_package_imports():
    assert lattice.__version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the test**

Run: `uv sync && uv run pytest -v`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml uv.lock src/lattice/__init__.py tests/__init__.py tests/test_import.py
git commit -m "chore: scaffold uv project with src layout and pytest"
```

---

### Task 2: Core domain types

**Files:**
- Create: `src/lattice/core/__init__.py`
- Create: `src/lattice/core/types.py`
- Test: `tests/core/__init__.py`, `tests/core/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces (spec §5 — every later task imports these from `lattice.core.types`):
  - `Document(id: str, kind: str, text: str, timestamp: float, metadata: dict[str, str] = {})`
  - `Unit(id: str, document_id: str, text: str, order: int, kind: str, speaker: str | None = None)`
  - `Mention(surface: str, unit_id: str, span: tuple[int, int], context: str, head: str = "", lemma: str = "")`
  - `ScoredMention(mention: Mention, salience: float, selected: bool)`
  - `Concept(id: str, label: str, embedding: tuple[float, ...], first_seen: str, updated_at: str)`
  - `Relation(type: str, source_id: str, target_id: str, confidence: float, provenance: str)`
  - `Resolution(concept: Concept, mention: ScoredMention, is_new: bool)`
  - `GraphDelta(document_id: str, concepts_added: tuple[Concept, ...], concepts_updated: tuple[Concept, ...], relations_added: tuple[Relation, ...], errors: tuple[str, ...] = ())`
  - `GraphSnapshot(concepts: tuple[Concept, ...], relations: tuple[Relation, ...])`

- [ ] **Step 1: Write the failing tests**

`tests/core/__init__.py`: empty file.

`tests/core/test_types.py`:

```python
import dataclasses

import pytest

from lattice.core.types import (
    Concept,
    Document,
    GraphDelta,
    GraphSnapshot,
    Mention,
    Relation,
    Resolution,
    ScoredMention,
    Unit,
)


def test_document_construction_and_defaults():
    doc = Document(id="d1", kind="note", text="hello", timestamp=1.0)
    assert doc.metadata == {}


def test_document_is_immutable():
    doc = Document(id="d1", kind="note", text="hello", timestamp=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.id = "d2"


def test_unit_speaker_defaults_to_none():
    unit = Unit(id="d1:u0", document_id="d1", text="hello", order=0, kind="block")
    assert unit.speaker is None


def test_mention_span_is_tuple():
    m = Mention(surface="vector", unit_id="d1:u0", span=(4, 10), context="the vector store")
    assert m.span == (4, 10)
    assert m.head == "" and m.lemma == ""


def test_scored_mention_wraps_mention():
    m = Mention(surface="vector", unit_id="d1:u0", span=(0, 6), context="vector")
    sm = ScoredMention(mention=m, salience=0.5, selected=True)
    assert sm.mention.surface == "vector"


def test_concept_equality_is_value_based():
    a = Concept(id="c1", label="vector store", embedding=(1.0, 0.0), first_seen="d1", updated_at="d1")
    b = Concept(id="c1", label="vector store", embedding=(1.0, 0.0), first_seen="d1", updated_at="d1")
    assert a == b


def test_relation_fields():
    r = Relation(type="CO_OCCURS", source_id="c1", target_id="c2", confidence=1.0, provenance="d1")
    assert r.type == "CO_OCCURS"


def test_resolution_marks_new_vs_merged():
    c = Concept(id="c1", label="x", embedding=(1.0,), first_seen="d1", updated_at="d1")
    m = Mention(surface="x", unit_id="d1:u0", span=(0, 1), context="x")
    sm = ScoredMention(mention=m, salience=1.0, selected=True)
    assert Resolution(concept=c, mention=sm, is_new=True).is_new


def test_graph_delta_errors_default_empty():
    delta = GraphDelta(document_id="d1", concepts_added=(), concepts_updated=(), relations_added=())
    assert delta.errors == ()


def test_graph_snapshot_holds_tuples():
    snap = GraphSnapshot(concepts=(), relations=())
    assert snap.concepts == () and snap.relations == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.core'`

- [ ] **Step 3: Implement the types**

`src/lattice/core/__init__.py`: empty file.

`src/lattice/core/types.py`:

```python
"""Domain contracts (spec §5). Pure data types — stdlib only, zero external deps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """One unit of input text: a note, a transcript, any document."""

    id: str
    kind: str  # e.g. "note", "transcript"
    text: str
    timestamp: float  # stream ordering
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Unit:
    """An ordered segment of a document (turn, block, or sentence)."""

    id: str
    document_id: str
    text: str
    order: int
    kind: str  # "turn" | "block" | "sentence"
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class Mention:
    """A candidate concept occurrence inside a unit. `span` is (start, end)
    character offsets into the unit's text."""

    surface: str
    unit_id: str
    span: tuple[int, int]
    context: str
    head: str = ""
    lemma: str = ""


@dataclass(frozen=True, slots=True)
class ScoredMention:
    """A mention with its salience score and whether the scorer selected it."""

    mention: Mention
    salience: float
    selected: bool


@dataclass(frozen=True, slots=True)
class Concept:
    """A canonical node in the concept graph. `first_seen`/`updated_at` are
    document ids (stream provenance, not wall-clock)."""

    id: str
    label: str
    embedding: tuple[float, ...]
    first_seen: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class Relation:
    """A typed edge between two concepts. `provenance` is the document id
    that evidenced the relation."""

    type: str
    source_id: str
    target_id: str
    confidence: float
    provenance: str


@dataclass(frozen=True, slots=True)
class Resolution:
    """The resolver's verdict for one selected mention: the canonical concept
    it maps to, and whether that concept was newly created (is_new=True) or
    merged into an existing one (is_new=False)."""

    concept: Concept
    mention: ScoredMention
    is_new: bool


@dataclass(frozen=True, slots=True)
class GraphDelta:
    """What one document changed in the graph. Errors are always recorded
    here, never silently dropped (spec §8)."""

    document_id: str
    concepts_added: tuple[Concept, ...]
    concepts_updated: tuple[Concept, ...]
    relations_added: tuple[Relation, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """An immutable point-in-time view of the accreting graph (spec §4.2)."""

    concepts: tuple[Concept, ...]
    relations: tuple[Relation, ...]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_types.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lattice/core/ tests/core/
git commit -m "feat: add core domain contracts (Document through GraphSnapshot)"
```

---

### Task 3: Ports

**Files:**
- Create: `src/lattice/ports/__init__.py`
- Create: `src/lattice/ports/segmenter.py`, `extractor.py`, `scorer.py`, `resolver.py`, `relation_inducer.py`, `graph_integrator.py`, `embedder.py`, `concept_store.py`, `dataset.py`, `metric.py`
- Test: `tests/ports/__init__.py`, `tests/ports/test_ports_are_abstract.py`

**Interfaces:**
- Consumes: all types from `lattice.core.types` (Task 2).
- Produces — the ten ABCs every adapter and the orchestrator depend on. Exact signatures (spec §6):
  - `Segmenter.segment(document: Document) -> list[Unit]`
  - `Extractor.extract(units: Sequence[Unit]) -> list[Mention]`
  - `Scorer.score(mentions: Sequence[Mention], units: Sequence[Unit]) -> list[ScoredMention]`
  - `Resolver.resolve(scored_mentions: Sequence[ScoredMention], document: Document) -> list[Resolution]` (receives **only selected** mentions; the orchestrator filters)
  - `RelationInducer.induce(resolutions: Sequence[Resolution], units: Sequence[Unit], document: Document) -> list[Relation]`
  - `GraphIntegrator.apply(resolutions: Sequence[Resolution], relations: Sequence[Relation]) -> None`, `.snapshot() -> GraphSnapshot`, `.reset() -> None`
  - `Embedder.embed(texts: Sequence[str]) -> list[tuple[float, ...]]`, `.dim -> int` (property)
  - `ConceptStore.upsert(concept: Concept) -> None`, `.get(concept_id: str) -> Concept | None`, `.find_by_label(label: str) -> Concept | None`, `.nearest(embedding: tuple[float, ...], k: int = 1) -> list[tuple[Concept, float]]`, `.all() -> list[Concept]`, `.reset() -> None`
  - `Dataset.documents() -> Iterator[Document]`, `.ground_truth() -> dict[str, object]`
  - `Metric.evaluate(snapshot: GraphSnapshot, ground_truth: dict[str, object]) -> dict[str, float]`
- All ten are re-exported from `lattice.ports`.

- [ ] **Step 1: Write the failing test**

`tests/ports/__init__.py`: empty file.

`tests/ports/test_ports_are_abstract.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ports/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.ports'`

- [ ] **Step 3: Implement the ports**

`src/lattice/ports/segmenter.py`:

```python
from abc import ABC, abstractmethod

from lattice.core.types import Document, Unit


class Segmenter(ABC):
    """Splits a document into ordered units (spec §6)."""

    @abstractmethod
    def segment(self, document: Document) -> list[Unit]: ...
```

`src/lattice/ports/extractor.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Mention, Unit


class Extractor(ABC):
    """Finds candidate concept mentions in units (spec §6)."""

    @abstractmethod
    def extract(self, units: Sequence[Unit]) -> list[Mention]: ...
```

`src/lattice/ports/scorer.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit


class Scorer(ABC):
    """Assigns salience to mentions and selects the keepers (spec §6).
    Units provide document context (some scorers need the full document)."""

    @abstractmethod
    def score(self, mentions: Sequence[Mention], units: Sequence[Unit]) -> list[ScoredMention]: ...
```

`src/lattice/ports/resolver.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Document, Resolution, ScoredMention


class Resolver(ABC):
    """Maps selected mentions to canonical concepts, preserving identity
    across documents via its backing ConceptStore (spec §6). Stateful.
    Receives only mentions with selected=True; the orchestrator filters."""

    @abstractmethod
    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]: ...
```

`src/lattice/ports/relation_inducer.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit


class RelationInducer(ABC):
    """Induces typed relations between resolved concepts (spec §6)."""

    @abstractmethod
    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]: ...
```

`src/lattice/ports/graph_integrator.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import GraphSnapshot, Relation, Resolution


class GraphIntegrator(ABC):
    """Applies concepts and relations into the accreting graph (spec §6).
    Stateful; snapshot()/reset() are the reproducibility contract (spec §4.2)."""

    @abstractmethod
    def apply(self, resolutions: Sequence[Resolution], relations: Sequence[Relation]) -> None: ...

    @abstractmethod
    def snapshot(self) -> GraphSnapshot: ...

    @abstractmethod
    def reset(self) -> None: ...
```

`src/lattice/ports/embedder.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Cross-cutting port: text → fixed-dimension vector (spec §6)."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...
```

`src/lattice/ports/concept_store.py`:

```python
from abc import ABC, abstractmethod

from lattice.core.types import Concept


class ConceptStore(ABC):
    """Cross-cutting port: the memory backing the Resolver (spec §6).
    Stateful; reset() is the reproducibility contract (spec §4.2)."""

    @abstractmethod
    def upsert(self, concept: Concept) -> None: ...

    @abstractmethod
    def get(self, concept_id: str) -> Concept | None: ...

    @abstractmethod
    def find_by_label(self, label: str) -> Concept | None: ...

    @abstractmethod
    def nearest(
        self, embedding: tuple[float, ...], k: int = 1
    ) -> list[tuple[Concept, float]]: ...

    @abstractmethod
    def all(self) -> list[Concept]: ...

    @abstractmethod
    def reset(self) -> None: ...
```

`src/lattice/ports/dataset.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Iterator

from lattice.core.types import Document


class Dataset(ABC):
    """Harness port: yields documents in stream order plus ground truth (spec §6, §9).
    The ground-truth shape is metric-specific; each Metric documents what it expects."""

    @abstractmethod
    def documents(self) -> Iterator[Document]: ...

    @abstractmethod
    def ground_truth(self) -> dict[str, object]: ...
```

`src/lattice/ports/metric.py`:

```python
from abc import ABC, abstractmethod

from lattice.core.types import GraphSnapshot


class Metric(ABC):
    """Harness port: scores a graph snapshot against ground truth (spec §6, §9)."""

    @abstractmethod
    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]: ...
```

`src/lattice/ports/__init__.py`:

```python
from lattice.ports.concept_store import ConceptStore
from lattice.ports.dataset import Dataset
from lattice.ports.embedder import Embedder
from lattice.ports.extractor import Extractor
from lattice.ports.graph_integrator import GraphIntegrator
from lattice.ports.metric import Metric
from lattice.ports.relation_inducer import RelationInducer
from lattice.ports.resolver import Resolver
from lattice.ports.scorer import Scorer
from lattice.ports.segmenter import Segmenter

__all__ = [
    "ConceptStore",
    "Dataset",
    "Embedder",
    "Extractor",
    "GraphIntegrator",
    "Metric",
    "RelationInducer",
    "Resolver",
    "Scorer",
    "Segmenter",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ports/ -v`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lattice/ports/ tests/ports/
git commit -m "feat: add the ten abstract ports"
```

---

### Task 4: Registry

**Files:**
- Create: `src/lattice/registry/__init__.py`
- Create: `src/lattice/registry/registry.py`
- Create: `tests/conftest.py`
- Test: `tests/registry/__init__.py`, `tests/registry/test_registry.py`

**Interfaces:**
- Consumes: port classes (any `type`) — no import of specific ports needed.
- Produces (spec §7.1 — every adapter task and the factory use these):
  - `register(port: type, name: str)` — class decorator; validates `issubclass`, rejects duplicate `(port, name)`.
  - `lookup(port: type, name: str) -> type` — raises `RegistryError` with known names listed if missing.
  - `available(port: type) -> dict[str, type]`
  - `RegistryError(Exception)`
  - Module-level `_REGISTRY: dict[type, dict[str, type]]` (tests snapshot/restore it via the `clean_registry` fixture in `tests/conftest.py`).

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:

```python
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
```

`tests/registry/__init__.py`: empty file.

`tests/registry/test_registry.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/registry/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.registry'`

- [ ] **Step 3: Implement the registry**

`src/lattice/registry/__init__.py`: empty file.

`src/lattice/registry/registry.py`:

```python
"""Adapter registration and lookup (spec §7.1). Adapters self-register under
(port, name) at import time; the factory resolves names from config here.
Adding a new algorithm = one new decorated class, nothing else changes."""

_REGISTRY: dict[type, dict[str, type]] = {}


class RegistryError(Exception):
    pass


def register(port: type, name: str):
    """Class decorator: `@register(Scorer, "mderank")`."""

    def decorator(adapter_cls: type) -> type:
        if not issubclass(adapter_cls, port):
            raise RegistryError(
                f"{adapter_cls.__name__} does not implement {port.__name__}"
            )
        adapters = _REGISTRY.setdefault(port, {})
        if name in adapters:
            raise RegistryError(
                f"duplicate adapter {name!r} for port {port.__name__}"
            )
        adapters[name] = adapter_cls
        return adapter_cls

    return decorator


def lookup(port: type, name: str) -> type:
    adapters = _REGISTRY.get(port, {})
    if name not in adapters:
        known = ", ".join(sorted(adapters)) or "none registered"
        raise RegistryError(
            f"no adapter {name!r} for port {port.__name__} (known: {known})"
        )
    return adapters[name]


def available(port: type) -> dict[str, type]:
    return dict(_REGISTRY.get(port, {}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/registry/ -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lattice/registry/ tests/registry/ tests/conftest.py
git commit -m "feat: add adapter registry with register/lookup/available"
```

---

### Task 5: Config schema and TOML loader

**Files:**
- Create: `src/lattice/config/__init__.py`
- Create: `src/lattice/config/schema.py`
- Create: `src/lattice/config/loader.py`
- Test: `tests/config/__init__.py`, `tests/config/test_schema.py`, `tests/config/test_loader.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pydantic only).
- Produces (spec §7.2 — the factory in Task 15 and the harness in Task 17 consume these):
  - `AdapterSpec(name: str, params: dict[str, Any] = {})` — pydantic model, `extra="forbid"`.
  - `RunPolicy(on_error: Literal["fail", "skip"] = "fail", seed: int = 0)` — `extra="forbid"`.
  - `RunConfig` — required `AdapterSpec` fields `segmenter, extractor, scorer, resolver, relation_inducer, graph_integrator`; defaulted fields `embedder=AdapterSpec(name="hashing")`, `concept_store=AdapterSpec(name="in-memory")`, `run=RunPolicy()`; `extra="forbid"`.
  - `load_config(path: str | Path, model: type[M] = RunConfig) -> M` — reads TOML via `tomllib`, validates with `model`.

- [ ] **Step 1: Write the failing tests**

`tests/config/__init__.py`: empty file.

`tests/config/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from lattice.config.schema import AdapterSpec, RunConfig, RunPolicy


def _minimal_config() -> dict:
    return {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "frequency"},
        "resolver": {"name": "exact-label"},
        "relation_inducer": {"name": "co-occurrence"},
        "graph_integrator": {"name": "in-memory"},
    }


def test_minimal_config_validates_with_defaults():
    config = RunConfig.model_validate(_minimal_config())
    assert config.embedder == AdapterSpec(name="hashing")
    assert config.concept_store == AdapterSpec(name="in-memory")
    assert config.run == RunPolicy(on_error="fail", seed=0)


def test_params_default_to_empty_dict():
    assert AdapterSpec(name="x").params == {}


def test_missing_required_section_rejected():
    data = _minimal_config()
    del data["scorer"]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_unknown_key_rejected():
    data = _minimal_config()
    data["scorrer"] = {"name": "typo"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_invalid_on_error_rejected():
    data = _minimal_config()
    data["run"] = {"on_error": "explode"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)
```

`tests/config/test_loader.py`:

```python
import pytest
from pydantic import ValidationError

from lattice.config.loader import load_config

VALID_TOML = """
[segmenter]
name = "block"

[extractor]
name = "token"
[extractor.params]
min_length = 4

[scorer]
name = "frequency"

[resolver]
name = "exact-label"

[relation_inducer]
name = "co-occurrence"

[graph_integrator]
name = "in-memory"

[run]
on_error = "skip"
"""


def test_load_valid_toml(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(VALID_TOML)
    config = load_config(path)
    assert config.segmenter.name == "block"
    assert config.extractor.params == {"min_length": 4}
    assert config.run.on_error == "skip"


def test_load_invalid_toml_raises_validation_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[segmenter]\nname = "block"\n')  # missing required sections
    with pytest.raises(ValidationError):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.config'`

- [ ] **Step 3: Implement schema and loader**

`src/lattice/config/__init__.py`: empty file.

`src/lattice/config/schema.py`:

```python
"""Declarative run configuration (spec §7.2). A run is one adapter name +
params per port. `extra="forbid"` everywhere so config typos fail loudly
instead of silently changing an experiment."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_error: Literal["fail", "skip"] = "fail"  # spec §8
    seed: int = 0  # stamped for reproducibility (spec §7)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmenter: AdapterSpec
    extractor: AdapterSpec
    scorer: AdapterSpec
    resolver: AdapterSpec
    relation_inducer: AdapterSpec
    graph_integrator: AdapterSpec
    embedder: AdapterSpec = AdapterSpec(name="hashing")
    concept_store: AdapterSpec = AdapterSpec(name="in-memory")
    run: RunPolicy = RunPolicy()
```

`src/lattice/config/loader.py`:

```python
import tomllib
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from lattice.config.schema import RunConfig

M = TypeVar("M", bound=BaseModel)


def load_config(path: str | Path, model: type[M] = RunConfig) -> M:
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return model.model_validate(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/ -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lattice/config/ tests/config/
git commit -m "feat: add pydantic run config schema and TOML loader"
```

---

### Task 6: Test helpers + Segmenter contract + block adapter

**Files:**
- Create: `tests/helpers.py`
- Create: `tests/contracts/__init__.py`, `tests/contracts/segmenter_contract.py`
- Create: `src/lattice/adapters/__init__.py`, `src/lattice/adapters/segmenter/__init__.py`, `src/lattice/adapters/segmenter/block.py`
- Test: `tests/adapters/__init__.py`, `tests/adapters/test_block_segmenter.py`

**Interfaces:**
- Consumes: `Document`, `Unit` (Task 2); `Segmenter` port (Task 3); `register` (Task 4).
- Produces:
  - `tests/helpers.py` fixture factories used by all later test tasks — exact signatures:
    - `make_document(id: str = "doc-x", text: str = "some text", timestamp: float = 1.0) -> Document`
    - `make_unit(id: str = "doc-x:u0", document_id: str = "doc-x", text: str = "some text", order: int = 0) -> Unit`
    - `make_mention(surface: str = "concept", unit_id: str = "doc-x:u0", span: tuple[int, int] = (0, 7), context: str = "concept text") -> Mention`
    - `make_scored_mention(surface: str = "concept", unit_id: str = "doc-x:u0", salience: float = 1.0, selected: bool = True) -> ScoredMention`
    - `make_concept(id: str = "c1", label: str = "concept", dim: int = 4, first_seen: str = "doc-x") -> Concept`
    - `make_resolution(concept: Concept | None = None, surface: str = "concept", unit_id: str = "doc-x:u0", is_new: bool = True) -> Resolution`
  - `SegmenterContract` — subclass and implement `make_segmenter() -> Segmenter`; used by every future segmenter adapter.
  - `BlockSegmenter` registered as `(Segmenter, "block")`; unit ids are `f"{document.id}:u{i}"`, kind `"block"`.
  - `lattice/adapters/__init__.py` imports adapter modules for registration side effects (each adapter task appends one import line).

- [ ] **Step 1: Write the shared test helpers**

`tests/helpers.py`:

```python
"""Fixture factories shared by contract and adapter tests."""

from lattice.core.types import (
    Concept,
    Document,
    Mention,
    Resolution,
    ScoredMention,
    Unit,
)


def make_document(
    id: str = "doc-x", text: str = "some text", timestamp: float = 1.0
) -> Document:
    return Document(id=id, kind="note", text=text, timestamp=timestamp)


def make_unit(
    id: str = "doc-x:u0",
    document_id: str = "doc-x",
    text: str = "some text",
    order: int = 0,
) -> Unit:
    return Unit(id=id, document_id=document_id, text=text, order=order, kind="block")


def make_mention(
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    span: tuple[int, int] = (0, 7),
    context: str = "concept text",
) -> Mention:
    return Mention(
        surface=surface, unit_id=unit_id, span=span, context=context,
        head=surface, lemma=surface,
    )


def make_scored_mention(
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    salience: float = 1.0,
    selected: bool = True,
) -> ScoredMention:
    return ScoredMention(
        mention=make_mention(surface=surface, unit_id=unit_id),
        salience=salience,
        selected=selected,
    )


def make_concept(
    id: str = "c1", label: str = "concept", dim: int = 4, first_seen: str = "doc-x"
) -> Concept:
    return Concept(
        id=id,
        label=label,
        embedding=(1.0,) + (0.0,) * (dim - 1),
        first_seen=first_seen,
        updated_at=first_seen,
    )


def make_resolution(
    concept: Concept | None = None,
    surface: str = "concept",
    unit_id: str = "doc-x:u0",
    is_new: bool = True,
) -> Resolution:
    return Resolution(
        concept=concept or make_concept(id=f"c:{surface}", label=surface),
        mention=make_scored_mention(surface=surface, unit_id=unit_id),
        is_new=is_new,
    )
```

- [ ] **Step 2: Write the segmenter contract and the failing adapter test**

`tests/contracts/__init__.py`: empty file.

`tests/contracts/segmenter_contract.py`:

```python
"""Contract every Segmenter adapter must satisfy (spec §11: LSP backbone).
Subclass in the adapter's test module and implement make_segmenter()."""

from lattice.ports import Segmenter
from tests.helpers import make_document


class SegmenterContract:
    def make_segmenter(self) -> Segmenter:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_units_reference_source_document(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert units, "expected at least one unit for non-empty text"
        assert all(u.document_id == doc.id for u in units)

    def test_units_are_ordered_from_zero(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert [u.order for u in units] == list(range(len(units)))

    def test_unit_ids_are_unique(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert len({u.id for u in units}) == len(units)

    def test_empty_text_yields_no_units(self):
        assert self.make_segmenter().segment(make_document(text="")) == []
```

`tests/adapters/__init__.py`: empty file.

`tests/adapters/test_block_segmenter.py`:

```python
from lattice.adapters.segmenter.block import BlockSegmenter
from tests.contracts.segmenter_contract import SegmenterContract
from tests.helpers import make_document


class TestBlockSegmenter(SegmenterContract):
    def make_segmenter(self) -> BlockSegmenter:
        return BlockSegmenter()

    def test_splits_on_blank_lines(self):
        units = self.make_segmenter().segment(
            make_document(text="First block.\n\nSecond block.")
        )
        assert [u.text for u in units] == ["First block.", "Second block."]

    def test_strips_whitespace_and_drops_empty_blocks(self):
        units = self.make_segmenter().segment(
            make_document(text="  First.  \n\n\n\n  Second.  ")
        )
        assert [u.text for u in units] == ["First.", "Second."]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_block_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters'`

- [ ] **Step 4: Implement the block segmenter**

`src/lattice/adapters/__init__.py`:

```python
"""Importing this package registers all built-in adapters (spec §7.1)."""

from lattice.adapters.segmenter import block  # noqa: F401
```

`src/lattice/adapters/segmenter/__init__.py`: empty file.

`src/lattice/adapters/segmenter/block.py`:

```python
from lattice.core.types import Document, Unit
from lattice.ports import Segmenter
from lattice.registry.registry import register


@register(Segmenter, "block")
class BlockSegmenter(Segmenter):
    """Splits document text into blocks on blank lines."""

    def segment(self, document: Document) -> list[Unit]:
        blocks = [b.strip() for b in document.text.split("\n\n")]
        blocks = [b for b in blocks if b]
        return [
            Unit(
                id=f"{document.id}:u{i}",
                document_id=document.id,
                text=block,
                order=i,
                kind="block",
            )
            for i, block in enumerate(blocks)
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_block_segmenter.py -v`
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add tests/helpers.py tests/contracts/ tests/adapters/ src/lattice/adapters/
git commit -m "feat: add segmenter contract suite and block segmenter adapter"
```

---

### Task 7: Extractor contract + token adapter

**Files:**
- Create: `tests/contracts/extractor_contract.py`
- Create: `src/lattice/adapters/extractor/__init__.py`, `src/lattice/adapters/extractor/token.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_token_extractor.py`

**Interfaces:**
- Consumes: `Mention`, `Unit` (Task 2); `Extractor` port (Task 3); `register` (Task 4); `make_unit` (Task 6).
- Produces:
  - `ExtractorContract` — subclass implements `make_extractor() -> Extractor`.
  - `TokenExtractor(min_length: int = 4)` registered as `(Extractor, "token")`. Surfaces are lowercased; spans index into the unit's text.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/extractor_contract.py`:

```python
"""Contract every Extractor adapter must satisfy. Spans must index into the
owning unit's text, and the spanned slice must match the surface
case-insensitively (adapters may normalize surface case)."""

from lattice.ports import Extractor
from tests.helpers import make_unit


class ExtractorContract:
    def make_extractor(self) -> Extractor:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_mentions_reference_their_units(self):
        units = [
            make_unit(id="d:u0", text="Vector stores index embeddings."),
            make_unit(id="d:u1", text="Encoders produce embeddings.", order=1),
        ]
        mentions = self.make_extractor().extract(units)
        assert mentions, "expected mentions from non-trivial text"
        unit_ids = {u.id for u in units}
        assert all(m.unit_id in unit_ids for m in mentions)

    def test_spans_slice_the_unit_text(self):
        units = [make_unit(id="d:u0", text="Vector stores index embeddings.")]
        mentions = self.make_extractor().extract(units)
        by_id = {u.id: u for u in units}
        for m in mentions:
            start, end = m.span
            assert 0 <= start < end <= len(by_id[m.unit_id].text)
            assert by_id[m.unit_id].text[start:end].lower() == m.surface.lower()

    def test_no_units_yields_no_mentions(self):
        assert self.make_extractor().extract([]) == []
```

`tests/adapters/test_token_extractor.py`:

```python
from lattice.adapters.extractor.token import TokenExtractor
from tests.contracts.extractor_contract import ExtractorContract
from tests.helpers import make_unit


class TestTokenExtractor(ExtractorContract):
    def make_extractor(self) -> TokenExtractor:
        return TokenExtractor()

    def test_short_words_filtered_by_min_length(self):
        mentions = TokenExtractor(min_length=5).extract(
            [make_unit(text="The vector store maps text")]
        )
        assert {m.surface for m in mentions} == {"vector", "store"}

    def test_surfaces_are_lowercased(self):
        mentions = TokenExtractor().extract([make_unit(text="Vector Embeddings")])
        assert {m.surface for m in mentions} == {"vector", "embeddings"}

    def test_every_occurrence_is_a_mention(self):
        mentions = TokenExtractor().extract([make_unit(text="store the store")])
        assert [m.surface for m in mentions] == ["store", "store"]
        assert mentions[0].span != mentions[1].span
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_token_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.extractor'`

- [ ] **Step 3: Implement the token extractor**

`src/lattice/adapters/extractor/__init__.py`: empty file.

`src/lattice/adapters/extractor/token.py`:

```python
import re
from collections.abc import Sequence

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register

_WORD = re.compile(r"[A-Za-z][A-Za-z-]+")


@register(Extractor, "token")
class TokenExtractor(Extractor):
    """Trivial walking-skeleton extractor: every word of at least min_length
    characters is a candidate mention. Real noun-phrase extraction arrives in
    Milestone 2; this exists so the skeleton runs with zero NLP deps."""

    def __init__(self, min_length: int = 4):
        self.min_length = min_length

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            for match in _WORD.finditer(unit.text):
                word = match.group()
                if len(word) < self.min_length:
                    continue
                mentions.append(
                    Mention(
                        surface=word.lower(),
                        unit_id=unit.id,
                        span=(match.start(), match.end()),
                        context=unit.text,
                        head=word.lower(),
                        lemma=word.lower(),
                    )
                )
        return mentions
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.extractor import token  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_token_extractor.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/extractor_contract.py tests/adapters/test_token_extractor.py src/lattice/adapters/
git commit -m "feat: add extractor contract suite and token extractor adapter"
```

---

### Task 8: Embedder contract + hashing adapter

**Files:**
- Create: `tests/contracts/embedder_contract.py`
- Create: `src/lattice/adapters/embedder/__init__.py`, `src/lattice/adapters/embedder/hashing.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_hashing_embedder.py`

**Interfaces:**
- Consumes: `Embedder` port (Task 3); `register` (Task 4).
- Produces:
  - `EmbedderContract` — subclass implements `make_embedder() -> Embedder`.
  - `HashingEmbedder(dim: int = 64)` registered as `(Embedder, "hashing")`; deterministic character-trigram hashing, L2-normalized vectors.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/embedder_contract.py`:

```python
"""Contract every Embedder adapter must satisfy."""

from lattice.ports import Embedder


class EmbedderContract:
    def make_embedder(self) -> Embedder:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_one_vector_per_text(self):
        vectors = self.make_embedder().embed(["vector store", "encoder"])
        assert len(vectors) == 2

    def test_vectors_have_declared_dim(self):
        embedder = self.make_embedder()
        [vector] = embedder.embed(["vector store"])
        assert len(vector) == embedder.dim

    def test_embedding_is_deterministic(self):
        embedder = self.make_embedder()
        assert embedder.embed(["vector store"]) == embedder.embed(["vector store"])

    def test_empty_input_yields_empty_list(self):
        assert self.make_embedder().embed([]) == []
```

`tests/adapters/test_hashing_embedder.py`:

```python
import math

from lattice.adapters.embedder.hashing import HashingEmbedder
from tests.contracts.embedder_contract import EmbedderContract


class TestHashingEmbedder(EmbedderContract):
    def make_embedder(self) -> HashingEmbedder:
        return HashingEmbedder(dim=32)

    def test_dim_is_configurable(self):
        assert HashingEmbedder(dim=16).dim == 16

    def test_vectors_are_unit_normalized(self):
        [vector] = HashingEmbedder(dim=32).embed(["vector store"])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_different_texts_differ(self):
        embedder = HashingEmbedder(dim=32)
        [a] = embedder.embed(["vector store"])
        [b] = embedder.embed(["completely unrelated phrase"])
        assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_hashing_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.embedder'`

- [ ] **Step 3: Implement the hashing embedder**

`src/lattice/adapters/embedder/__init__.py`: empty file.

`src/lattice/adapters/embedder/hashing.py`:

```python
import hashlib
import math
from collections.abc import Sequence

from lattice.ports import Embedder
from lattice.registry.registry import register


@register(Embedder, "hashing")
class HashingEmbedder(Embedder):
    """Deterministic character-trigram hashing embedder. Not semantically
    meaningful — a stand-in so the skeleton runs without a model download.
    A sentence-transformer adapter replaces it for real experiments (M2)."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dim
        padded = f" {text.lower()} "
        for i in range(max(len(padded) - 2, 0)):
            trigram = padded[i : i + 3]
            digest = hashlib.md5(trigram.encode()).hexdigest()
            vec[int(digest, 16) % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return tuple(vec)
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.embedder import hashing  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_hashing_embedder.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/embedder_contract.py tests/adapters/test_hashing_embedder.py src/lattice/adapters/
git commit -m "feat: add embedder contract suite and hashing embedder adapter"
```

---

### Task 9: Scorer contract + frequency adapter

**Files:**
- Create: `tests/contracts/scorer_contract.py`
- Create: `src/lattice/adapters/scorer/__init__.py`, `src/lattice/adapters/scorer/frequency.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_frequency_scorer.py`

**Interfaces:**
- Consumes: `ScoredMention` (Task 2); `Scorer` port (Task 3); `register` (Task 4); `make_mention`, `make_unit` (Task 6).
- Produces:
  - `ScorerContract` — subclass implements `make_scorer() -> Scorer`.
  - `FrequencyScorer(top_k: int = 10)` registered as `(Scorer, "frequency")`; salience = surface frequency / max frequency; selects all mentions of the `top_k` most frequent surfaces, ties broken alphabetically.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/scorer_contract.py`:

```python
"""Contract every Scorer adapter must satisfy: score every mention it is
given (no drops, no additions), mark selection via the boolean flag."""

import math

from lattice.ports import Scorer
from tests.helpers import make_mention, make_unit


class ScorerContract:
    def make_scorer(self) -> Scorer:
        raise NotImplementedError("subclass must provide the adapter under test")

    def _fixture(self):
        unit = make_unit(id="d:u0", text="vector store vector")
        mentions = [
            make_mention(surface="vector", unit_id="d:u0", span=(0, 6)),
            make_mention(surface="store", unit_id="d:u0", span=(7, 12)),
            make_mention(surface="vector", unit_id="d:u0", span=(13, 19)),
        ]
        return mentions, [unit]

    def test_every_mention_scored_exactly_once(self):
        mentions, units = self._fixture()
        scored = self.make_scorer().score(mentions, units)
        assert sorted(sm.mention.span for sm in scored) == sorted(m.span for m in mentions)

    def test_salience_is_finite(self):
        mentions, units = self._fixture()
        assert all(math.isfinite(sm.salience) for sm in self.make_scorer().score(mentions, units))

    def test_empty_input_yields_empty_output(self):
        assert self.make_scorer().score([], []) == []
```

`tests/adapters/test_frequency_scorer.py`:

```python
from lattice.adapters.scorer.frequency import FrequencyScorer
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


def _mentions(*surfaces: str):
    return [
        make_mention(surface=s, unit_id="d:u0", span=(i * 10, i * 10 + len(s)))
        for i, s in enumerate(surfaces)
    ]


class TestFrequencyScorer(ScorerContract):
    def make_scorer(self) -> FrequencyScorer:
        return FrequencyScorer()

    def test_most_frequent_surface_has_max_salience(self):
        scored = FrequencyScorer().score(
            _mentions("vector", "vector", "store"), [make_unit()]
        )
        by_surface = {sm.mention.surface: sm.salience for sm in scored}
        assert by_surface["vector"] == 1.0
        assert by_surface["store"] == 0.5

    def test_top_k_limits_selected_surfaces(self):
        scored = FrequencyScorer(top_k=1).score(
            _mentions("vector", "vector", "store"), [make_unit()]
        )
        selected = {sm.mention.surface for sm in scored if sm.selected}
        assert selected == {"vector"}

    def test_ties_break_alphabetically(self):
        scored = FrequencyScorer(top_k=1).score(
            _mentions("zebra", "apple"), [make_unit()]
        )
        selected = {sm.mention.surface for sm in scored if sm.selected}
        assert selected == {"apple"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_frequency_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer'`

- [ ] **Step 3: Implement the frequency scorer**

`src/lattice/adapters/scorer/__init__.py`: empty file.

`src/lattice/adapters/scorer/frequency.py`:

```python
from collections import Counter
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.ports import Scorer
from lattice.registry.registry import register


@register(Scorer, "frequency")
class FrequencyScorer(Scorer):
    """Trivial walking-skeleton scorer: salience = surface frequency
    normalized by the max frequency. Selects every mention of the top_k most
    frequent surfaces; ties break alphabetically for determinism."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        counts = Counter(m.surface for m in mentions)
        max_count = max(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m,
                salience=counts[m.surface] / max_count,
                selected=m.surface in top_surfaces,
            )
            for m in mentions
        ]
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.scorer import frequency  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_frequency_scorer.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/scorer_contract.py tests/adapters/test_frequency_scorer.py src/lattice/adapters/
git commit -m "feat: add scorer contract suite and frequency scorer adapter"
```

---

### Task 10: ConceptStore contract + in-memory adapter

**Files:**
- Create: `tests/contracts/concept_store_contract.py`
- Create: `src/lattice/adapters/concept_store/__init__.py`, `src/lattice/adapters/concept_store/in_memory.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_in_memory_concept_store.py`

**Interfaces:**
- Consumes: `Concept` (Task 2); `ConceptStore` port (Task 3); `register` (Task 4); `make_concept` (Task 6).
- Produces:
  - `ConceptStoreContract` — subclass implements `make_store() -> ConceptStore`.
  - `InMemoryConceptStore()` registered as `(ConceptStore, "in-memory")`; brute-force cosine `nearest`, deterministic ordering (score desc, then concept id).

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/concept_store_contract.py`:

```python
"""Contract every ConceptStore adapter must satisfy. The store is the
resolver's memory: identity must survive upserts, and reset() must fully
clear state between experiment runs (spec §4.2)."""

from lattice.core.types import Concept
from lattice.ports import ConceptStore
from tests.helpers import make_concept


class ConceptStoreContract:
    def make_store(self) -> ConceptStore:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_upsert_then_get(self):
        store = self.make_store()
        concept = make_concept(id="c1", label="vector store")
        store.upsert(concept)
        assert store.get("c1") == concept

    def test_get_missing_returns_none(self):
        assert self.make_store().get("nope") is None

    def test_find_by_label(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="vector store"))
        found = store.find_by_label("vector store")
        assert found is not None and found.id == "c1"

    def test_upsert_same_id_replaces(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="old label"))
        store.upsert(make_concept(id="c1", label="new label"))
        assert store.get("c1").label == "new label"
        assert store.find_by_label("old label") is None
        assert len(store.all()) == 1

    def test_nearest_returns_most_similar_first(self):
        store = self.make_store()
        a = Concept(id="a", label="a", embedding=(1.0, 0.0), first_seen="d", updated_at="d")
        b = Concept(id="b", label="b", embedding=(0.0, 1.0), first_seen="d", updated_at="d")
        store.upsert(a)
        store.upsert(b)
        [(top, score)] = store.nearest((0.9, 0.1), k=1)
        assert top.id == "a"
        assert score > 0.9

    def test_reset_clears_everything(self):
        store = self.make_store()
        store.upsert(make_concept(id="c1", label="vector store"))
        store.reset()
        assert store.all() == []
        assert store.get("c1") is None
        assert store.find_by_label("vector store") is None
```

`tests/adapters/test_in_memory_concept_store.py`:

```python
from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from tests.contracts.concept_store_contract import ConceptStoreContract


class TestInMemoryConceptStore(ConceptStoreContract):
    def make_store(self) -> InMemoryConceptStore:
        return InMemoryConceptStore()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_in_memory_concept_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.concept_store'`

- [ ] **Step 3: Implement the in-memory store**

`src/lattice/adapters/concept_store/__init__.py`: empty file.

`src/lattice/adapters/concept_store/in_memory.py`:

```python
import math
from collections.abc import Sequence

from lattice.core.types import Concept
from lattice.ports import ConceptStore
from lattice.registry.registry import register


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@register(ConceptStore, "in-memory")
class InMemoryConceptStore(ConceptStore):
    """Dict-backed store with brute-force cosine nearest-neighbour. Fine for
    experiments; a vector-index adapter can replace it behind the same port."""

    def __init__(self):
        self._by_id: dict[str, Concept] = {}
        self._id_by_label: dict[str, str] = {}

    def upsert(self, concept: Concept) -> None:
        old = self._by_id.get(concept.id)
        if old is not None:
            self._id_by_label.pop(old.label, None)
        self._by_id[concept.id] = concept
        self._id_by_label[concept.label] = concept.id

    def get(self, concept_id: str) -> Concept | None:
        return self._by_id.get(concept_id)

    def find_by_label(self, label: str) -> Concept | None:
        concept_id = self._id_by_label.get(label)
        return self._by_id.get(concept_id) if concept_id is not None else None

    def nearest(
        self, embedding: tuple[float, ...], k: int = 1
    ) -> list[tuple[Concept, float]]:
        scored = [
            (concept, _cosine(embedding, concept.embedding))
            for concept in self._by_id.values()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]

    def all(self) -> list[Concept]:
        return list(self._by_id.values())

    def reset(self) -> None:
        self._by_id.clear()
        self._id_by_label.clear()
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.concept_store import in_memory  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_in_memory_concept_store.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/concept_store_contract.py tests/adapters/test_in_memory_concept_store.py src/lattice/adapters/
git commit -m "feat: add concept store contract suite and in-memory adapter"
```

---

### Task 11: Resolver contract + exact-label adapter

**Files:**
- Create: `tests/contracts/resolver_contract.py`
- Create: `src/lattice/adapters/resolver/__init__.py`, `src/lattice/adapters/resolver/exact_label.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_exact_label_resolver.py`

**Interfaces:**
- Consumes: `Concept`, `Resolution` (Task 2); `Resolver`, `Embedder`, `ConceptStore` ports (Task 3); `register` (Task 4); `make_document`, `make_scored_mention` (Task 6); `HashingEmbedder` (Task 8); `InMemoryConceptStore` (Task 10).
- Produces:
  - `ResolverContract` — subclass implements `make_resolver() -> Resolver` (fully wired with its own embedder + store; the resolver must keep memory across `resolve()` calls).
  - `ExactLabelResolver(embedder: Embedder, concept_store: ConceptStore)` registered as `(Resolver, "exact-label")`. Constructor parameter names `embedder` and `concept_store` are load-bearing: the factory (Task 15) injects shared dependencies by parameter name. Concept ids are `uuid5(NAMESPACE_URL, f"lattice:concept:{label}")`.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/resolver_contract.py`:

```python
"""Contract every Resolver adapter must satisfy. The heart of lattice
(spec §1): the same surface in different documents must resolve to the SAME
concept, with identity provenance preserved."""

from lattice.ports import Resolver
from tests.helpers import make_document, make_scored_mention


class ResolverContract:
    def make_resolver(self) -> Resolver:
        raise NotImplementedError(
            "subclass must provide a fully wired adapter (own embedder + store)"
        )

    def test_new_surface_creates_new_concept(self):
        resolver = self.make_resolver()
        [resolution] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d1")
        )
        assert resolution.is_new
        assert resolution.concept.first_seen == "d1"
        assert resolution.concept.updated_at == "d1"

    def test_same_surface_across_documents_resolves_to_one_concept(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d1:u0")],
            make_document(id="d1"),
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vector store", unit_id="d2:u0")],
            make_document(id="d2"),
        )
        assert r2.concept.id == r1.concept.id
        assert r1.is_new and not r2.is_new
        assert r2.concept.first_seen == "d1"
        assert r2.concept.updated_at == "d2"

    def test_empty_input_yields_no_resolutions(self):
        assert self.make_resolver().resolve([], make_document(id="d1")) == []
```

`tests/adapters/test_exact_label_resolver.py`:

```python
from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from tests.contracts.resolver_contract import ResolverContract
from tests.helpers import make_document, make_scored_mention


class TestExactLabelResolver(ResolverContract):
    def make_resolver(self) -> ExactLabelResolver:
        return ExactLabelResolver(
            embedder=HashingEmbedder(dim=16),
            concept_store=InMemoryConceptStore(),
        )

    def test_distinct_surfaces_create_distinct_concepts(self):
        resolver = self.make_resolver()
        r1, r2 = resolver.resolve(
            [
                make_scored_mention(surface="vector store"),
                make_scored_mention(surface="encoder"),
            ],
            make_document(id="d1"),
        )
        assert r1.concept.id != r2.concept.id

    def test_labels_are_normalized_lowercase(self):
        resolver = self.make_resolver()
        [r1] = resolver.resolve(
            [make_scored_mention(surface="Vector Store")], make_document(id="d1")
        )
        [r2] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d2")
        )
        assert r1.concept.id == r2.concept.id
        assert r1.concept.label == "vector store"

    def test_concept_gets_embedding_from_embedder(self):
        resolver = self.make_resolver()
        [r] = resolver.resolve(
            [make_scored_mention(surface="vector store")], make_document(id="d1")
        )
        assert len(r.concept.embedding) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_exact_label_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.resolver'`

- [ ] **Step 3: Implement the exact-label resolver**

`src/lattice/adapters/resolver/__init__.py`: empty file.

`src/lattice/adapters/resolver/exact_label.py`:

```python
import uuid
from collections.abc import Sequence
from dataclasses import replace

from lattice.core.types import Concept, Document, Resolution, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@register(Resolver, "exact-label")
class ExactLabelResolver(Resolver):
    """Trivial walking-skeleton resolver: normalizes the surface to lowercase
    and merges only on exact label match against the store. Embedding-NN and
    clustering resolvers arrive in Milestone 3 behind the same port."""

    def __init__(self, embedder: Embedder, concept_store: ConceptStore):
        self.embedder = embedder
        self.concept_store = concept_store

    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]:
        resolutions: list[Resolution] = []
        for scored_mention in scored_mentions:
            label = scored_mention.mention.surface.strip().lower()
            existing = self.concept_store.find_by_label(label)
            if existing is not None:
                updated = replace(existing, updated_at=document.id)
                self.concept_store.upsert(updated)
                resolutions.append(
                    Resolution(concept=updated, mention=scored_mention, is_new=False)
                )
            else:
                [embedding] = self.embedder.embed([label])
                concept = Concept(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{label}")),
                    label=label,
                    embedding=embedding,
                    first_seen=document.id,
                    updated_at=document.id,
                )
                self.concept_store.upsert(concept)
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.resolver import exact_label  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_exact_label_resolver.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/resolver_contract.py tests/adapters/test_exact_label_resolver.py src/lattice/adapters/
git commit -m "feat: add resolver contract suite and exact-label resolver adapter"
```

---

### Task 12: RelationInducer contract + co-occurrence adapter

**Files:**
- Create: `tests/contracts/relation_inducer_contract.py`
- Create: `src/lattice/adapters/relation_inducer/__init__.py`, `src/lattice/adapters/relation_inducer/co_occurrence.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_co_occurrence_inducer.py`

**Interfaces:**
- Consumes: `Relation`, `Resolution` (Task 2); `RelationInducer` port (Task 3); `register` (Task 4); `make_document`, `make_resolution`, `make_unit` (Task 6).
- Produces:
  - `RelationInducerContract` — subclass implements `make_inducer() -> RelationInducer`.
  - `CoOccurrenceInducer()` registered as `(RelationInducer, "co-occurrence")`; emits one `CO_OCCURS` relation per unordered pair of distinct concepts mentioned in the same unit, pair endpoints sorted by concept id, output sorted, `confidence=1.0`, `provenance=document.id`.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/relation_inducer_contract.py`:

```python
"""Contract every RelationInducer adapter must satisfy."""

from lattice.ports import RelationInducer
from tests.helpers import make_document, make_resolution, make_unit


class RelationInducerContract:
    def make_inducer(self) -> RelationInducer:
        raise NotImplementedError("subclass must provide the adapter under test")

    def _fixture(self):
        document = make_document(id="d1")
        units = [make_unit(id="d1:u0", document_id="d1", text="vector store and encoder")]
        resolutions = [
            make_resolution(surface="vector store", unit_id="d1:u0"),
            make_resolution(surface="encoder", unit_id="d1:u0"),
        ]
        return resolutions, units, document

    def test_relations_reference_resolved_concepts(self):
        resolutions, units, document = self._fixture()
        relations = self.make_inducer().induce(resolutions, units, document)
        concept_ids = {r.concept.id for r in resolutions}
        for relation in relations:
            assert relation.source_id in concept_ids
            assert relation.target_id in concept_ids

    def test_provenance_is_document_id(self):
        resolutions, units, document = self._fixture()
        relations = self.make_inducer().induce(resolutions, units, document)
        assert all(r.provenance == document.id for r in relations)

    def test_empty_resolutions_yield_no_relations(self):
        _, units, document = self._fixture()
        assert self.make_inducer().induce([], units, document) == []
```

`tests/adapters/test_co_occurrence_inducer.py`:

```python
from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestCoOccurrenceInducer(RelationInducerContract):
    def make_inducer(self) -> CoOccurrenceInducer:
        return CoOccurrenceInducer()

    def test_same_unit_concepts_co_occur(self):
        relations = CoOccurrenceInducer().induce(
            [
                make_resolution(surface="vector store", unit_id="d1:u0"),
                make_resolution(surface="encoder", unit_id="d1:u0"),
            ],
            [make_unit(id="d1:u0", document_id="d1")],
            make_document(id="d1"),
        )
        assert len(relations) == 1
        assert relations[0].type == "CO_OCCURS"

    def test_cross_unit_concepts_do_not_co_occur(self):
        relations = CoOccurrenceInducer().induce(
            [
                make_resolution(surface="vector store", unit_id="d1:u0"),
                make_resolution(surface="encoder", unit_id="d1:u1"),
            ],
            [
                make_unit(id="d1:u0", document_id="d1"),
                make_unit(id="d1:u1", document_id="d1", order=1),
            ],
            make_document(id="d1"),
        )
        assert relations == []

    def test_pair_emitted_once_with_sorted_endpoints(self):
        resolutions = [
            make_resolution(surface="b-concept", unit_id="d1:u0"),
            make_resolution(surface="a-concept", unit_id="d1:u0"),
            make_resolution(surface="b-concept", unit_id="d1:u0"),
        ]
        relations = CoOccurrenceInducer().induce(
            resolutions, [make_unit(id="d1:u0", document_id="d1")], make_document(id="d1")
        )
        assert len(relations) == 1
        assert relations[0].source_id < relations[0].target_id

    def test_same_concept_twice_yields_no_self_relation(self):
        resolutions = [
            make_resolution(surface="vector store", unit_id="d1:u0"),
            make_resolution(surface="vector store", unit_id="d1:u0"),
        ]
        relations = CoOccurrenceInducer().induce(
            resolutions, [make_unit(id="d1:u0", document_id="d1")], make_document(id="d1")
        )
        assert relations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_co_occurrence_inducer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.relation_inducer'`

- [ ] **Step 3: Implement the co-occurrence inducer**

`src/lattice/adapters/relation_inducer/__init__.py`: empty file.

`src/lattice/adapters/relation_inducer/co_occurrence.py`:

```python
from collections.abc import Sequence
from itertools import combinations

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register


@register(RelationInducer, "co-occurrence")
class CoOccurrenceInducer(RelationInducer):
    """Trivial walking-skeleton inducer: one CO_OCCURS relation per unordered
    pair of distinct concepts mentioned in the same unit. Hearst-pattern and
    head-modifier IS_A inducers arrive in Milestone 4 behind the same port."""

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        concepts_by_unit: dict[str, set[str]] = {}
        for resolution in resolutions:
            unit_id = resolution.mention.mention.unit_id
            concepts_by_unit.setdefault(unit_id, set()).add(resolution.concept.id)
        pairs: set[tuple[str, str]] = set()
        for concept_ids in concepts_by_unit.values():
            pairs.update(combinations(sorted(concept_ids), 2))
        return [
            Relation(
                type="CO_OCCURS",
                source_id=source_id,
                target_id=target_id,
                confidence=1.0,
                provenance=document.id,
            )
            for source_id, target_id in sorted(pairs)
        ]
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.relation_inducer import co_occurrence  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_co_occurrence_inducer.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/relation_inducer_contract.py tests/adapters/test_co_occurrence_inducer.py src/lattice/adapters/
git commit -m "feat: add relation inducer contract suite and co-occurrence adapter"
```

---

### Task 13: GraphIntegrator contract + in-memory adapter

**Files:**
- Create: `tests/contracts/graph_integrator_contract.py`
- Create: `src/lattice/adapters/graph_integrator/__init__.py`, `src/lattice/adapters/graph_integrator/in_memory.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_in_memory_graph_integrator.py`

**Interfaces:**
- Consumes: `GraphSnapshot`, `Relation`, `Resolution`, `Concept` (Task 2); `GraphIntegrator` port (Task 3); `register` (Task 4); `make_concept`, `make_resolution` (Task 6).
- Produces:
  - `GraphIntegratorContract` — subclass implements `make_integrator() -> GraphIntegrator`.
  - `InMemoryGraphIntegrator()` registered as `(GraphIntegrator, "in-memory")`; concepts keyed by id (last write wins), relations keyed by `(type, source_id, target_id)`; snapshots sorted for determinism.

- [ ] **Step 1: Write the contract and the failing adapter test**

`tests/contracts/graph_integrator_contract.py`:

```python
"""Contract every GraphIntegrator adapter must satisfy: the accreting graph
dedupes by identity, and snapshot()/reset() honor spec §4.2."""

from dataclasses import replace

from lattice.core.types import Relation
from lattice.ports import GraphIntegrator
from tests.helpers import make_concept, make_resolution


class GraphIntegratorContract:
    def make_integrator(self) -> GraphIntegrator:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_applied_concepts_and_relations_appear_in_snapshot(self):
        integrator = self.make_integrator()
        r1 = make_resolution(surface="vector store")
        r2 = make_resolution(surface="encoder")
        relation = Relation(
            type="CO_OCCURS",
            source_id=r1.concept.id,
            target_id=r2.concept.id,
            confidence=1.0,
            provenance="d1",
        )
        integrator.apply([r1, r2], [relation])
        snapshot = integrator.snapshot()
        assert {c.id for c in snapshot.concepts} == {r1.concept.id, r2.concept.id}
        assert snapshot.relations == (relation,)

    def test_reapplying_same_concept_does_not_duplicate(self):
        integrator = self.make_integrator()
        concept = make_concept(id="c1", label="vector store")
        integrator.apply([make_resolution(concept=concept)], [])
        integrator.apply([make_resolution(concept=concept, is_new=False)], [])
        assert len(integrator.snapshot().concepts) == 1

    def test_updated_concept_replaces_previous_version(self):
        integrator = self.make_integrator()
        v1 = make_concept(id="c1", label="vector store", first_seen="d1")
        integrator.apply([make_resolution(concept=v1)], [])
        v2 = replace(v1, updated_at="d2")
        integrator.apply([make_resolution(concept=v2, is_new=False)], [])
        [stored] = integrator.snapshot().concepts
        assert stored.updated_at == "d2"

    def test_reset_empties_the_graph(self):
        integrator = self.make_integrator()
        integrator.apply([make_resolution(surface="vector store")], [])
        integrator.reset()
        snapshot = integrator.snapshot()
        assert snapshot.concepts == () and snapshot.relations == ()
```

`tests/adapters/test_in_memory_graph_integrator.py`:

```python
from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from tests.contracts.graph_integrator_contract import GraphIntegratorContract
from tests.helpers import make_resolution


class TestInMemoryGraphIntegrator(GraphIntegratorContract):
    def make_integrator(self) -> InMemoryGraphIntegrator:
        return InMemoryGraphIntegrator()

    def test_snapshot_is_sorted_for_determinism(self):
        integrator = self.make_integrator()
        rb = make_resolution(surface="b-concept")
        ra = make_resolution(surface="a-concept")
        integrator.apply([rb, ra], [])
        snapshot = integrator.snapshot()
        ids = [c.id for c in snapshot.concepts]
        assert ids == sorted(ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/test_in_memory_graph_integrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.graph_integrator'`

- [ ] **Step 3: Implement the in-memory integrator**

`src/lattice/adapters/graph_integrator/__init__.py`: empty file.

`src/lattice/adapters/graph_integrator/in_memory.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Concept, GraphSnapshot, Relation, Resolution
from lattice.ports import GraphIntegrator
from lattice.registry.registry import register


@register(GraphIntegrator, "in-memory")
class InMemoryGraphIntegrator(GraphIntegrator):
    """Dict-backed accreting graph. Concepts are keyed by id (last write
    wins); relations by (type, source, target). Snapshots are sorted so
    identical runs produce identical snapshots (spec §7 reproducibility)."""

    def __init__(self):
        self._concepts: dict[str, Concept] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}

    def apply(
        self, resolutions: Sequence[Resolution], relations: Sequence[Relation]
    ) -> None:
        for resolution in resolutions:
            self._concepts[resolution.concept.id] = resolution.concept
        for relation in relations:
            key = (relation.type, relation.source_id, relation.target_id)
            self._relations[key] = relation

    def snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            concepts=tuple(
                sorted(self._concepts.values(), key=lambda c: c.id)
            ),
            relations=tuple(
                sorted(
                    self._relations.values(),
                    key=lambda r: (r.type, r.source_id, r.target_id),
                )
            ),
        )

    def reset(self) -> None:
        self._concepts.clear()
        self._relations.clear()
```

Note: `sorted()` on `Relation` values needs no `__lt__` because the `key=` function is always provided.

- [ ] **Step 4: Append the registration import**

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.graph_integrator import in_memory as graph_in_memory  # noqa: F401
```

(The alias avoids shadowing the concept-store `in_memory` import in the same module.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_in_memory_graph_integrator.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add tests/contracts/graph_integrator_contract.py tests/adapters/test_in_memory_graph_integrator.py src/lattice/adapters/
git commit -m "feat: add graph integrator contract suite and in-memory adapter"
```

---

### Task 14: Orchestrator

**Files:**
- Create: `src/lattice/orchestrator/__init__.py`
- Create: `src/lattice/orchestrator/orchestrator.py`
- Test: `tests/orchestrator/__init__.py`, `tests/orchestrator/test_orchestrator.py`

**Interfaces:**
- Consumes: all six stage ports (Task 3); all trivial adapters (Tasks 6–13) for the integration tests.
- Produces (spec §4.1, §8 — the factory and harness consume this):
  - `Orchestrator(*, segmenter, extractor, scorer, resolver, relation_inducer, graph_integrator, on_error: Literal["fail", "skip"] = "fail")` — keyword-only; stage adapters exposed as same-named attributes.
  - `.process(document: Document) -> GraphDelta` — filters to `selected=True` before the resolver; a concept created then re-mentioned in the same document appears only in `concepts_added`.
  - `.process_stream(documents: Iterable[Document]) -> list[GraphDelta]` — the fold; batch has no separate code path.
  - `.snapshot() -> GraphSnapshot` — delegates to the integrator.

- [ ] **Step 1: Write the failing tests**

`tests/orchestrator/__init__.py`: empty file.

`tests/orchestrator/test_orchestrator.py`:

```python
import pytest

from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.extractor.token import TokenExtractor
from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.orchestrator.orchestrator import Orchestrator
from lattice.ports import Extractor
from tests.helpers import make_document


def build_orchestrator(**overrides) -> Orchestrator:
    embedder = HashingEmbedder(dim=16)
    store = InMemoryConceptStore()
    stages = {
        "segmenter": BlockSegmenter(),
        "extractor": TokenExtractor(min_length=4),
        "scorer": FrequencyScorer(top_k=10),
        "resolver": ExactLabelResolver(embedder=embedder, concept_store=store),
        "relation_inducer": CoOccurrenceInducer(),
        "graph_integrator": InMemoryGraphIntegrator(),
        "on_error": "fail",
    }
    stages.update(overrides)
    return Orchestrator(**stages)


class ExplodingExtractor(Extractor):
    def extract(self, units):
        raise RuntimeError("boom")


def test_process_returns_delta_with_new_concepts():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(
        make_document(id="d1", text="The vector store indexes embeddings.")
    )
    labels = {c.label for c in delta.concepts_added}
    assert {"vector", "store", "indexes", "embeddings"} == labels
    assert delta.concepts_updated == ()
    assert delta.errors == ()
    assert delta.document_id == "d1"


def test_process_produces_co_occurrence_relations():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(make_document(id="d1", text="vector store"))
    assert len(delta.relations_added) == 1
    assert delta.relations_added[0].type == "CO_OCCURS"


def test_second_document_merges_instead_of_duplicating():
    orchestrator = build_orchestrator()
    orchestrator.process(make_document(id="d1", text="vector store"))
    delta2 = orchestrator.process(make_document(id="d2", text="vector store"))
    assert delta2.concepts_added == ()
    assert {c.label for c in delta2.concepts_updated} == {"vector", "store"}
    assert len(orchestrator.snapshot().concepts) == 2


def test_repeated_surface_in_one_document_counts_as_added_only():
    orchestrator = build_orchestrator()
    delta = orchestrator.process(
        make_document(id="d1", text="vector store\n\nvector store")
    )
    assert {c.label for c in delta.concepts_added} == {"vector", "store"}
    assert delta.concepts_updated == ()


def test_process_stream_folds_in_order():
    orchestrator = build_orchestrator()
    deltas = orchestrator.process_stream(
        [
            make_document(id="d1", text="vector store", timestamp=1.0),
            make_document(id="d2", text="vector encoder", timestamp=2.0),
        ]
    )
    assert [d.document_id for d in deltas] == ["d1", "d2"]
    assert {c.label for c in deltas[1].concepts_added} == {"encoder"}
    assert {c.label for c in deltas[1].concepts_updated} == {"vector"}


def test_unselected_mentions_never_reach_the_graph():
    orchestrator = build_orchestrator(scorer=FrequencyScorer(top_k=1))
    delta = orchestrator.process(
        make_document(id="d1", text="vector vector store")
    )
    assert {c.label for c in delta.concepts_added} == {"vector"}


def test_on_error_fail_raises():
    orchestrator = build_orchestrator(extractor=ExplodingExtractor())
    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.process(make_document(id="d1"))


def test_on_error_skip_records_error_and_continues():
    orchestrator = build_orchestrator(
        extractor=ExplodingExtractor(), on_error="skip"
    )
    deltas = orchestrator.process_stream(
        [make_document(id="d1"), make_document(id="d2")]
    )
    assert len(deltas) == 2
    for delta in deltas:
        assert delta.concepts_added == ()
        assert len(delta.errors) == 1
        assert "boom" in delta.errors[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/orchestrator/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.orchestrator'`

- [ ] **Step 3: Implement the orchestrator**

`src/lattice/orchestrator/__init__.py`: empty file.

`src/lattice/orchestrator/orchestrator.py`:

```python
"""Thin orchestrator (spec §4): one document in, one GraphDelta out.
Batch is a fold over the stream — process_stream() just calls process()
per document; there is no separate batch code path (spec §4.1)."""

from collections.abc import Iterable
from typing import Literal

from lattice.core.types import Concept, Document, GraphDelta, GraphSnapshot
from lattice.ports import (
    Extractor,
    GraphIntegrator,
    RelationInducer,
    Resolver,
    Scorer,
    Segmenter,
)


class Orchestrator:
    """Runs the six-stage pipeline over one document at a time.

    Error policy (spec §8): "fail" re-raises the stage exception (a crash
    never silently shrinks the scored corpus); "skip" records the error in
    the GraphDelta and moves on (one poison document can't halt the stream).
    Under "skip", stages that mutated their stores before the failing stage
    keep those mutations — transactional deltas are deferred past M1.
    """

    def __init__(
        self,
        *,
        segmenter: Segmenter,
        extractor: Extractor,
        scorer: Scorer,
        resolver: Resolver,
        relation_inducer: RelationInducer,
        graph_integrator: GraphIntegrator,
        on_error: Literal["fail", "skip"] = "fail",
    ) -> None:
        self.segmenter = segmenter
        self.extractor = extractor
        self.scorer = scorer
        self.resolver = resolver
        self.relation_inducer = relation_inducer
        self.graph_integrator = graph_integrator
        self.on_error = on_error

    def process(self, document: Document) -> GraphDelta:
        try:
            units = self.segmenter.segment(document)
            mentions = self.extractor.extract(units)
            scored = self.scorer.score(mentions, units)
            selected = [sm for sm in scored if sm.selected]
            resolutions = self.resolver.resolve(selected, document)
            relations = self.relation_inducer.induce(resolutions, units, document)
            self.graph_integrator.apply(resolutions, relations)
        except Exception as exc:
            if self.on_error == "fail":
                raise
            return GraphDelta(
                document_id=document.id,
                concepts_added=(),
                concepts_updated=(),
                relations_added=(),
                errors=(f"{type(exc).__name__}: {exc}",),
            )

        added: dict[str, Concept] = {}
        updated: dict[str, Concept] = {}
        for resolution in resolutions:
            if resolution.is_new:
                added[resolution.concept.id] = resolution.concept
            else:
                updated[resolution.concept.id] = resolution.concept
        # A concept created and then re-mentioned within the same document
        # counts as added, not updated.
        for concept_id in added:
            updated.pop(concept_id, None)

        return GraphDelta(
            document_id=document.id,
            concepts_added=tuple(added.values()),
            concepts_updated=tuple(updated.values()),
            relations_added=tuple(relations),
            errors=(),
        )

    def process_stream(self, documents: Iterable[Document]) -> list[GraphDelta]:
        return [self.process(document) for document in documents]

    def snapshot(self) -> GraphSnapshot:
        return self.graph_integrator.snapshot()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/orchestrator/ -v`
Expected: `8 passed`

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `uv run pytest`
Expected: all tests pass, zero failures.

- [ ] **Step 6: Commit**

```bash
git add src/lattice/orchestrator/ tests/orchestrator/
git commit -m "feat: add orchestrator with stream fold and fail/skip error policy"
```

---

### Task 15: Factory (composition root)

**Files:**
- Create: `src/lattice/config/factory.py`
- Test: `tests/config/test_factory.py`

**Interfaces:**
- Consumes: `RunConfig`, `AdapterSpec` (Task 5); `lookup` (Task 4); `Orchestrator` (Task 14); all ports (Task 3); `lattice.adapters` package import for registration (Tasks 6–13).
- Produces (spec §7.3 — the harness consumes these):
  - `instantiate(port: type, spec: AdapterSpec, shared: dict[str, object] | None = None)` — registry lookup + construct with `spec.params`, injecting any `shared` entry whose key matches a constructor parameter name (explicit params win over injection).
  - `build_orchestrator(config: RunConfig) -> Orchestrator` — builds embedder + concept store first, injects them into stage adapters that declare `embedder`/`concept_store` parameters, wires the orchestrator with `on_error` from `config.run`. The only place concrete classes meet names.

- [ ] **Step 1: Write the failing tests**

`tests/config/test_factory.py`:

```python
import pytest

from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.ports import Scorer
from lattice.registry.registry import RegistryError


def make_run_config(**overrides) -> RunConfig:
    data = {
        "segmenter": {"name": "block"},
        "extractor": {"name": "token"},
        "scorer": {"name": "frequency"},
        "resolver": {"name": "exact-label"},
        "relation_inducer": {"name": "co-occurrence"},
        "graph_integrator": {"name": "in-memory"},
    }
    data.update(overrides)
    return RunConfig.model_validate(data)


def test_build_orchestrator_wires_configured_adapters():
    orchestrator = build_orchestrator(make_run_config())
    assert isinstance(orchestrator.segmenter, BlockSegmenter)
    assert isinstance(orchestrator.scorer, FrequencyScorer)


def test_params_reach_the_adapter():
    config = make_run_config(scorer={"name": "frequency", "params": {"top_k": 3}})
    orchestrator = build_orchestrator(config)
    assert orchestrator.scorer.top_k == 3


def test_shared_dependencies_injected_by_parameter_name():
    config = make_run_config(embedder={"name": "hashing", "params": {"dim": 32}})
    orchestrator = build_orchestrator(config)
    assert orchestrator.resolver.embedder.dim == 32
    assert orchestrator.resolver.concept_store is not None


def test_on_error_policy_flows_from_config():
    config = make_run_config(run={"on_error": "skip"})
    assert build_orchestrator(config).on_error == "skip"


def test_unknown_adapter_name_raises_registry_error():
    config = make_run_config(scorer={"name": "does-not-exist"})
    with pytest.raises(RegistryError, match="does-not-exist"):
        build_orchestrator(config)


def test_instantiate_explicit_params_win_over_injection():
    scorer = instantiate(
        Scorer,
        AdapterSpec(name="frequency", params={"top_k": 7}),
        shared={"top_k": 99},
    )
    assert scorer.top_k == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `lattice.config.factory`

- [ ] **Step 3: Implement the factory**

`src/lattice/config/factory.py`:

```python
"""Composition root (spec §7.3): validated config → registry lookup →
instantiate with params → inject shared deps → wired orchestrator.
This is the single DIP composition root — the only place concrete adapter
classes are resolved from names."""

import inspect

import lattice.adapters  # noqa: F401  (importing registers all built-in adapters)
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.orchestrator.orchestrator import Orchestrator
from lattice.ports import (
    ConceptStore,
    Embedder,
    Extractor,
    GraphIntegrator,
    RelationInducer,
    Resolver,
    Scorer,
    Segmenter,
)
from lattice.registry.registry import lookup


def instantiate(
    port: type, spec: AdapterSpec, shared: dict[str, object] | None = None
):
    """Instantiate the registered adapter for `spec`. Any `shared` dependency
    whose key matches a constructor parameter name is injected, unless the
    config already supplies that param explicitly."""
    adapter_cls = lookup(port, spec.name)
    kwargs = dict(spec.params)
    parameters = inspect.signature(adapter_cls.__init__).parameters
    for name, dependency in (shared or {}).items():
        if name in parameters and name not in kwargs:
            kwargs[name] = dependency
    return adapter_cls(**kwargs)


def build_orchestrator(config: RunConfig) -> Orchestrator:
    embedder = instantiate(Embedder, config.embedder)
    concept_store = instantiate(ConceptStore, config.concept_store)
    shared = {"embedder": embedder, "concept_store": concept_store}
    return Orchestrator(
        segmenter=instantiate(Segmenter, config.segmenter, shared),
        extractor=instantiate(Extractor, config.extractor, shared),
        scorer=instantiate(Scorer, config.scorer, shared),
        resolver=instantiate(Resolver, config.resolver, shared),
        relation_inducer=instantiate(RelationInducer, config.relation_inducer, shared),
        graph_integrator=instantiate(GraphIntegrator, config.graph_integrator, shared),
        on_error=config.run.on_error,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_factory.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lattice/config/factory.py tests/config/test_factory.py
git commit -m "feat: add factory composition root with shared-dependency injection"
```

---

### Task 16: Dataset + Metric contracts and toy adapters

**Files:**
- Create: `tests/contracts/dataset_contract.py`, `tests/contracts/metric_contract.py`
- Create: `src/lattice/adapters/dataset/__init__.py`, `src/lattice/adapters/dataset/toy.py`
- Create: `src/lattice/adapters/metric/__init__.py`, `src/lattice/adapters/metric/label_f1.py`
- Modify: `src/lattice/adapters/__init__.py` (add two import lines)
- Test: `tests/adapters/test_toy_dataset.py`, `tests/adapters/test_label_f1_metric.py`

**Interfaces:**
- Consumes: `Document`, `GraphSnapshot` (Task 2); `Dataset`, `Metric` ports (Task 3); `register` (Task 4); `make_concept` (Task 6).
- Produces:
  - `DatasetContract` — subclass implements `make_dataset() -> Dataset`.
  - `MetricContract` — subclass implements `make_metric() -> Metric` and `make_ground_truth() -> dict`.
  - `ToyDataset()` registered as `(Dataset, "toy")` — 3 fixture documents (ids `doc-1`..`doc-3`), ground truth `{"concept_labels": ["vector", "store", "embeddings", "encoder"]}`.
  - `LabelF1()` registered as `(Metric, "label-f1")` — returns `{"precision": p, "recall": r, "f1": f}` comparing lowercase snapshot labels against `ground_truth["concept_labels"]`.

- [ ] **Step 1: Write the contracts and failing adapter tests**

`tests/contracts/dataset_contract.py`:

```python
"""Contract every Dataset adapter must satisfy: documents arrive in stream
order with unique ids, and ground truth is a dict."""

from lattice.ports import Dataset


class DatasetContract:
    def make_dataset(self) -> Dataset:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_yields_at_least_one_document(self):
        assert list(self.make_dataset().documents())

    def test_document_ids_are_unique(self):
        docs = list(self.make_dataset().documents())
        assert len({d.id for d in docs}) == len(docs)

    def test_documents_arrive_in_timestamp_order(self):
        timestamps = [d.timestamp for d in self.make_dataset().documents()]
        assert timestamps == sorted(timestamps)

    def test_ground_truth_is_a_dict(self):
        assert isinstance(self.make_dataset().ground_truth(), dict)
```

`tests/contracts/metric_contract.py`:

```python
"""Contract every Metric adapter must satisfy: float values, no crash on an
empty snapshot."""

from lattice.core.types import GraphSnapshot
from lattice.ports import Metric


class MetricContract:
    def make_metric(self) -> Metric:
        raise NotImplementedError("subclass must provide the adapter under test")

    def make_ground_truth(self) -> dict:
        raise NotImplementedError("subclass must provide matching ground truth")

    def test_returns_dict_of_floats(self):
        result = self.make_metric().evaluate(
            GraphSnapshot(concepts=(), relations=()), self.make_ground_truth()
        )
        assert result and all(isinstance(v, float) for v in result.values())

    def test_handles_empty_snapshot_without_crashing(self):
        self.make_metric().evaluate(
            GraphSnapshot(concepts=(), relations=()), self.make_ground_truth()
        )
```

`tests/adapters/test_toy_dataset.py`:

```python
from lattice.adapters.dataset.toy import ToyDataset
from tests.contracts.dataset_contract import DatasetContract


class TestToyDataset(DatasetContract):
    def make_dataset(self) -> ToyDataset:
        return ToyDataset()

    def test_has_three_documents(self):
        assert len(list(ToyDataset().documents())) == 3

    def test_ground_truth_lists_expected_labels(self):
        truth = ToyDataset().ground_truth()
        assert truth["concept_labels"] == ["vector", "store", "embeddings", "encoder"]
```

`tests/adapters/test_label_f1_metric.py`:

```python
from lattice.adapters.metric.label_f1 import LabelF1
from lattice.core.types import GraphSnapshot
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


def snapshot_of(*labels: str) -> GraphSnapshot:
    return GraphSnapshot(
        concepts=tuple(make_concept(id=f"c:{l}", label=l) for l in labels),
        relations=(),
    )


class TestLabelF1(MetricContract):
    def make_metric(self) -> LabelF1:
        return LabelF1()

    def make_ground_truth(self) -> dict:
        return {"concept_labels": ["vector", "store"]}

    def test_perfect_match_scores_one(self):
        result = LabelF1().evaluate(snapshot_of("vector", "store"), self.make_ground_truth())
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_disjoint_labels_score_zero(self):
        result = LabelF1().evaluate(snapshot_of("apple", "zebra"), self.make_ground_truth())
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_partial_overlap(self):
        result = LabelF1().evaluate(snapshot_of("vector", "zebra"), self.make_ground_truth())
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_empty_snapshot_scores_zero(self):
        result = LabelF1().evaluate(snapshot_of(), self.make_ground_truth())
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_gold_labels_compared_case_insensitively(self):
        result = LabelF1().evaluate(
            snapshot_of("vector", "store"), {"concept_labels": ["Vector", "STORE"]}
        )
        assert result["f1"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/test_toy_dataset.py tests/adapters/test_label_f1_metric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.dataset'`

- [ ] **Step 3: Implement the toy dataset and label-F1 metric**

`src/lattice/adapters/dataset/__init__.py`: empty file.

`src/lattice/adapters/dataset/toy.py`:

```python
from collections.abc import Iterator

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "toy")
class ToyDataset(Dataset):
    """Three tiny in-code documents for the walking skeleton. Real benchmark
    datasets (Inspec, SemEval, ECB+) arrive in Milestones 2-4 behind the
    same port."""

    _DOCS = (
        Document(
            id="doc-1",
            kind="note",
            timestamp=1.0,
            text="The vector store indexes embeddings.\n\nThe vector store returns neighbors.",
        ),
        Document(
            id="doc-2",
            kind="note",
            timestamp=2.0,
            text="Embeddings come from the encoder model.",
        ),
        Document(
            id="doc-3",
            kind="note",
            timestamp=3.0,
            text="The vector store holds embeddings from the encoder.",
        ),
    )

    def documents(self) -> Iterator[Document]:
        yield from self._DOCS

    def ground_truth(self) -> dict[str, object]:
        return {"concept_labels": ["vector", "store", "embeddings", "encoder"]}
```

`src/lattice/adapters/metric/__init__.py`: empty file.

`src/lattice/adapters/metric/label_f1.py`:

```python
from lattice.core.types import GraphSnapshot
from lattice.ports import Metric
from lattice.registry.registry import register


@register(Metric, "label-f1")
class LabelF1(Metric):
    """Precision/recall/F1 of snapshot concept labels against
    ground_truth["concept_labels"]. Case-insensitive set comparison."""

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        gold = {str(label).lower() for label in ground_truth.get("concept_labels", [])}
        predicted = {concept.label.lower() for concept in snapshot.concepts}
        true_positives = len(gold & predicted)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.dataset import toy  # noqa: F401
from lattice.adapters.metric import label_f1  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_toy_dataset.py tests/adapters/test_label_f1_metric.py -v`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/contracts/dataset_contract.py tests/contracts/metric_contract.py tests/adapters/test_toy_dataset.py tests/adapters/test_label_f1_metric.py src/lattice/adapters/
git commit -m "feat: add dataset/metric contracts, toy dataset, and label-F1 metric"
```

---

### Task 17: Harness runner + first end-to-end run

**Files:**
- Create: `src/lattice/harness/__init__.py`
- Create: `src/lattice/harness/runner.py`
- Create: `src/lattice/harness/__main__.py`
- Create: `configs/walking-skeleton.toml`
- Test: `tests/harness/__init__.py`, `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: `RunConfig`, `AdapterSpec`, `load_config` (Task 5); `build_orchestrator`, `instantiate` (Task 15); `Dataset`, `Metric` ports (Task 3); `ToyDataset`, `LabelF1` via registry (Task 16).
- Produces (spec §9 run flow — the deliverable of Milestone 1):
  - `ExperimentConfig(RunConfig)` — adds `dataset: AdapterSpec` and `metrics: list[AdapterSpec] = []`.
  - `RunReport(config: dict[str, Any], documents_processed: int, errors: tuple[str, ...], metrics: dict[str, dict[str, float]])` — frozen dataclass; `config` is the fully resolved config stamp (spec §7 reproducibility).
  - `run_experiment(config: ExperimentConfig) -> RunReport`
  - `run_from_path(path: str | Path) -> RunReport`
  - `python -m lattice.harness configs/walking-skeleton.toml` prints the report as JSON.

- [ ] **Step 1: Write the run config**

`configs/walking-skeleton.toml`:

```toml
# First end-to-end run: every port filled by its trivial walking-skeleton
# adapter, evaluated on the toy fixture corpus.

[segmenter]
name = "block"

[extractor]
name = "token"
[extractor.params]
min_length = 4

[scorer]
name = "frequency"
[scorer.params]
top_k = 10

[resolver]
name = "exact-label"

[relation_inducer]
name = "co-occurrence"

[graph_integrator]
name = "in-memory"

[embedder]
name = "hashing"
[embedder.params]
dim = 64

[concept_store]
name = "in-memory"

[run]
on_error = "fail"
seed = 0

[dataset]
name = "toy"

[[metrics]]
name = "label-f1"
```

- [ ] **Step 2: Write the failing tests**

`tests/harness/__init__.py`: empty file.

`tests/harness/test_runner.py`:

```python
from lattice.harness.runner import ExperimentConfig, run_experiment, run_from_path

CONFIG_PATH = "configs/walking-skeleton.toml"


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token", "params": {"min_length": 4}},
            "scorer": {"name": "frequency", "params": {"top_k": 10}},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "dataset": {"name": "toy"},
            "metrics": [{"name": "label-f1"}],
        }
    )


def test_run_experiment_end_to_end():
    report = run_experiment(_experiment_config())
    assert report.documents_processed == 3
    assert report.errors == ()
    # every gold label is found by the trivial pipeline on the toy corpus
    assert report.metrics["label-f1"]["recall"] == 1.0
    # the trivial extractor over-generates, so precision is imperfect
    assert 0.0 < report.metrics["label-f1"]["precision"] < 1.0


def test_report_stamps_the_resolved_config():
    report = run_experiment(_experiment_config())
    assert report.config["scorer"]["name"] == "frequency"
    assert report.config["scorer"]["params"] == {"top_k": 10}
    assert report.config["run"]["seed"] == 0
    assert report.config["embedder"]["name"] == "hashing"  # default stamped too


def test_rerunning_the_same_config_reproduces_the_report():
    assert run_experiment(_experiment_config()) == run_experiment(_experiment_config())


def test_run_from_path_loads_toml_and_runs():
    report = run_from_path(CONFIG_PATH)
    assert report.documents_processed == 3
    assert report.metrics["label-f1"]["recall"] == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/harness/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.harness'`

- [ ] **Step 4: Implement the harness**

`src/lattice/harness/__init__.py`: empty file.

`src/lattice/harness/runner.py`:

```python
"""Experiment runner (spec §9): load dataset → fold process() over its
documents → snapshot the graph → score with metrics → emit a report stamped
with the fully resolved config (spec §7 reproducibility)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.loader import load_config
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.ports import Dataset, Metric


class ExperimentConfig(RunConfig):
    dataset: AdapterSpec
    metrics: list[AdapterSpec] = Field(default_factory=list)


@dataclass(frozen=True)
class RunReport:
    config: dict[str, Any]
    documents_processed: int
    errors: tuple[str, ...]
    metrics: dict[str, dict[str, float]]


def run_experiment(config: ExperimentConfig) -> RunReport:
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    deltas = orchestrator.process_stream(dataset.documents())
    snapshot = orchestrator.snapshot()
    ground_truth = dataset.ground_truth()
    metric_results = {
        spec.name: instantiate(Metric, spec).evaluate(snapshot, ground_truth)
        for spec in config.metrics
    }
    return RunReport(
        config=config.model_dump(),
        documents_processed=len(deltas),
        errors=tuple(error for delta in deltas for error in delta.errors),
        metrics=metric_results,
    )


def run_from_path(path: str | Path) -> RunReport:
    return run_experiment(load_config(path, model=ExperimentConfig))
```

`src/lattice/harness/__main__.py`:

```python
import dataclasses
import json
import sys

from lattice.harness.runner import run_from_path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m lattice.harness <config.toml>")
    report = run_from_path(sys.argv[1])
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/harness/ -v`
Expected: `4 passed`

- [ ] **Step 6: Run the full suite and the demo**

Run: `uv run pytest`
Expected: all tests pass, zero failures.

Run: `uv run python -m lattice.harness configs/walking-skeleton.toml`
Expected: JSON report on stdout with `"documents_processed": 3`, empty `errors`, and a `label-f1` block showing `"recall": 1.0`.

- [ ] **Step 7: Commit**

```bash
git add src/lattice/harness/ configs/ tests/harness/
git commit -m "feat: add experiment harness with stamped reports and first e2e run"
```

---

## Milestone 1 exit criteria

- `uv run pytest` — full suite green.
- `uv run python -m lattice.harness configs/walking-skeleton.toml` — end-to-end run over the toy corpus produces a stamped, reproducible report with recall 1.0 and zero errors.
- Every port has a contract test suite; every adapter passes its port's contract (spec §11).
- Swapping any adapter = editing one TOML line (spec §7); adding one = one new decorated class.

## Deferred to later milestones (per spec §13)

- M2: noun-chunk extractor (spaCy), sentence-transformer embedder, MDERank/HCUKE scorers, Inspec/SemEval datasets, F1@k metrics, sweep expander.
- M3: embedding-NN resolver, ECB+/ConEL-2 datasets, clustering metrics.
- M4: Hearst-pattern and head-modifier relation inducers, TExEval-2 dataset.
- M5: intrinsic integration harness. M6: engine API hardening.
