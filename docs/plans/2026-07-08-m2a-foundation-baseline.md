# lattice M2a — Foundation + Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M2a of the extraction + salience track (spec: `docs/2026-07-08-m2-extraction-salience-design.md`): heavy-dep packaging, `GraphDelta.selected_mentions`, the `DocumentMetric` port, F1@k metric, sweep runner, Inspec dataset, noun-chunk extractor, sentence-transformer embedder, embedding-cosine baseline — ending in a reproducible baseline sweep.

**Architecture:** Pure additions behind M1's ports plus two sanctioned deltas: one new `GraphDelta` field and one new harness port (`DocumentMetric`). All scorers consume the injected `Embedder`. ML libraries are imported only inside adapter `__init__` bodies and the fetch scripts, so `import lattice.adapters` (and the default test suite) works without the `ml` group.

**Tech Stack:** Python ≥3.12, uv, pydantic v2, pytest, snowballstemmer (new core dep), ruff (dev). Optional `ml` group: sentence-transformers, spacy, datasets.

## Global Constraints

- `requires-python = ">=3.12"`; uv-managed; run tests as `chflags nohidden .venv/lib/python*/site-packages/*.pth && uv run --no-sync pytest` (known macOS UF_HIDDEN quirk — recurs after every `uv sync`).
- Core runtime deps after this plan: `pydantic>=2.7`, `snowballstemmer>=2.2` **only**. Dev: `pytest>=8.0`, `ruff>=0.4`. Optional `ml` group only: `sentence-transformers>=3.0`, `spacy>=3.7`, `datasets>=2.19`.
- `src/lattice/core/` stays stdlib-only (parent spec §5).
- **The default test suite must pass with the `ml` group NOT installed.** ML-dependent test modules guard with `pytest.importorskip` + `pytestmark = pytest.mark.ml` and module-level skip when models are missing. No adapter module imports an ML library at module scope — only inside `__init__`.
- No model or dataset downloads inside tests or adapters; downloads happen only in `scripts/fetch_*.py`.
- Determinism: ties break lexicographically; identical configs reproduce byte-identical reports (parent spec §7).
- Registry names (kebab-case): `"noun-chunk"`, `"sentence-transformer"`, `"embedding-cosine"`, `"inspec"`, `"f1-at-k"`.
- TDD per task; conventional commits.
- Every new adapter passes its port's existing contract suite; `DocumentMetric` gets a new contract suite.

---

### Task 1: Packaging + hygiene (deps, ruff, py.typed, markers)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/lattice/py.typed`
- Test: existing suite + `uv run ruff check .`

**Interfaces:**
- Consumes: M1 pyproject (Task 1 of the M1 plan).
- Produces: `snowballstemmer` importable; `ml` dependency group; `ml` pytest marker registered; ruff configured; `data/` and `reports/` git-ignored. NOTE: this task is the one sanctioned edit of pyproject.toml in M2a.

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `[project]` dependencies line and `[dependency-groups]`, and append tool config, so the full file reads:

```toml
[project]
name = "lattice"
version = "0.1.0"
description = "Concept-memory engine: documents in, an accreting normalized concept graph out"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.7", "snowballstemmer>=2.2"]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.4"]
ml = ["sentence-transformers>=3.0", "spacy>=3.7", "datasets>=2.19"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lattice"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["ml: requires the ml dependency group and downloaded models"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 2: Append to `.gitignore`**

```gitignore
data/
reports/
```

- [ ] **Step 3: Create `src/lattice/py.typed`** (empty file).

- [ ] **Step 4: Sync and verify**

Run: `uv sync && chflags nohidden .venv/lib/python*/site-packages/*.pth && uv run --no-sync pytest -q`
Expected: `120 passed` (suite unchanged).

Run: `uv run --no-sync ruff check .`
Expected: clean, or a handful of mechanical violations (import order, unused import) in M1 files — fix them in place (no behavior changes), re-run until clean, and note each fixed file in the commit body.

Run: `uv run --no-sync python -c "import snowballstemmer; print(snowballstemmer.stemmer('english').stemWords(['networks']))"`
Expected: `['network']`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/lattice/py.typed uv.lock $(git diff --name-only)
git commit -m "chore: add ml dep group, snowballstemmer, ruff config, py.typed, ml marker"
```

---

### Task 2: `GraphDelta.selected_mentions`

**Files:**
- Modify: `src/lattice/core/types.py` (GraphDelta)
- Modify: `src/lattice/orchestrator/orchestrator.py` (populate field)
- Test: `tests/core/test_types.py`, `tests/orchestrator/test_orchestrator.py` (add tests)

**Interfaces:**
- Consumes: `ScoredMention`, `GraphDelta` (M1 Task 2); orchestrator `process()` (M1 Task 14).
- Produces: `GraphDelta.selected_mentions: tuple[ScoredMention, ...] = ()` — the scorer's selected output in scorer order; populated on the success path, empty on the skip path. Task 4's metric and Task 11's e2e rely on it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_types.py`:

```python
def test_graph_delta_selected_mentions_default_empty():
    delta = GraphDelta(document_id="d1", concepts_added=(), concepts_updated=(), relations_added=())
    assert delta.selected_mentions == ()
```

Append to `tests/orchestrator/test_orchestrator.py`:

```python
def test_delta_carries_selected_mentions_in_scorer_order():
    orchestrator = build_orchestrator(scorer=FrequencyScorer(top_k=1))
    delta = orchestrator.process(make_document(id="d1", text="vector vector store"))
    surfaces = [sm.mention.surface for sm in delta.selected_mentions]
    assert surfaces == ["vector", "vector"]
    assert all(sm.selected for sm in delta.selected_mentions)


def test_skip_path_has_no_selected_mentions():
    orchestrator = build_orchestrator(extractor=ExplodingExtractor(), on_error="skip")
    delta = orchestrator.process(make_document(id="d1"))
    assert delta.selected_mentions == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/core/test_types.py tests/orchestrator/test_orchestrator.py -v`
Expected: the three new tests FAIL (`unexpected keyword`/`no attribute 'selected_mentions'`); all others pass.

- [ ] **Step 3: Implement**

In `src/lattice/core/types.py`, add to `GraphDelta` after `errors`:

```python
    selected_mentions: tuple[ScoredMention, ...] = ()
```

and extend the docstring with: `selected_mentions` is the scorer's selected output for this document (pre-resolver), the unit of per-document evaluation (M2 spec §4).

In `src/lattice/orchestrator/orchestrator.py`, success-path return gains:

```python
            selected_mentions=tuple(selected),
```

(the skip path keeps the default).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/core/ tests/orchestrator/ -v`
Expected: all pass (13 + 10).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `123 passed`

```bash
git add src/lattice/core/types.py src/lattice/orchestrator/orchestrator.py tests/core/test_types.py tests/orchestrator/test_orchestrator.py
git commit -m "feat: carry scorer-selected mentions on GraphDelta for per-document evaluation"
```

---

### Task 3: `DocumentMetric` port + harness wiring

**Files:**
- Create: `src/lattice/ports/document_metric.py`
- Modify: `src/lattice/ports/__init__.py`, `tests/ports/test_ports_are_abstract.py`
- Modify: `src/lattice/harness/runner.py`
- Create: `tests/contracts/document_metric_contract.py`
- Test: `tests/harness/test_runner.py` (add tests)

**Interfaces:**
- Consumes: `GraphDelta` (Task 2); `ExperimentConfig`, `run_experiment`, `instantiate` (M1 Tasks 15/17).
- Produces:
  - `DocumentMetric.evaluate_documents(deltas: Sequence[GraphDelta], ground_truth: dict[str, object]) -> dict[str, float]` (ABC, exported from `lattice.ports`).
  - `ExperimentConfig.document_metrics: list[AdapterSpec] = []`.
  - `run_experiment` evaluates both metric families into one `RunReport.metrics` dict keyed by adapter name; duplicate names across families raise `ValueError`.
  - `DocumentMetricContract` — subclass implements `make_metric()`, `make_ground_truth()`, `make_deltas()` (consistent with the ground truth).

- [ ] **Step 1: Write the failing tests**

`tests/contracts/document_metric_contract.py`:

```python
"""Contract every DocumentMetric adapter must satisfy."""

import pytest

from lattice.core.types import GraphDelta
from lattice.ports import DocumentMetric


class DocumentMetricContract:
    def make_metric(self) -> DocumentMetric:
        raise NotImplementedError("subclass must provide the adapter under test")

    def make_ground_truth(self) -> dict:
        raise NotImplementedError("subclass must provide matching ground truth")

    def make_deltas(self) -> list[GraphDelta]:
        raise NotImplementedError("subclass must provide deltas consistent with ground truth")

    def test_returns_dict_of_floats(self):
        result = self.make_metric().evaluate_documents(self.make_deltas(), self.make_ground_truth())
        assert result and all(isinstance(v, float) for v in result.values())

    def test_unknown_document_raises(self):
        deltas = [
            GraphDelta(document_id="not-in-ground-truth", concepts_added=(),
                       concepts_updated=(), relations_added=())
        ]
        with pytest.raises(ValueError):
            self.make_metric().evaluate_documents(deltas, self.make_ground_truth())
```

Append to `tests/ports/test_ports_are_abstract.py`: import `DocumentMetric` from `lattice.ports` and add it to `ALL_PORTS`.

Append to `tests/harness/test_runner.py`:

```python
from lattice.core.types import GraphDelta
from lattice.ports import DocumentMetric
from lattice.registry.registry import register


@register(DocumentMetric, "count-docs")
class _CountDocs(DocumentMetric):
    def evaluate_documents(self, deltas, ground_truth):
        return {"documents": float(len(list(deltas)))}


def test_document_metrics_evaluated_from_deltas():
    config = _experiment_config()
    config = config.model_copy(update={"document_metrics": [AdapterSpec(name="count-docs")]})
    report = run_experiment(config)
    assert report.metrics["count-docs"] == {"documents": 3.0}


def test_duplicate_metric_names_across_families_rejected():
    config = _experiment_config().model_copy(
        update={"document_metrics": [AdapterSpec(name="label-f1")]}
    )
    with pytest.raises(ValueError, match="label-f1"):
        run_experiment(config)
```

(add `import pytest` and `from lattice.config.schema import AdapterSpec` to that file's imports; the duplicate test needs a trivial `DocumentMetric` registered as `"label-f1"` in the test module:)

```python
@register(DocumentMetric, "label-f1")
class _ShadowLabelF1(DocumentMetric):
    def evaluate_documents(self, deltas, ground_truth):
        return {"x": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/ports/ tests/harness/ -v`
Expected: FAIL — `ImportError: cannot import name 'DocumentMetric'`.

- [ ] **Step 3: Implement**

`src/lattice/ports/document_metric.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import GraphDelta


class DocumentMetric(ABC):
    """Harness port: scores per-document pipeline output (the run's GraphDeltas)
    against per-document ground truth (M2 spec §5). Complements the
    snapshot-level Metric port."""

    @abstractmethod
    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]: ...
```

`src/lattice/ports/__init__.py`: add `from lattice.ports.document_metric import DocumentMetric` and `"DocumentMetric"` to `__all__` (alphabetical position).

In `src/lattice/harness/runner.py`:
- `ExperimentConfig` gains: `document_metrics: list[AdapterSpec] = Field(default_factory=list)`
- import `DocumentMetric` from `lattice.ports`
- in `run_experiment`, after `metric_results`:

```python
    document_results = {
        spec.name: instantiate(DocumentMetric, spec).evaluate_documents(deltas, ground_truth)
        for spec in config.document_metrics
    }
    duplicates = set(metric_results) & set(document_results)
    if duplicates:
        raise ValueError(f"metric name(s) used by both families: {sorted(duplicates)}")
    all_metrics = {**metric_results, **document_results}
```

and stamp `metrics=all_metrics` in the returned `RunReport`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/ports/ tests/harness/ -v`
Expected: all pass (12 port tests; 6 harness tests).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `126 passed`

```bash
git add src/lattice/ports/ src/lattice/harness/runner.py tests/ports/ tests/harness/ tests/contracts/document_metric_contract.py
git commit -m "feat: add DocumentMetric port and per-document evaluation in the harness"
```

---

### Task 4: F1@k document metric

**Files:**
- Create: `src/lattice/adapters/document_metric/__init__.py`, `src/lattice/adapters/document_metric/f1_at_k.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Modify: `tests/helpers.py` (add `make_delta`)
- Test: `tests/adapters/test_f1_at_k.py`

**Interfaces:**
- Consumes: `DocumentMetric` port + contract (Task 3); `GraphDelta.selected_mentions` (Task 2); `make_scored_mention` (M1).
- Produces:
  - `F1AtK(ks: list[int] | None = None)` registered `(DocumentMetric, "f1-at-k")`; output keys `"precision@K"`, `"recall@K"`, `"f1@K"` for each K (defaults 5/10/15); ranks by salience desc, ties lexicographic; Snowball-stemmed exact phrase match; unknown document id or empty deltas → `ValueError`.
  - `make_delta(document_id: str = "d1", selected: list[tuple[str, float]] = []) -> GraphDelta` in `tests/helpers.py` (builds `selected_mentions` from (surface, salience) pairs).

- [ ] **Step 1: Add the helper and write the failing tests**

Append to `tests/helpers.py`:

```python
def make_delta(
    document_id: str = "d1",
    selected: list[tuple[str, float]] | None = None,
) -> "GraphDelta":
    return GraphDelta(
        document_id=document_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=(),
        selected_mentions=tuple(
            make_scored_mention(surface=s, salience=sal) for s, sal in (selected or [])
        ),
    )
```

(add `GraphDelta` to the `lattice.core.types` import in that file.)

`tests/adapters/test_f1_at_k.py`:

```python
import pytest

from lattice.adapters.document_metric.f1_at_k import F1AtK
from tests.contracts.document_metric_contract import DocumentMetricContract
from tests.helpers import make_delta


class TestF1AtK(DocumentMetricContract):
    def make_metric(self) -> F1AtK:
        return F1AtK(ks=[5])

    def make_ground_truth(self) -> dict:
        return {"keyphrases_by_document": {"d1": ["vector store", "encoder"]}}

    def make_deltas(self):
        return [make_delta("d1", [("vector store", 0.9), ("encoder", 0.8)])]

    def test_perfect_at_k(self):
        result = self.make_metric().evaluate_documents(self.make_deltas(), self.make_ground_truth())
        assert result == {"precision@5": 1.0, "recall@5": 1.0, "f1@5": 1.0}

    def test_stemming_matches_inflections(self):
        gt = {"keyphrases_by_document": {"d1": ["neural networks"]}}
        deltas = [make_delta("d1", [("neural network", 0.9)])]
        assert F1AtK(ks=[5]).evaluate_documents(deltas, gt)["f1@5"] == 1.0

    def test_k_truncates_by_salience_rank(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"]}}
        deltas = [make_delta("d1", [("beta", 0.9), ("alpha", 0.5)])]
        result = F1AtK(ks=[1]).evaluate_documents(deltas, gt)
        assert result["recall@1"] == 0.0  # only top-1 (beta) is kept

    def test_salience_ties_break_lexicographically(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"]}}
        deltas = [make_delta("d1", [("beta", 0.9), ("alpha", 0.9)])]
        assert F1AtK(ks=[1]).evaluate_documents(deltas, gt)["recall@1"] == 1.0

    def test_macro_average_over_documents(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"], "d2": ["beta"]}}
        deltas = [
            make_delta("d1", [("alpha", 1.0)]),   # f1 = 1.0
            make_delta("d2", [("gamma", 1.0)]),   # f1 = 0.0
        ]
        assert F1AtK(ks=[5]).evaluate_documents(deltas, gt)["f1@5"] == 0.5

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="no documents"):
            F1AtK().evaluate_documents([], self.make_ground_truth())

    def test_missing_ground_truth_key_raises(self):
        with pytest.raises(ValueError, match="keyphrases_by_document"):
            F1AtK().evaluate_documents(self.make_deltas(), {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/adapters/test_f1_at_k.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.document_metric'`.

- [ ] **Step 3: Implement**

`src/lattice/adapters/document_metric/__init__.py`: empty file.

`src/lattice/adapters/document_metric/f1_at_k.py`:

```python
from collections.abc import Sequence

import snowballstemmer

from lattice.core.types import GraphDelta
from lattice.ports import DocumentMetric
from lattice.registry.registry import register


@register(DocumentMetric, "f1-at-k")
class F1AtK(DocumentMetric):
    """Literature-standard keyphrase evaluation (M2 spec §6.5): per document,
    the top-k selected surfaces (salience desc, ties lexicographic) are
    compared to gold keyphrases as Snowball-stemmed exact phrase matches;
    precision/recall/F1 are macro-averaged over documents."""

    def __init__(self, ks: list[int] | None = None):
        self.ks = list(ks) if ks is not None else [5, 10, 15]
        self._stemmer = snowballstemmer.stemmer("english")

    def _stem_phrase(self, phrase: str) -> str:
        return " ".join(self._stemmer.stemWords(phrase.lower().split()))

    def _ranked_unique_surfaces(self, delta: GraphDelta) -> list[str]:
        best: dict[str, float] = {}
        for scored in delta.selected_mentions:
            surface = scored.mention.surface
            if surface not in best or scored.salience > best[surface]:
                best[surface] = scored.salience
        return [s for s, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]

    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]:
        by_document = ground_truth.get("keyphrases_by_document")
        if not isinstance(by_document, dict):
            raise ValueError('f1-at-k requires ground_truth["keyphrases_by_document"]')
        deltas = list(deltas)
        if not deltas:
            raise ValueError("no documents to evaluate")
        results: dict[str, float] = {}
        for k in self.ks:
            precisions: list[float] = []
            recalls: list[float] = []
            f1s: list[float] = []
            for delta in deltas:
                if delta.document_id not in by_document:
                    raise ValueError(
                        f"document {delta.document_id!r} missing from ground truth"
                    )
                gold = {self._stem_phrase(p) for p in by_document[delta.document_id]}
                predicted = {
                    self._stem_phrase(s) for s in self._ranked_unique_surfaces(delta)[:k]
                }
                true_positives = len(gold & predicted)
                precision = true_positives / len(predicted) if predicted else 0.0
                recall = true_positives / len(gold) if gold else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            count = len(deltas)
            results[f"precision@{k}"] = sum(precisions) / count
            results[f"recall@{k}"] = sum(recalls) / count
            results[f"f1@{k}"] = sum(f1s) / count
        return results
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.document_metric import f1_at_k  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_f1_at_k.py -v`
Expected: `9 passed` (2 contract + 7 specific).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `135 passed`

```bash
git add src/lattice/adapters/ tests/adapters/test_f1_at_k.py tests/helpers.py
git commit -m "feat: add stemmed F1@k document metric adapter"
```

---

### Task 5: Sweep runner

**Files:**
- Create: `src/lattice/harness/sweep.py`
- Modify: `src/lattice/harness/__main__.py` (add `--sweep` mode)
- Test: `tests/harness/test_sweep.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `RunReport`, `run_experiment` (M1 Task 17 + Task 3); `AdapterSpec`, `load_config` (M1 Task 5).
- Produces (M2 spec §7):
  - `SweepConfig(BaseModel, extra="forbid")` — `base: ExperimentConfig`, `axes: dict[str, list[AdapterSpec]] = {}`.
  - `expand(sweep: SweepConfig) -> list[ExperimentConfig]` — cartesian product, axes iterated in sorted-name order, deterministic output order; unknown axis name → `ValueError`.
  - `run_sweep(sweep: SweepConfig) -> SweepReport` — frozen dataclass: `sweep: dict`, `runs: list[RunReport]`, `table: list[dict[str, object]]` (one row per run: axis columns `axis:<port>` = adapter name, then flattened metric columns `"<metric>.<key>"`).
  - `write_reports(report: SweepReport, out_dir: str | Path) -> tuple[Path, Path]` — writes `sweep-report.json` and `sweep-report.md`.
  - CLI: `python -m lattice.harness <config.toml>` still runs single experiments; `python -m lattice.harness --sweep <sweep.toml> [out_dir]` runs a sweep, prints the markdown table, writes reports (default out_dir `reports/`).

- [ ] **Step 1: Write the failing tests**

`tests/harness/test_sweep.py`:

```python
import json

import pytest

from lattice.config.schema import AdapterSpec
from lattice.harness.sweep import SweepConfig, expand, run_sweep, write_reports

BASE = {
    "segmenter": {"name": "block"},
    "extractor": {"name": "token", "params": {"min_length": 4}},
    "scorer": {"name": "frequency"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "dataset": {"name": "toy"},
    "metrics": [{"name": "label-f1"}],
}


def make_sweep(axes) -> SweepConfig:
    return SweepConfig.model_validate({"base": BASE, "axes": axes})


def test_expand_cartesian_product_in_sorted_axis_order():
    sweep = make_sweep(
        {
            "scorer": [{"name": "frequency", "params": {"top_k": 5}},
                        {"name": "frequency", "params": {"top_k": 10}}],
            "extractor": [{"name": "token", "params": {"min_length": 3}}],
        }
    )
    configs = expand(sweep)
    assert len(configs) == 2
    assert all(c.extractor.params == {"min_length": 3} for c in configs)
    assert [c.scorer.params["top_k"] for c in configs] == [5, 10]


def test_expand_no_axes_yields_base_only():
    configs = expand(make_sweep({}))
    assert len(configs) == 1
    assert configs[0].scorer.name == "frequency"


def test_expand_unknown_axis_rejected():
    with pytest.raises(ValueError, match="not-a-port"):
        expand(make_sweep({"not-a-port": [{"name": "x"}]}))


def test_run_sweep_produces_one_row_per_config():
    sweep = make_sweep(
        {"scorer": [{"name": "frequency", "params": {"top_k": 5}},
                     {"name": "frequency", "params": {"top_k": 10}}]}
    )
    report = run_sweep(sweep)
    assert len(report.runs) == 2
    assert len(report.table) == 2
    assert report.table[0]["axis:scorer"] == "frequency"
    assert "label-f1.f1" in report.table[0]


def test_sweep_is_reproducible():
    sweep = make_sweep({"scorer": [{"name": "frequency"}]})
    assert run_sweep(sweep) == run_sweep(sweep)


def test_write_reports(tmp_path):
    report = run_sweep(make_sweep({}))
    json_path, md_path = write_reports(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert len(data["runs"]) == 1
    assert md_path.read_text().startswith("|")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/harness/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.harness.sweep'`.

- [ ] **Step 3: Implement**

`src/lattice/harness/sweep.py`:

```python
"""Declarative sweeps (M2 spec §7): a base experiment config plus axes of
adapter alternatives, expanded as a cartesian product. Each config runs with
a fresh factory-built orchestrator — no state is reused across runs."""

import dataclasses
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lattice.config.schema import AdapterSpec
from lattice.harness.runner import ExperimentConfig, RunReport, run_experiment


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: ExperimentConfig
    axes: dict[str, list[AdapterSpec]] = Field(default_factory=dict)


def expand(sweep: SweepConfig) -> list[ExperimentConfig]:
    for axis in sweep.axes:
        if axis not in ExperimentConfig.model_fields:
            raise ValueError(f"unknown sweep axis {axis!r} (not an ExperimentConfig field)")
    axis_names = sorted(sweep.axes)
    configs: list[ExperimentConfig] = []
    for combo in itertools.product(*(sweep.axes[name] for name in axis_names)):
        data = sweep.base.model_dump()
        for name, spec in zip(axis_names, combo):
            data[name] = spec.model_dump()
        configs.append(ExperimentConfig.model_validate(data))
    return configs


@dataclass(frozen=True)
class SweepReport:
    sweep: dict[str, Any]
    runs: list[RunReport]
    table: list[dict[str, object]]


def _row(config: ExperimentConfig, report: RunReport, axis_names: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        f"axis:{name}": getattr(config, name).name for name in axis_names
    }
    for metric_name, values in sorted(report.metrics.items()):
        for key, value in sorted(values.items()):
            row[f"{metric_name}.{key}"] = value
    row["errors"] = len(report.errors)
    return row


def run_sweep(sweep: SweepConfig) -> SweepReport:
    axis_names = sorted(sweep.axes)
    configs = expand(sweep)
    runs = [run_experiment(config) for config in configs]
    table = [_row(config, report, axis_names) for config, report in zip(configs, runs)]
    return SweepReport(sweep=sweep.model_dump(), runs=runs, table=table)


def _markdown_table(table: list[dict[str, object]]) -> str:
    if not table:
        return "(empty sweep)\n"
    columns = list(table[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in (row[c] for c in columns)
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: SweepReport, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "sweep-report.json"
    md_path = out / "sweep-report.md"
    json_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))
    md_path.write_text(_markdown_table(report.table))
    return json_path, md_path
```

Replace `src/lattice/harness/__main__.py` with:

```python
import dataclasses
import json
import sys

from lattice.config.loader import load_config
from lattice.harness.runner import run_from_path
from lattice.harness.sweep import SweepConfig, run_sweep, write_reports


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--sweep":
        if len(args) not in (2, 3):
            raise SystemExit("usage: python -m lattice.harness --sweep <sweep.toml> [out_dir]")
        sweep = load_config(args[1], model=SweepConfig)
        report = run_sweep(sweep)
        json_path, md_path = write_reports(report, args[2] if len(args) == 3 else "reports")
        print(md_path.read_text())
        print(f"reports: {json_path} {md_path}")
        return
    if len(args) != 1:
        raise SystemExit("usage: python -m lattice.harness <config.toml> | --sweep <sweep.toml> [out_dir]")
    print(json.dumps(dataclasses.asdict(run_from_path(args[0])), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/harness/ -v`
Expected: all pass (6 runner + 6 sweep).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `141 passed`

```bash
git add src/lattice/harness/ tests/harness/test_sweep.py
git commit -m "feat: add declarative sweep runner with stamped JSON and markdown reports"
```

---

### Task 6: Fetch scripts (datasets + models)

**Files:**
- Create: `scripts/fetch_datasets.py`, `scripts/fetch_models.py`
- Test: `tests/scripts/__init__.py`, `tests/scripts/test_fetch_datasets.py`

**Interfaces:**
- Consumes: nothing in-package (scripts are standalone; `datasets` imported lazily).
- Produces:
  - `record_to_line(record: dict) -> str` (pure, unit-tested) — converts a midas/inspec record to lattice JSONL: `{"id", "text", "keyphrases"}`; text = tokens joined by single spaces; keyphrases = extractive + abstractive.
  - `python scripts/fetch_datasets.py inspec [--root data]` → writes `data/inspec/{train,validation,test}.jsonl` + `CHECKSUMS` (sha256 per file).
  - `python scripts/fetch_models.py` → downloads `en_core_web_sm` + warms `all-MiniLM-L6-v2` cache. Both require the `ml` group and are the ONLY places downloads happen.

- [ ] **Step 1: Write the failing test**

`tests/scripts/__init__.py`: empty file.

`tests/scripts/test_fetch_datasets.py`:

```python
import json

from scripts.fetch_datasets import record_to_line


def test_record_to_line_joins_tokens_and_merges_keyphrases():
    record = {
        "id": "1789",
        "document": ["Neural", "networks", "learn", "representations", "."],
        "extractive_keyphrases": ["neural networks"],
        "abstractive_keyphrases": ["representation learning"],
    }
    parsed = json.loads(record_to_line(record))
    assert parsed == {
        "id": "1789",
        "text": "Neural networks learn representations .",
        "keyphrases": ["neural networks", "representation learning"],
    }


def test_record_to_line_missing_id_uses_fallback():
    record = {"document": ["x"], "extractive_keyphrases": [], "abstractive_keyphrases": []}
    parsed = json.loads(record_to_line(record, fallback_id="42"))
    assert parsed["id"] == "42"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/scripts/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_datasets'` (add `scripts/__init__.py`, empty, so the module imports under `pythonpath=["."]`).

- [ ] **Step 3: Implement**

`scripts/__init__.py`: empty file.

`scripts/fetch_datasets.py`:

```python
"""Fetch benchmark datasets and convert once to lattice's plain JSONL format
(M2 spec §6.4). Requires the ml dependency group:
    uv run --group ml python scripts/fetch_datasets.py inspec
The emitted JSONL is the only format adapters read — stdlib only at runtime."""

import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def record_to_line(record: dict, fallback_id: str = "") -> str:
    return json.dumps(
        {
            "id": str(record.get("id") or fallback_id),
            "text": " ".join(record["document"]),
            "keyphrases": list(record["extractive_keyphrases"])
            + list(record["abstractive_keyphrases"]),
        },
        sort_keys=True,
    )


def fetch_inspec(root: Path) -> None:
    from datasets import load_dataset  # ml group; imported lazily on purpose

    out_dir = root / "inspec"
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    for split in SPLITS:
        dataset = load_dataset("midas/inspec", "extraction", split=split)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for i, record in enumerate(dataset):
                f.write(record_to_line(record, fallback_id=str(i)) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        print(f"wrote {out_path} ({digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["inspec"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    fetch_inspec(Path(args.root))


if __name__ == "__main__":
    main()
```

`scripts/fetch_models.py`:

```python
"""Download the ML models M2 needs (the only sanctioned model-download path).
    uv run --group ml python scripts/fetch_models.py"""

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"sentence-transformer ready: dim={model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/scripts/ -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/scripts/
git commit -m "feat: add dataset and model fetch scripts (JSONL conversion, checksums)"
```

---

### Task 7: Inspec dataset adapter + mini fixture

**Files:**
- Create: `src/lattice/adapters/dataset/inspec.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Create: `tests/fixtures/mini_inspec/test.jsonl`
- Test: `tests/adapters/test_inspec_dataset.py`

**Interfaces:**
- Consumes: `Dataset` port + `DatasetContract` (M1); the JSONL format from Task 6.
- Produces: `InspecDataset(root: str = "data/inspec", split: str = "test", limit: int | None = None)` registered `(Dataset, "inspec")`; documents `kind="abstract"`, `timestamp=float(index)`; `ground_truth()` = `{"keyphrases_by_document": {id: [...]}}`; missing file → `FileNotFoundError` naming the fetch script. The mini fixture is the ml-free e2e corpus for Task 11.

- [ ] **Step 1: Write the fixture and failing tests**

`tests/fixtures/mini_inspec/test.jsonl` (exactly 3 lines):

```jsonl
{"id": "mini-1", "keyphrases": ["vector databases", "approximate nearest neighbor search"], "text": "Vector databases index high dimensional embeddings and answer similarity queries . Modern systems implement approximate nearest neighbor search to trade accuracy for speed ."}
{"id": "mini-2", "keyphrases": ["sentence embeddings", "semantic similarity"], "text": "Sentence embeddings map text to dense vectors . They enable semantic similarity comparison between documents and queries ."}
{"id": "mini-3", "keyphrases": ["keyphrase extraction", "unsupervised methods"], "text": "Keyphrase extraction identifies the most important phrases in a document . Unsupervised methods rank candidate phrases without labeled training data ."}
```

`tests/adapters/test_inspec_dataset.py`:

```python
import pytest

from lattice.adapters.dataset.inspec import InspecDataset
from tests.contracts.dataset_contract import DatasetContract

FIXTURE_ROOT = "tests/fixtures/mini_inspec"


class TestInspecDataset(DatasetContract):
    def make_dataset(self) -> InspecDataset:
        return InspecDataset(root=FIXTURE_ROOT, split="test")

    def test_documents_have_abstract_kind_and_text(self):
        docs = list(self.make_dataset().documents())
        assert len(docs) == 3
        assert docs[0].kind == "abstract"
        assert "Vector databases" in docs[0].text

    def test_ground_truth_keyed_by_document(self):
        truth = self.make_dataset().ground_truth()
        assert truth["keyphrases_by_document"]["mini-2"] == [
            "sentence embeddings", "semantic similarity",
        ]

    def test_limit_truncates(self):
        assert len(list(InspecDataset(root=FIXTURE_ROOT, split="test", limit=2).documents())) == 2

    def test_missing_file_names_the_fetch_script(self):
        with pytest.raises(FileNotFoundError, match="fetch_datasets"):
            list(InspecDataset(root="does/not/exist").documents())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/adapters/test_inspec_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.dataset.inspec'`.

- [ ] **Step 3: Implement**

`src/lattice/adapters/dataset/inspec.py`:

```python
import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "inspec")
class InspecDataset(Dataset):
    """Inspec keyphrase benchmark (Hulth 2003) read from the plain JSONL
    emitted by scripts/fetch_datasets.py. Gold = the uncontrolled keyword
    set as merged by the midas/inspec distribution (M2 spec §6.4).
    Stdlib-only at runtime."""

    def __init__(self, root: str = "data/inspec", split: str = "test", limit: int | None = None):
        self.path = Path(root) / f"{split}.jsonl"
        self.limit = limit

    def _records(self) -> Iterator[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run `uv run --group ml python "
                f"scripts/fetch_datasets.py inspec` first"
            )
        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    return
                yield json.loads(line)

    def documents(self) -> Iterator[Document]:
        for i, record in enumerate(self._records()):
            yield Document(
                id=record["id"], kind="abstract", text=record["text"], timestamp=float(i)
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "keyphrases_by_document": {
                record["id"]: list(record["keyphrases"]) for record in self._records()
            }
        }
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.dataset import inspec  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_inspec_dataset.py -v`
Expected: `8 passed` (4 contract + 4 specific).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `151 passed`

```bash
git add src/lattice/adapters/ tests/adapters/test_inspec_dataset.py tests/fixtures/
git commit -m "feat: add Inspec dataset adapter with committed mini fixture"
```

---

### Task 8: Embedding-cosine scorer (+ shared cosine util)

**Files:**
- Create: `src/lattice/core/vectors.py`
- Modify: `src/lattice/adapters/concept_store/in_memory.py` (use shared cosine)
- Create: `src/lattice/adapters/scorer/embedding_cosine.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/core/test_vectors.py`, `tests/adapters/test_embedding_cosine_scorer.py`

**Interfaces:**
- Consumes: `Embedder` port (injected — tests use `HashingEmbedder`, so this task is ml-free); `ScorerContract` (M1).
- Produces:
  - `lattice.core.vectors.cosine(a: Sequence[float], b: Sequence[float]) -> float` (zero-norm-safe, stdlib-only).
  - `EmbeddingCosineScorer(embedder: Embedder, top_k: int = 10)` registered `(Scorer, "embedding-cosine")`: salience = cosine(candidate surface embedding, document embedding); document text = unit texts joined by `"\n"`; each unique surface embedded once; top-k unique surfaces selected, ties lexicographic. Constructor param name `embedder` is load-bearing (factory injection).

- [ ] **Step 1: Write the failing tests**

`tests/core/test_vectors.py`:

```python
import math

from lattice.core.vectors import cosine


def test_cosine_identical_vectors():
    assert math.isclose(cosine((1.0, 2.0), (1.0, 2.0)), 1.0)


def test_cosine_orthogonal_vectors():
    assert cosine((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_cosine_zero_vector_is_zero():
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0
```

`tests/adapters/test_embedding_cosine_scorer.py`:

```python
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.embedding_cosine import EmbeddingCosineScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class CountingEmbedder(Embedder):
    """Test double: hashing embedder that counts embed() texts."""

    def __init__(self):
        self.inner = HashingEmbedder(dim=16)
        self.texts_embedded: list[str] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts):
        self.texts_embedded.extend(texts)
        return self.inner.embed(texts)


class TestEmbeddingCosineScorer(ScorerContract):
    def make_scorer(self) -> EmbeddingCosineScorer:
        return EmbeddingCosineScorer(embedder=HashingEmbedder(dim=16))

    def test_each_unique_surface_embedded_once(self):
        embedder = CountingEmbedder()
        scorer = EmbeddingCosineScorer(embedder=embedder)
        mentions = [
            make_mention(surface="vector store", span=(0, 12)),
            make_mention(surface="vector store", span=(20, 32)),
            make_mention(surface="encoder", span=(40, 47)),
        ]
        scorer.score(mentions, [make_unit(text="vector store text vector store encoder")])
        assert len(embedder.texts_embedded) == 3  # 1 document + 2 unique surfaces

    def test_document_similar_surface_scores_highest(self):
        scorer = self.make_scorer()
        unit = make_unit(text="vector store")
        mentions = [
            make_mention(surface="vector store", unit_id=unit.id, span=(0, 12)),
            make_mention(surface="zzz unrelated", unit_id=unit.id, span=(0, 3)),
        ]
        scored = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert scored["vector store"] > scored["zzz unrelated"]

    def test_top_k_selects_unique_surfaces_with_lexicographic_ties(self):
        scorer = EmbeddingCosineScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        unit = make_unit(text="alpha beta")
        mentions = [
            make_mention(surface="alpha", unit_id=unit.id, span=(0, 5)),
            make_mention(surface="beta", unit_id=unit.id, span=(6, 10)),
        ]
        scored = scorer.score(mentions, [unit])
        assert sum(1 for sm in scored if sm.selected) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/core/test_vectors.py tests/adapters/test_embedding_cosine_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `lattice.core.vectors`.

- [ ] **Step 3: Implement**

`src/lattice/core/vectors.py`:

```python
"""Shared pure vector math. Stdlib only (parent spec §5)."""

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
```

In `src/lattice/adapters/concept_store/in_memory.py`: delete the local `_cosine` and its `math` import; add `from lattice.core.vectors import cosine` and replace the two `_cosine(` call sites with `cosine(`.

`src/lattice/adapters/scorer/embedding_cosine.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.core.vectors import cosine
from lattice.ports import Embedder, Scorer
from lattice.registry.registry import register


@register(Scorer, "embedding-cosine")
class EmbeddingCosineScorer(Scorer):
    """KeyBERT/SIFRank-style baseline (M2 spec §6.3): salience is the cosine
    similarity between the candidate surface embedding and the whole-document
    embedding. Consumes the injected Embedder; each unique surface is embedded
    once per call."""

    def __init__(self, embedder: Embedder, top_k: int = 10):
        self.embedder = embedder
        self.top_k = top_k

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        document_text = "\n".join(unit.text for unit in units)
        surfaces = sorted({m.surface for m in mentions})
        document_vector, *candidate_vectors = self.embedder.embed([document_text, *surfaces])
        salience = {
            surface: cosine(vector, document_vector)
            for surface, vector in zip(surfaces, candidate_vectors)
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.scorer import embedding_cosine  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/core/test_vectors.py tests/adapters/test_embedding_cosine_scorer.py tests/adapters/test_in_memory_concept_store.py -v`
Expected: all pass (3 + 6 + 6).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `160 passed`

```bash
git add src/lattice/core/vectors.py src/lattice/adapters/ tests/core/test_vectors.py tests/adapters/test_embedding_cosine_scorer.py
git commit -m "feat: add embedding-cosine baseline scorer and shared cosine util"
```

---

### Task 9: Noun-chunk extractor (ml)

**Files:**
- Create: `src/lattice/adapters/extractor/noun_chunk.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_noun_chunk_extractor.py` (ml-marked)

**Interfaces:**
- Consumes: `Extractor` port + `ExtractorContract` (M1); spaCy (imported ONLY inside `__init__`).
- Produces: `NounChunkExtractor(model: str = "en_core_web_sm", max_tokens: int = 5)` registered `(Extractor, "noun-chunk")`: spaCy noun chunks; leading `DET`/`PRON` tokens trimmed; all-pronoun chunks and chunks longer than `max_tokens` (after trimming) dropped; surface = lowercased slice of the unit text; `head` = chunk root text lowercased; `lemma` = trimmed-token lemmas joined, lowercased.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_noun_chunk_extractor.py`:

```python
import pytest

pytestmark = pytest.mark.ml
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed (run scripts/fetch_models.py)", allow_module_level=True)

from lattice.adapters.extractor.noun_chunk import NounChunkExtractor  # noqa: E402
from tests.contracts.extractor_contract import ExtractorContract  # noqa: E402
from tests.helpers import make_unit  # noqa: E402


class TestNounChunkExtractor(ExtractorContract):
    def make_extractor(self) -> NounChunkExtractor:
        return NounChunkExtractor()

    def test_extracts_noun_phrases_not_verbs(self):
        mentions = self.make_extractor().extract(
            [make_unit(text="The vector store indexes dense embeddings.")]
        )
        surfaces = {m.surface for m in mentions}
        assert "vector store" in surfaces
        assert "dense embeddings" in surfaces
        assert not any("indexes" == s for s in surfaces)

    def test_leading_determiner_trimmed(self):
        mentions = self.make_extractor().extract([make_unit(text="The encoder produces vectors.")])
        assert "encoder" in {m.surface for m in mentions}
        assert not any(s.startswith("the ") for s in {m.surface for m in mentions})

    def test_pronoun_only_chunks_dropped(self):
        mentions = self.make_extractor().extract([make_unit(text="It maps text to vectors.")])
        assert "it" not in {m.surface for m in mentions}

    def test_max_tokens_filters_long_chunks(self):
        text = "The very long deeply nested compound noun phrase construction persists."
        short = NounChunkExtractor(max_tokens=2).extract([make_unit(text=text)])
        assert all(len(m.surface.split()) <= 2 for m in short)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync --group ml pytest tests/adapters/test_noun_chunk_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.extractor.noun_chunk'`. (Without the ml group the module must SKIP, not fail — verify: `uv run --no-sync pytest tests/adapters/test_noun_chunk_extractor.py -v` in a lean env reports skipped.)

- [ ] **Step 3: Implement**

`src/lattice/adapters/extractor/noun_chunk.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register

_TRIM_POS = {"DET", "PRON"}


@register(Extractor, "noun-chunk")
class NounChunkExtractor(Extractor):
    """PoS-bounded noun-phrase candidates via spaCy noun chunks (M2 spec §6.1).
    Leading determiners/pronouns are trimmed; over-long and pronoun-only
    chunks are dropped. spaCy is imported lazily so this module is importable
    without the ml dependency group."""

    def __init__(self, model: str = "en_core_web_sm", max_tokens: int = 5):
        import spacy

        self.nlp = spacy.load(model, disable=["ner"])
        self.max_tokens = max_tokens

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            doc = self.nlp(unit.text)
            for chunk in doc.noun_chunks:
                tokens = list(chunk)
                while tokens and tokens[0].pos_ in _TRIM_POS:
                    tokens = tokens[1:]
                if not tokens or len(tokens) > self.max_tokens:
                    continue
                start = tokens[0].idx
                end = tokens[-1].idx + len(tokens[-1].text)
                mentions.append(
                    Mention(
                        surface=unit.text[start:end].lower(),
                        unit_id=unit.id,
                        span=(start, end),
                        context=unit.text,
                        head=chunk.root.text.lower(),
                        lemma=" ".join(t.lemma_ for t in tokens).lower(),
                    )
                )
        return mentions
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.extractor import noun_chunk  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv sync --group ml && chflags nohidden .venv/lib/python*/site-packages/*.pth && uv run --no-sync python scripts/fetch_models.py` (first time only), then
`uv run --no-sync pytest tests/adapters/test_noun_chunk_extractor.py -v`
Expected: `7 passed` (3 contract + 4 specific).

- [ ] **Step 5: Lean-env check + full suite + commit**

Verify module import stays lazy: `uv run --no-sync python -c "import lattice.adapters; print('ok')"`
Expected: `ok` (works even before the ml sync — spaCy is only imported in `__init__`).

Run: `uv run --no-sync pytest -q` — Expected: `167 passed` (or `160 passed, 7 skipped` in a lean env).

```bash
git add src/lattice/adapters/ tests/adapters/test_noun_chunk_extractor.py
git commit -m "feat: add spaCy noun-chunk extractor adapter (ml)"
```

---

### Task 10: Sentence-transformer embedder (ml)

**Files:**
- Create: `src/lattice/adapters/embedder/sentence_transformer.py`
- Modify: `src/lattice/adapters/__init__.py` (add import line)
- Test: `tests/adapters/test_sentence_transformer_embedder.py` (ml-marked)

**Interfaces:**
- Consumes: `Embedder` port + `EmbedderContract` (M1); sentence-transformers (imported ONLY inside `__init__`).
- Produces: `SentenceTransformerEmbedder(model: str = "all-MiniLM-L6-v2", batch_size: int = 32, device: str = "cpu")` registered `(Embedder, "sentence-transformer")`; L2-normalized tuples; `dim` from the model (384 for MiniLM).

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_sentence_transformer_embedder.py`:

```python
import math

import pytest

pytestmark = pytest.mark.ml
pytest.importorskip("sentence_transformers")

from lattice.adapters.embedder.sentence_transformer import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from tests.contracts.embedder_contract import EmbedderContract  # noqa: E402


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except OSError:
        pytest.skip("all-MiniLM-L6-v2 not cached (run scripts/fetch_models.py)")


class TestSentenceTransformerEmbedder(EmbedderContract):
    _instance = None

    def make_embedder(self) -> SentenceTransformerEmbedder:
        if TestSentenceTransformerEmbedder._instance is None:
            try:
                TestSentenceTransformerEmbedder._instance = SentenceTransformerEmbedder()
            except OSError:
                pytest.skip("all-MiniLM-L6-v2 not cached (run scripts/fetch_models.py)")
        return TestSentenceTransformerEmbedder._instance

    def test_dim_is_384_for_minilm(self):
        assert self.make_embedder().dim == 384

    def test_vectors_are_unit_normalized(self):
        [vector] = self.make_embedder().embed(["vector store"])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-5)

    def test_semantic_neighbors_beat_strangers(self):
        embedder = self.make_embedder()
        a, b, c = embedder.embed(["vector database", "vector store", "banana bread recipe"])
        from lattice.core.vectors import cosine

        assert cosine(a, b) > cosine(a, c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/adapters/test_sentence_transformer_embedder.py -v`
Expected (ml env): FAIL — `ModuleNotFoundError` for the adapter module. (Lean env: all skipped.)

- [ ] **Step 3: Implement**

`src/lattice/adapters/embedder/sentence_transformer.py`:

```python
from collections.abc import Sequence

from lattice.ports import Embedder
from lattice.registry.registry import register


@register(Embedder, "sentence-transformer")
class SentenceTransformerEmbedder(Embedder):
    """Real semantic embedder (M2 spec §6.2). Default all-MiniLM-L6-v2 for
    literature parity (parent spec §14). Deterministic inference; outputs are
    L2-normalized. sentence-transformers is imported lazily so this module is
    importable without the ml dependency group."""

    def __init__(
        self, model: str = "all-MiniLM-L6-v2", batch_size: int = 32, device: str = "cpu"
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self._batch_size = batch_size
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [tuple(float(x) for x in vector) for vector in vectors]
```

Append to `src/lattice/adapters/__init__.py`:

```python
from lattice.adapters.embedder import sentence_transformer  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_sentence_transformer_embedder.py -v`
Expected: `7 passed` (4 contract + 3 specific).

- [ ] **Step 5: Full suite + commit**

Run: `uv run --no-sync pytest -q` — Expected: `174 passed` (ml env) / lean env: same count minus 14 skipped.

```bash
git add src/lattice/adapters/ tests/adapters/test_sentence_transformer_embedder.py
git commit -m "feat: add sentence-transformer embedder adapter (ml)"
```

---

### Task 11: Baseline configs + end-to-end + exit criteria

**Files:**
- Create: `configs/m2a-baseline.toml`, `configs/m2a-baseline-sweep.toml`
- Test: `tests/harness/test_m2a_e2e.py` (one pure test + ml-marked tests)

**Interfaces:**
- Consumes: everything above.
- Produces: the M2a deliverable — a runnable baseline experiment and sweep; e2e proof on the mini fixture both with and without the ml stack.

- [ ] **Step 1: Write the configs**

`configs/m2a-baseline.toml`:

```toml
# M2a baseline: real extraction + salience on the Inspec test split.
# Prereqs: uv sync --group ml && uv run --group ml python scripts/fetch_models.py
#          && uv run --group ml python scripts/fetch_datasets.py inspec

[segmenter]
name = "block"

[extractor]
name = "noun-chunk"
[extractor.params]
max_tokens = 5

[scorer]
name = "embedding-cosine"
[scorer.params]
top_k = 15

[resolver]
name = "exact-label"

[relation_inducer]
name = "co-occurrence"

[graph_integrator]
name = "in-memory"

[embedder]
name = "sentence-transformer"
[embedder.params]
model = "all-MiniLM-L6-v2"

[concept_store]
name = "in-memory"

[run]
on_error = "fail"
seed = 0

[dataset]
name = "inspec"
[dataset.params]
split = "test"

[[metrics]]
name = "label-f1"

[[document_metrics]]
name = "f1-at-k"
```

`configs/m2a-baseline-sweep.toml`:

```toml
# First sweep: the trivial frequency scorer vs the embedding-cosine baseline.

[base.segmenter]
name = "block"

[base.extractor]
name = "noun-chunk"

[base.resolver]
name = "exact-label"

[base.relation_inducer]
name = "co-occurrence"

[base.graph_integrator]
name = "in-memory"

[base.embedder]
name = "sentence-transformer"

[base.concept_store]
name = "in-memory"

[base.run]
on_error = "fail"
seed = 0

[base.dataset]
name = "inspec"
[base.dataset.params]
split = "test"

[base.scorer]
name = "embedding-cosine"

[[base.document_metrics]]
name = "f1-at-k"

[axes]
scorer = [
  { name = "frequency", params = { top_k = 15 } },
  { name = "embedding-cosine", params = { top_k = 15 } },
]
```

- [ ] **Step 2: Write the failing tests**

`tests/harness/test_m2a_e2e.py`:

```python
import pytest

from lattice.harness.runner import ExperimentConfig, run_experiment

MINI_ROOT = "tests/fixtures/mini_inspec"


def _config(extractor: dict, embedder: dict, scorer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": extractor,
            "scorer": scorer,
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": embedder,
            "dataset": {"name": "inspec", "params": {"root": MINI_ROOT}},
            "document_metrics": [{"name": "f1-at-k"}],
        }
    )


def test_document_metric_pipeline_pure():
    """The full per-document evaluation path with M1's pure adapters — proves
    the M2a wiring without the ml stack."""
    report = run_experiment(
        _config(
            extractor={"name": "token", "params": {"min_length": 4}},
            embedder={"name": "hashing"},
            scorer={"name": "frequency", "params": {"top_k": 15}},
        )
    )
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["f1-at-k"]) == {
        "precision@5", "recall@5", "f1@5",
        "precision@10", "recall@10", "f1@10",
        "precision@15", "recall@15", "f1@15",
    }


@pytest.mark.ml
def test_real_baseline_on_mini_fixture():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(
            _config(
                extractor={"name": "noun-chunk"},
                embedder={"name": "sentence-transformer"},
                scorer={"name": "embedding-cosine", "params": {"top_k": 15}},
            )
        )
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert report.metrics["f1-at-k"]["recall@10"] > 0.4  # gold phrases appear verbatim
    assert report.config["embedder"]["name"] == "sentence-transformer"


@pytest.mark.ml
def test_real_baseline_is_reproducible():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    config = _config(
        extractor={"name": "noun-chunk"},
        embedder={"name": "sentence-transformer"},
        scorer={"name": "embedding-cosine", "params": {"top_k": 15}},
    )
    try:
        assert run_experiment(config) == run_experiment(config)
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
```

- [ ] **Step 3: Run tests**

Run: `uv run --no-sync pytest tests/harness/test_m2a_e2e.py -v`
Expected: `3 passed` in an ml env with models; `1 passed, 2 skipped` lean. The pure test must pass BEFORE any ml install.

- [ ] **Step 4: Full suite, both environments**

Run: `uv run --no-sync pytest -q`
Expected (ml env with models): `177 passed`. Record the exact counts in the task report for both this and (if practical) a lean check.

- [ ] **Step 5: Exit-criteria dry run (manual, documented in the report)**

If `data/inspec/` exists (fetch scripts run):

```bash
uv run --no-sync python -m lattice.harness --sweep configs/m2a-baseline-sweep.toml
```

Expected: a two-row markdown table (frequency vs embedding-cosine); `f1@10` for embedding-cosine strictly greater than for frequency, and in the 0.15–0.40 sanity band (M2 spec §11 gives ~0.25–0.35 for tuned literature baselines; the untuned first run may land below — record the actual numbers in the report, do NOT tune in this task). If the dataset has not been fetched, note that the run awaits `scripts/fetch_datasets.py` and complete the task on the mini-fixture evidence.

- [ ] **Step 6: Commit**

```bash
git add configs/ tests/harness/test_m2a_e2e.py
git commit -m "feat: add M2a baseline configs and end-to-end evaluation tests"
```

---

## M2a exit criteria

- Default (lean) suite green without the `ml` group; full suite green with it.
- `python -m lattice.harness --sweep configs/m2a-baseline-sweep.toml` produces a reproducible two-scorer table on Inspec (after one-time fetch scripts).
- `ruff check .` clean; `py.typed` shipped.
- Every new adapter passes its port's contract suite; `DocumentMetric` has its own contract.

## Deferred to M2b (separate plan)

- MDERank scorer (masked-document re-embedding) and HCUKE scorer (hierarchical significance) — spec §6.6/§6.7.
- SemEval/DUC datasets; PromptRank decision (post-M2b).
