# Statistical Intervals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shipped, tested `lattice.harness.stats` layer that turns the M2–M5 point-estimate metrics into interval claims (bootstrap CIs, paired-delta, threshold-sensitivity, order-permutation).

**Architecture:** Metrics opt into resampling via a `Resamplable` mixin: they declare a `kind` (`macro`/`pooled`/`holistic`) and, for macro/pooled, emit per-document records plus a pure `aggregate(records, global_context)` that recomputes the metric — verified to equal the frozen `evaluate()` on the full document set. A generic engine resamples document ids (with replacement, seeded) and drives each metric's own `aggregate`; holistic metrics re-run the pipeline per resample. Intervals (BCa/percentile/paired) are pure stdlib math on the resample distribution.

**Tech Stack:** Python 3.13, stdlib only for the stats layer (`random`, `math`, `statistics.NormalDist`), pydantic configs, pytest. No new dependencies.

## Global Constraints

- **Additive only.** No existing type, port method, adapter behavior, config file, or recorded result changes. `Metric.evaluate` / `DocumentMetric.evaluate_documents` stay byte-identical.
- **No new dependencies** — `pyproject.toml` is frozen. The stats layer is stdlib-only.
- **No model downloads or network in tests.** Every test uses the toy fixtures under `tests/fixtures/` and the `hashing` embedder. No `@pytest.mark.ml` test is required or added.
- **Seeded and reproducible.** All randomness flows through one `random.Random(seed)`; a fixed seed produces identical output.
- **Equivalence is law.** For every opted-in metric, `aggregate(emit_records(full_context))` MUST equal `evaluate(...)` exactly on the full document set — the interval is centered on the published number.
- **Commit rules.** `reports/**` and datasets are gitignored (regenerable); `docs/results/*.md` IS committed. Never commit under `reports/`.
- **Statistical defaults:** BCa 95% CI (percentile alongside); B = 10000 item-level, 1000 holistic; order permutations K = 40; resolver threshold grid 0.65–0.95 step 0.05.
- Existing suite (413 tests) passes unchanged after every task.
- Run tests with: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest ...`

---

### Task 1: Stats package foundation — records, mixin, context

**Files:**
- Create: `src/lattice/harness/stats/__init__.py`
- Create: `src/lattice/harness/stats/records.py`
- Test: `tests/harness/stats/__init__.py` (empty), `tests/harness/stats/test_records.py`

**Interfaces:**
- Produces:
  - `EvaluationContext(deltas: tuple[GraphDelta, ...], snapshot: GraphSnapshot, ground_truth: dict[str, object])` — frozen dataclass.
  - `ResampleBundle(kind: str, per_document: dict[str, object], aggregate: Callable[[list, dict], dict[str, float]], global_context: dict = {})` — frozen dataclass; `aggregate` recomputes metric values from a list of per-document records (multiplicity-preserving) plus `global_context`.
  - `Resamplable` mixin: class attribute `kind: str = ""`; method `emit_records(self, context: EvaluationContext) -> ResampleBundle` (default raises `NotImplementedError`). Metrics inherit this alongside `Metric`/`DocumentMetric`.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_records.py
import pytest

from lattice.core.types import GraphSnapshot
from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable


def test_evaluation_context_holds_the_three_inputs():
    ctx = EvaluationContext(deltas=(), snapshot=GraphSnapshot(concepts=(), relations=()), ground_truth={"k": 1})
    assert ctx.ground_truth == {"k": 1}
    assert ctx.snapshot.concepts == ()


def test_resample_bundle_drives_its_aggregate():
    bundle = ResampleBundle(
        kind="macro",
        per_document={"d1": {"f1": 1.0}, "d2": {"f1": 0.0}},
        aggregate=lambda records, ctx: {"f1": sum(r["f1"] for r in records) / len(records)},
    )
    picked = [bundle.per_document["d1"], bundle.per_document["d2"]]
    assert bundle.aggregate(picked, bundle.global_context) == {"f1": 0.5}
    assert bundle.global_context == {}


def test_resamplable_default_emit_raises():
    class M(Resamplable):
        pass
    assert M().kind == ""
    with pytest.raises(NotImplementedError):
        M().emit_records(EvaluationContext((), GraphSnapshot((), ()), {}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lattice.harness.stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lattice/harness/stats/__init__.py
"""Statistical-intervals layer over the benchmark harness (docs/2026-07-14-statistical-intervals-design.md)."""
```

```python
# src/lattice/harness/stats/records.py
"""Per-document detail capture for resampling. A resamplable metric declares
its `kind` and emits a ResampleBundle whose `aggregate` recomputes the metric
from a (possibly multiplicity-bearing) list of per-document records — proven
equal to the metric's own evaluate() on the full document set."""

from collections.abc import Callable
from dataclasses import dataclass, field

from lattice.core.types import GraphDelta, GraphSnapshot


@dataclass(frozen=True)
class EvaluationContext:
    deltas: tuple[GraphDelta, ...]
    snapshot: GraphSnapshot
    ground_truth: dict[str, object]


@dataclass(frozen=True)
class ResampleBundle:
    kind: str
    per_document: dict[str, object]
    aggregate: Callable[[list, dict], dict[str, float]]
    global_context: dict = field(default_factory=dict)


class Resamplable:
    """Opt-in marker. `kind` is "macro" | "pooled" | "holistic". macro/pooled
    metrics implement emit_records; holistic metrics do not (the engine
    re-runs the pipeline for them)."""

    kind: str = ""

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_records.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/stats/__init__.py src/lattice/harness/stats/records.py tests/harness/stats/
git commit -m "feat: stats records, ResampleBundle, Resamplable mixin"
```

---

### Task 2: Interval math — percentile, BCa, paired delta

**Files:**
- Create: `src/lattice/harness/stats/intervals.py`
- Test: `tests/harness/stats/test_intervals.py`

**Interfaces:**
- Produces:
  - `Interval(lo: float, hi: float, method: str)` — frozen dataclass.
  - `DeltaResult(estimate: float, lo: float, hi: float, prob_positive: float)` — frozen dataclass.
  - `percentile_interval(estimate: float, resamples: list[float], level: float = 0.95) -> Interval`
  - `bca_interval(estimate: float, resamples: list[float], jackknife: list[float], level: float = 0.95) -> Interval`
  - `paired_delta(resamples_a: list[float], resamples_b: list[float], estimate_a: float, estimate_b: float, level: float = 0.95) -> DeltaResult`

All pure; no pipeline. Implementations below are verified (scratchpad `verify_stats.py`, 14/14).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_intervals.py
from lattice.harness.stats.intervals import (
    bca_interval, paired_delta, percentile_interval,
)


def test_percentile_linear_interpolation():
    # 0..99: numpy linear 2.5th/97.5th percentiles are 2.475 and 97.525
    iv = percentile_interval(0.0, [float(i) for i in range(100)])
    assert iv.method == "percentile"
    assert abs(iv.lo - 2.475) < 1e-9
    assert abs(iv.hi - 97.525) < 1e-9


def test_bca_reduces_to_percentile_when_symmetric():
    # estimate 0, exactly half strictly below -> z0=0; symmetric jackknife -> acc=0
    resamples = list(range(-50, 0)) + list(range(1, 51))
    resamples = [float(x) for x in resamples]
    iv = bca_interval(0.0, resamples, [1.0, -1.0, 2.0, -2.0, 0.0])
    pv = percentile_interval(0.0, resamples)
    assert iv.method == "bca"
    assert abs(iv.lo - pv.lo) < 1e-9 and abs(iv.hi - pv.hi) < 1e-9


def test_bca_degenerate_zero_width():
    iv = bca_interval(5.0, [5.0] * 20, [5.0] * 4)
    assert (iv.lo, iv.hi, iv.method) == (5.0, 5.0, "degenerate")


def test_bca_falls_back_when_estimate_outside_resamples():
    iv = bca_interval(0.0, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert iv.method == "percentile-fallback"


def test_paired_delta_sign_and_probability():
    d = paired_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 2.0, 3.0)
    assert d.estimate == -1.0
    assert d.prob_positive == 0.0
    assert d.lo == -1.0 and d.hi == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_intervals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lattice.harness.stats.intervals'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lattice/harness/stats/intervals.py
"""Bootstrap confidence intervals. Percentile and BCa (bias-corrected and
accelerated); paired delta for comparative claims. Stdlib only —
statistics.NormalDist supplies the normal CDF and its inverse."""

import math
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    method: str


@dataclass(frozen=True)
class DeltaResult:
    estimate: float
    lo: float
    hi: float
    prob_positive: float


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy 'linear' method): position q*(n-1)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def percentile_interval(estimate: float, resamples: list[float], level: float = 0.95) -> Interval:
    s = sorted(resamples)
    a = (1 - level) / 2
    return Interval(_percentile(s, a), _percentile(s, 1 - a), "percentile")


def bca_interval(
    estimate: float, resamples: list[float], jackknife: list[float], level: float = 0.95
) -> Interval:
    s = sorted(resamples)
    a = (1 - level) / 2
    if s[0] == s[-1]:
        return Interval(estimate, estimate, "degenerate")
    prop = sum(1 for r in resamples if r < estimate) / len(resamples)
    if prop <= 0.0 or prop >= 1.0:
        return Interval(_percentile(s, a), _percentile(s, 1 - a), "percentile-fallback")
    z0 = _N.inv_cdf(prop)
    jbar = sum(jackknife) / len(jackknife)
    num = sum((jbar - j) ** 3 for j in jackknife)
    den = 6.0 * (sum((jbar - j) ** 2 for j in jackknife)) ** 1.5
    acc = num / den if den != 0 else 0.0
    bounds = []
    for z_a in (_N.inv_cdf(a), _N.inv_cdf(1 - a)):
        adj = z0 + (z0 + z_a) / (1 - acc * (z0 + z_a))
        bounds.append(_percentile(s, _N.cdf(adj)))
    return Interval(bounds[0], bounds[1], "bca")


def paired_delta(
    resamples_a: list[float], resamples_b: list[float],
    estimate_a: float, estimate_b: float, level: float = 0.95,
) -> DeltaResult:
    deltas = [x - y for x, y in zip(resamples_a, resamples_b)]
    iv = percentile_interval(estimate_a - estimate_b, deltas, level)
    prob = sum(1 for d in deltas if d > 0) / len(deltas)
    return DeltaResult(estimate_a - estimate_b, iv.lo, iv.hi, prob)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_intervals.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/stats/intervals.py tests/harness/stats/test_intervals.py
git commit -m "feat: percentile/BCa intervals and paired delta"
```

---

### Task 3: Runner detail capture — run_on_documents, run_experiment_detailed

**Files:**
- Modify: `src/lattice/harness/runner.py` (add two functions + imports; do NOT touch `run_experiment`/`run_from_path`)
- Test: `tests/harness/stats/test_runner_detailed.py`

**Interfaces:**
- Consumes: `EvaluationContext`, `Resamplable`, `ResampleBundle` (Task 1); existing `build_orchestrator`, `instantiate`, `ExperimentConfig`.
- Produces:
  - `run_on_documents(config: ExperimentConfig, documents: Sequence[Document]) -> dict[str, float]` — build orchestrator, `process_stream(documents)`, snapshot, evaluate all `config.metrics` + `config.document_metrics` (with the shared embedder), return a FLAT dict keyed `f"{spec.name}.{key}"`. Used by holistic bootstrap and permutation.
  - `run_experiment_detailed(config: ExperimentConfig) -> tuple[RunReport, dict[str, ResampleBundle]]` — run the pipeline once; for each metric/document_metric that is `Resamplable` with `kind in ("macro","pooled")`, call `emit_records(context)` and collect bundles keyed by `spec.name`. Holistic and non-resamplable metrics contribute no bundle.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_runner_detailed.py
from lattice.core.types import GraphDelta
from lattice.harness.runner import (
    ExperimentConfig, run_experiment, run_experiment_detailed, run_on_documents,
)
from lattice.config.factory import instantiate
from lattice.ports import Dataset

M5 = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gold-mentions", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "embedding-nn", "params": {"threshold": 0.8}},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "mention-clusters", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "metrics": [{"name": "redundancy"}],
    "document_metrics": [{"name": "clustering"}],
})


def test_run_on_documents_matches_run_experiment_on_full_stream():
    docs = list(instantiate(Dataset, M5.dataset).documents())
    flat = run_on_documents(M5, docs)
    report = run_experiment(M5)
    expected = {
        f"{name}.{k}": v
        for name, values in report.metrics.items()
        for k, v in values.items()
    }
    assert flat == expected


# A test-only resamplable metric, registered at MODULE level (import-time, once):
# registry.register raises on duplicate names, so never register inside a test body.
from lattice.harness.stats.records import Resamplable, ResampleBundle  # noqa: E402
from lattice.ports import DocumentMetric  # noqa: E402
from lattice.registry.registry import register  # noqa: E402


@register(DocumentMetric, "toy-macro")
class ToyMacro(DocumentMetric, Resamplable):
    kind = "macro"

    def evaluate_documents(self, deltas, ground_truth):
        return {"n": float(len(list(deltas)))}

    def emit_records(self, context):
        return ResampleBundle(
            kind="macro",
            per_document={d.document_id: {"n": 1.0} for d in context.deltas},
            aggregate=lambda records, ctx: {"n": sum(r["n"] for r in records)},
        )


def test_detailed_returns_bundles_for_resamplable_only():
    # metrics=[] so only the resamplable toy-macro document_metric yields a bundle.
    config = ExperimentConfig.model_validate(
        M5.model_copy(update={"document_metrics": [{"name": "toy-macro"}], "metrics": []}).model_dump()
    )
    report, bundles = run_experiment_detailed(config)
    assert set(bundles) == {"toy-macro"}
    assert bundles["toy-macro"].kind == "macro"
    assert set(bundles["toy-macro"].per_document)         # one record per document
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_runner_detailed.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_on_documents'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/lattice/harness/runner.py` (after `run_experiment`, before `run_from_path`). Add imports at the top: `from collections.abc import Sequence`, `from lattice.core.types import Document`, `from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable`.

```python
def run_on_documents(
    config: ExperimentConfig, documents: "Sequence[Document]"
) -> dict[str, float]:
    """Run the pipeline over an explicit document list (a resample or a
    permutation) and return a flat {"<metric>.<key>": value} dict. Mirrors
    run_experiment's scoring but with a caller-supplied stream."""
    duplicates = {s.name for s in config.metrics} & {s.name for s in config.document_metrics}
    if duplicates:
        raise ValueError(f"metric name(s) used by both families: {sorted(duplicates)}")
    orchestrator = build_orchestrator(config)
    deltas = orchestrator.process_stream(documents)
    snapshot = orchestrator.snapshot()
    dataset = instantiate(Dataset, config.dataset)
    ground_truth = dataset.ground_truth()
    metric_shared: dict[str, object] = {"embedder": instantiate(Embedder, config.embedder)}
    flat: dict[str, float] = {}
    for spec in config.metrics:
        for key, value in instantiate(Metric, spec, metric_shared).evaluate(snapshot, ground_truth).items():
            flat[f"{spec.name}.{key}"] = value
    for spec in config.document_metrics:
        for key, value in instantiate(DocumentMetric, spec, metric_shared).evaluate_documents(deltas, ground_truth).items():
            flat[f"{spec.name}.{key}"] = value
    return flat


def run_experiment_detailed(
    config: ExperimentConfig,
) -> tuple[RunReport, dict[str, ResampleBundle]]:
    """Run the experiment once and, for every macro/pooled Resamplable metric,
    capture a ResampleBundle of per-document detail for item-level bootstrap.
    Holistic and non-resamplable metrics contribute no bundle."""
    duplicates = {s.name for s in config.metrics} & {s.name for s in config.document_metrics}
    if duplicates:
        raise ValueError(f"metric name(s) used by both families: {sorted(duplicates)}")
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    deltas = orchestrator.process_stream(dataset.documents())
    snapshot = orchestrator.snapshot()
    ground_truth = dataset.ground_truth()
    metric_shared: dict[str, object] = {"embedder": instantiate(Embedder, config.embedder)}
    context = EvaluationContext(tuple(deltas), snapshot, ground_truth)
    metric_results: dict[str, dict[str, float]] = {}
    document_results: dict[str, dict[str, float]] = {}
    bundles: dict[str, ResampleBundle] = {}
    for spec in config.metrics:
        metric = instantiate(Metric, spec, metric_shared)
        metric_results[spec.name] = metric.evaluate(snapshot, ground_truth)
        if isinstance(metric, Resamplable) and metric.kind in ("macro", "pooled"):
            bundles[spec.name] = metric.emit_records(context)
    for spec in config.document_metrics:
        metric = instantiate(DocumentMetric, spec, metric_shared)
        document_results[spec.name] = metric.evaluate_documents(deltas, ground_truth)
        if isinstance(metric, Resamplable) and metric.kind in ("macro", "pooled"):
            bundles[spec.name] = metric.emit_records(context)
    report = RunReport(
        config=config.model_dump(),
        documents_processed=len(deltas),
        errors=tuple(error for delta in deltas for error in delta.errors),
        metrics={**metric_results, **document_results},
    )
    return report, bundles
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_runner_detailed.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/runner.py tests/harness/stats/test_runner_detailed.py
git commit -m "feat: run_on_documents + run_experiment_detailed with bundle capture"
```

---

### Task 4: f1-at-k opts into resampling (macro)

**Files:**
- Modify: `src/lattice/adapters/document_metric/f1_at_k.py`
- Test: `tests/harness/stats/test_f1_at_k_resample.py`

**Interfaces:**
- Consumes: `Resamplable`, `ResampleBundle`, `EvaluationContext` (Task 1); `run_experiment_detailed` (Task 3).
- Produces: `F1AtK` gains `kind = "macro"`, a `_per_document_scores(delta) -> dict[str, float]` helper shared by `evaluate_documents` and `emit_records`, a static `_aggregate(records, ctx)` that means each key, and `emit_records`.

The refactor keeps `evaluate_documents`' return identical (equivalence by shared code path).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_f1_at_k_resample.py
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gold-mentions", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "mention-clusters", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "document_metrics": [{"name": "clustering"}],
})


def test_f1_at_k_equivalence_via_unit_deltas():
    # Exercise emit_records/_aggregate directly on hand-made deltas (like tests/helpers.make_delta).
    from lattice.adapters.document_metric.f1_at_k import F1AtK
    from lattice.harness.stats.records import EvaluationContext
    from lattice.core.types import GraphSnapshot
    from tests.helpers import make_delta

    gt = {"keyphrases_by_document": {"d1": ["alpha"], "d2": ["beta"]}}
    deltas = (make_delta("d1", [("alpha", 1.0)]), make_delta("d2", [("gamma", 1.0)]))
    metric = F1AtK(ks=[5])
    assert metric.kind == "macro"
    direct = metric.evaluate_documents(deltas, gt)
    bundle = metric.emit_records(EvaluationContext(deltas, GraphSnapshot((), ()), gt))
    records = [bundle.per_document[d.document_id] for d in deltas]
    assert bundle.aggregate(records, bundle.global_context) == direct
    assert direct["f1@5"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_f1_at_k_resample.py -v`
Expected: FAIL with `AttributeError: 'F1AtK' object has no attribute 'kind'` (or `emit_records`)

- [ ] **Step 3: Write minimal implementation**

Rewrite `src/lattice/adapters/document_metric/f1_at_k.py` to share per-document scoring. Change the class declaration to `class F1AtK(DocumentMetric, Resamplable):`, add `kind = "macro"`, extract the per-document loop into `_per_document_scores`, and add `_aggregate` + `emit_records`. Add imports: `from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable`.

```python
    kind = "macro"

    def _per_document_scores(self, delta: GraphDelta, by_document: dict) -> dict[str, float]:
        if delta.document_id not in by_document:
            raise ValueError(f"document {delta.document_id!r} missing from ground truth")
        gold = {self._stem_phrase(p) for p in by_document[delta.document_id]}
        ranked = self._ranked_unique_surfaces(delta)
        scores: dict[str, float] = {}
        for k in self.ks:
            predicted = {self._stem_phrase(s) for s in ranked[:k]}
            tp = len(gold & predicted)
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(gold) if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            scores[f"precision@{k}"] = precision
            scores[f"recall@{k}"] = recall
            scores[f"f1@{k}"] = f1
        return scores

    def evaluate_documents(self, deltas, ground_truth):
        by_document = ground_truth.get("keyphrases_by_document")
        if not isinstance(by_document, dict):
            raise ValueError('f1-at-k requires ground_truth["keyphrases_by_document"]')
        deltas = list(deltas)
        if not deltas:
            raise ValueError("no documents to evaluate")
        per_doc = [self._per_document_scores(d, by_document) for d in deltas]
        return self._aggregate(per_doc, {})

    @staticmethod
    def _aggregate(records: list, ctx: dict) -> dict[str, float]:
        keys = records[0].keys()
        n = len(records)
        return {k: sum(r[k] for r in records) / n for k in keys}

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        by_document = context.ground_truth.get("keyphrases_by_document")
        if not isinstance(by_document, dict):
            raise ValueError('f1-at-k requires ground_truth["keyphrases_by_document"]')
        return ResampleBundle(
            kind="macro",
            per_document={d.document_id: self._per_document_scores(d, by_document) for d in context.deltas},
            aggregate=self._aggregate,
        )
```

Note: `evaluate_documents` now returns `_aggregate(per_doc, {})`, which yields exactly the same key set and values as before (mean of per-document precision/recall/f1 for each k) — the existing `tests/adapters/test_f1_at_k.py` must still pass unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_f1_at_k_resample.py tests/adapters/test_f1_at_k.py -v`
Expected: PASS (new test + all pre-existing f1-at-k tests unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/adapters/document_metric/f1_at_k.py tests/harness/stats/test_f1_at_k_resample.py
git commit -m "feat: f1-at-k opts into macro resampling"
```

---

### Task 5: edge-f1 opts into resampling (pooled, provenance from deltas)

**Files:**
- Modify: `src/lattice/adapters/metric/edge_f1.py`
- Test: `tests/harness/stats/test_edge_f1_resample.py`

**Interfaces:**
- Consumes: `Resamplable`, `ResampleBundle`, `EvaluationContext` (Task 1); `run_experiment_detailed` (Task 3).
- Produces: `EdgeF1` gains `kind = "pooled"`, `_aggregate(records, ctx)` (union of per-doc label-pair sets vs `ctx["gold"]` → P/R/F1 + counts), and `emit_records` (per-document IS_A label-pairs from `delta.relations_added`, using the snapshot's id→label map; gold in `global_context`).

Equivalence verified: union of per-document sets == the snapshot's predicted IS_A label-pair set (integrator dedups by (type,src,tgt), but the union of label-pairs is identical).

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_edge_f1_resample.py
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gazetteer", "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "union", "params": {"members": [{"name": "hearst"}, {"name": "compound"}]}},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "taxonomy", "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"}},
    "metrics": [{"name": "edge-f1"}],
})


def test_edge_f1_equivalence_and_bundle_shape():
    report, bundles = run_experiment_detailed(CFG)
    assert set(bundles) == {"edge-f1"}
    bundle = bundles["edge-f1"]
    assert bundle.kind == "pooled"
    doc_ids = list(bundle.per_document)
    records = [bundle.per_document[d] for d in doc_ids]
    assert bundle.aggregate(records, bundle.global_context) == report.metrics["edge-f1"]
    # union of per-doc predicted edges equals the recomputed predicted_edges count
    union = set().union(*records) if records else set()
    assert float(len(union)) == report.metrics["edge-f1"]["predicted_edges"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_edge_f1_resample.py -v`
Expected: FAIL with `AttributeError` on `kind` / `KeyError: 'edge-f1'` in bundles

- [ ] **Step 3: Write minimal implementation**

Modify `src/lattice/adapters/metric/edge_f1.py`: change the class to `class EdgeF1(Metric, Resamplable):`, add `kind = "pooled"`, `_aggregate`, and `emit_records`. Add import `from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable`. `evaluate` is UNCHANGED.

```python
    kind = "pooled"

    @staticmethod
    def _aggregate(records: list, ctx: dict) -> dict[str, float]:
        predicted: set = set()
        for record in records:
            predicted |= record
        gold = ctx["gold"]
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "predicted_edges": float(len(predicted)), "gold_edges": float(len(gold)),
        }

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        label_of = {concept.id: concept.label.lower() for concept in context.snapshot.concepts}
        per_document = {
            delta.document_id: frozenset(
                (label_of[r.source_id], label_of[r.target_id])
                for r in delta.relations_added if r.type == "IS_A"
            )
            for delta in context.deltas
        }
        gold = frozenset(
            (str(hypo).lower(), str(hyper).lower())
            for hypo, hyper in context.ground_truth.get("is_a_edges", [])
        )
        return ResampleBundle(
            kind="pooled", per_document=per_document,
            aggregate=self._aggregate, global_context={"gold": gold},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_edge_f1_resample.py tests/harness/test_m4_e2e.py -v`
Expected: PASS (new test + M4 e2e unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/adapters/metric/edge_f1.py tests/harness/stats/test_edge_f1_resample.py
git commit -m "feat: edge-f1 opts into pooled resampling (provenance from deltas)"
```

---

### Task 6: clustering (B³/ARI) opts into resampling (pooled, unique re-keying)

**Files:**
- Modify: `src/lattice/adapters/document_metric/clustering.py`
- Test: `tests/harness/stats/test_clustering_resample.py`

**Interfaces:**
- Consumes: `Resamplable`, `ResampleBundle`, `EvaluationContext` (Task 1); `run_experiment_detailed` (Task 3); existing `_b_cubed`, `_ari`.
- Produces: `ClusteringMetric` gains `kind = "pooled"`, `_aggregate(records, ctx)` (flatten per-document `(mention_key, pred_cluster, gold_cluster)` rows, prefixing each with its list index so duplicated documents keep distinct mention keys — multiplicity-preserving — then `_b_cubed`/`_ari`), and `emit_records`.

Equivalence verified: on the full set, unique prefixes make keys distinct but preserve the partition, so `_b_cubed`/`_ari` return identical values to `evaluate_documents`.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_clustering_resample.py
import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gold-mentions", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "embedding-nn", "params": {"threshold": 0.8}},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "mention-clusters", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "document_metrics": [{"name": "clustering"}],
})


def test_clustering_equivalence():
    report, bundles = run_experiment_detailed(CFG)
    bundle = bundles["clustering"]
    assert bundle.kind == "pooled"
    doc_ids = list(bundle.per_document)
    records = [bundle.per_document[d] for d in doc_ids]
    assert bundle.aggregate(records, bundle.global_context) == report.metrics["clustering"]


def test_clustering_aggregate_respects_multiplicity():
    # Hand-built records: doc A has one perfect-precision mention; doc B has two
    # mentions sharing a predicted cluster split across two gold clusters
    # (precision 1/2 each). Duplicating doc A must reweight the B3-precision mean
    # toward A — which only happens if _aggregate re-keys each document instance
    # uniquely. A collapsed (non-prefixed) implementation gives the SAME value for
    # [A,B] and [A,A,B], so this test fails iff the index-prefixing is removed.
    # Verified against the real _aggregate: 2/3 and 3/4.
    doc_a = [("A:0-1", "C1", "G1")]
    doc_b = [("B:0-1", "C2", "G2"), ("B:2-3", "C2", "G3")]
    base = ClusteringMetric._aggregate([doc_a, doc_b], {})
    dup = ClusteringMetric._aggregate([doc_a, doc_a, doc_b], {})
    assert base["b3-precision"] == pytest.approx(2 / 3)   # (1 + 1/2 + 1/2) / 3
    assert dup["b3-precision"] == pytest.approx(3 / 4)     # (1 + 1 + 1/2 + 1/2) / 4
    assert dup["b3-precision"] != base["b3-precision"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_clustering_resample.py -v`
Expected: FAIL with `KeyError: 'clustering'` in bundles (metric not yet Resamplable)

- [ ] **Step 3: Write minimal implementation**

Modify `src/lattice/adapters/document_metric/clustering.py`: change class to `class ClusteringMetric(DocumentMetric, Resamplable):`, add `kind = "pooled"`, `_aggregate`, `emit_records`. Add import `from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable`. `evaluate_documents` is UNCHANGED.

```python
    kind = "pooled"

    @staticmethod
    def _aggregate(records: list, ctx: dict) -> dict[str, float]:
        pred: dict[str, str] = {}
        gold: dict[str, str] = {}
        for index, rows in enumerate(records):
            for mention_key, pred_cluster, gold_cluster in rows:
                key = f"{index}:{mention_key}"
                pred[key] = pred_cluster
                gold[key] = gold_cluster
        precision, recall, f1 = _b_cubed(pred, gold)
        return {"b3-precision": precision, "b3-recall": recall, "b3-f1": f1, "ari": _ari(pred, gold)}

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        by_mention = context.ground_truth.get("clusters_by_mention")
        if not isinstance(by_mention, dict):
            raise ValueError('clustering requires ground_truth["clusters_by_mention"]')
        gold = {str(k): str(v) for k, v in by_mention.items()}
        per_document: dict[str, list] = {}
        for delta in context.deltas:
            rows = []
            for resolution in delta.resolutions:
                start, end = resolution.mention.mention.span
                mention_key = f"{delta.document_id}:{start}-{end}"
                rows.append((mention_key, resolution.concept.id, gold[mention_key]))
            per_document[delta.document_id] = rows
        return ResampleBundle(kind="pooled", per_document=per_document, aggregate=self._aggregate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_clustering_resample.py tests/harness/test_m3_e2e.py -v`
Expected: PASS (new test + M3 e2e unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/adapters/document_metric/clustering.py tests/harness/stats/test_clustering_resample.py
git commit -m "feat: clustering opts into pooled resampling (unique re-keying)"
```

---

### Task 7: Holistic markers on the M5 intrinsic metrics

**Files:**
- Modify: `src/lattice/adapters/metric/redundancy.py`
- Modify: `src/lattice/adapters/metric/hierarchy_sanity.py`
- Modify: `src/lattice/adapters/document_metric/coherence.py`
- Test: `tests/harness/stats/test_holistic_kinds.py`

**Interfaces:**
- Consumes: `Resamplable` (Task 1).
- Produces: each of the three metrics inherits `Resamplable` and sets `kind = "holistic"`. No `emit_records` (default raises — never called for holistic). `evaluate`/`evaluate_documents` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_holistic_kinds.py
from lattice.adapters.document_metric.coherence import Coherence
from lattice.adapters.metric.hierarchy_sanity import HierarchySanity
from lattice.adapters.metric.redundancy import Redundancy
from lattice.harness.stats.records import Resamplable


def test_intrinsic_metrics_declare_holistic():
    assert Redundancy().kind == "holistic"
    assert HierarchySanity().kind == "holistic"
    # Coherence needs an embedder; a trivial stand-in is fine for the attribute check
    class _E:
        def embed(self, xs): return [(0.0,) for _ in xs]
    assert Coherence(_E()).kind == "holistic"
    assert isinstance(Redundancy(), Resamplable)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_holistic_kinds.py -v`
Expected: FAIL with `AttributeError: 'Redundancy' object has no attribute 'kind'`

- [ ] **Step 3: Write minimal implementation**

In each file, add the import `from lattice.harness.stats.records import Resamplable`, change the class base to include `Resamplable`, and add `kind = "holistic"` as the first class-body line:

- `redundancy.py`: `class Redundancy(Metric, Resamplable):` then `kind = "holistic"` (before `def __init__`).
- `hierarchy_sanity.py`: `class HierarchySanity(Metric, Resamplable):` then `kind = "holistic"` (before `def evaluate`).
- `coherence.py`: `class Coherence(DocumentMetric, Resamplable):` then `kind = "holistic"` (before `def __init__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_holistic_kinds.py tests/harness/test_m5_e2e.py -v`
Expected: PASS (new test + M5 e2e unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/adapters/metric/redundancy.py src/lattice/adapters/metric/hierarchy_sanity.py src/lattice/adapters/document_metric/coherence.py tests/harness/stats/test_holistic_kinds.py
git commit -m "feat: mark M5 intrinsic metrics holistic for resampling"
```

---

### Task 8: Resampling engine — bootstrap and bootstrap_holistic

**Files:**
- Create: `src/lattice/harness/stats/resample.py`
- Test: `tests/harness/stats/test_resample.py`

**Interfaces:**
- Consumes: `ResampleBundle` (Task 1); `run_on_documents`, `ExperimentConfig` (Task 3); `instantiate`, `Dataset`.
- Produces:
  - `jackknife(bundle: ResampleBundle, fixed_doc_ids: Sequence[str] = ()) -> dict[str, list[float]]` — leave-one-document-out (over non-fixed docs) aggregate values per metric key (for BCa acceleration).
  - `bootstrap(bundle: ResampleBundle, *, samples: int, seed: int, fixed_doc_ids: Sequence[str] = ()) -> dict[str, list[float]]` — resample non-fixed doc ids with replacement (fixed ids always included once), recompute via `bundle.aggregate`, return per-key resample distributions.
  - `bootstrap_holistic(config: ExperimentConfig, *, samples: int, seed: int, fixed_prefix: int = 0) -> dict[str, list[float]]` — resample the dataset's documents (holding the first `fixed_prefix` fixed), re-run the pipeline per resample via `run_on_documents`, return per-flattened-key distributions.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_resample.py
from lattice.harness.stats.records import ResampleBundle
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife
from lattice.harness.runner import ExperimentConfig


def _sum_bundle():
    return ResampleBundle(
        kind="pooled",
        per_document={"a": 1.0, "b": 2.0, "c": 3.0},
        aggregate=lambda records, ctx: {"total": float(sum(records))},
    )


def test_bootstrap_is_seed_deterministic_and_varies():
    b = _sum_bundle()
    assert bootstrap(b, samples=200, seed=7) == bootstrap(b, samples=200, seed=7)
    assert bootstrap(b, samples=200, seed=7) != bootstrap(b, samples=200, seed=8)


def test_bootstrap_all_fixed_is_constant_point_estimate():
    b = _sum_bundle()
    out = bootstrap(b, samples=25, seed=1, fixed_doc_ids=["a", "b", "c"])
    assert out["total"] == [6.0] * 25


def test_jackknife_leaves_one_out():
    b = _sum_bundle()
    jk = jackknife(b)
    assert sorted(jk["total"]) == [3.0, 4.0, 5.0]     # sum minus each of a,b,c


HCFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gold-mentions", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "embedding-nn", "params": {"threshold": 0.8}},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "mention-clusters", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "metrics": [{"name": "redundancy"}],
})


def test_bootstrap_holistic_runs_and_is_deterministic():
    a = bootstrap_holistic(HCFG, samples=5, seed=3)
    assert a == bootstrap_holistic(HCFG, samples=5, seed=3)     # same seed -> identical
    assert a != bootstrap_holistic(HCFG, samples=5, seed=4)     # different seed -> differs
    assert "redundancy.concept-count" in a
    assert len(a["redundancy.concept-count"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_resample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lattice.harness.stats.resample'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lattice/harness/stats/resample.py
"""The resampling engine. Item-level bootstrap drives a bundle's own
aggregate() over resampled document ids (multiplicity-preserving); holistic
bootstrap re-runs the pipeline per resample. One seeded Random throughout."""

import random
from collections import defaultdict
from collections.abc import Sequence

from lattice.config.factory import instantiate
from lattice.harness.runner import ExperimentConfig, run_on_documents
from lattice.harness.stats.records import ResampleBundle
from lattice.ports import Dataset


def _split(doc_ids: list[str], fixed_doc_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    fixed_set = set(fixed_doc_ids)
    fixed = [d for d in doc_ids if d in fixed_set]
    pool = [d for d in doc_ids if d not in fixed_set]
    return fixed, pool


def jackknife(bundle: ResampleBundle, fixed_doc_ids: Sequence[str] = ()) -> dict[str, list[float]]:
    doc_ids = list(bundle.per_document)
    fixed, pool = _split(doc_ids, fixed_doc_ids)
    out: dict[str, list[float]] = defaultdict(list)
    for i in range(len(pool)):
        kept = fixed + pool[:i] + pool[i + 1:]
        result = bundle.aggregate([bundle.per_document[d] for d in kept], bundle.global_context)
        for key, value in result.items():
            out[key].append(value)
    return dict(out)


def bootstrap(
    bundle: ResampleBundle, *, samples: int, seed: int, fixed_doc_ids: Sequence[str] = ()
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    doc_ids = list(bundle.per_document)
    fixed, pool = _split(doc_ids, fixed_doc_ids)
    n = len(pool)
    out: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = fixed + [pool[rng.randrange(n)] for _ in range(n)] if n else fixed
        result = bundle.aggregate([bundle.per_document[d] for d in drawn], bundle.global_context)
        for key, value in result.items():
            out[key].append(value)
    return dict(out)


def bootstrap_holistic(
    config: ExperimentConfig, *, samples: int, seed: int, fixed_prefix: int = 0
) -> dict[str, list[float]]:
    documents = list(instantiate(Dataset, config.dataset).documents())
    fixed = documents[:fixed_prefix]
    pool = documents[fixed_prefix:]
    n = len(pool)
    rng = random.Random(seed)
    out: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = fixed + [pool[rng.randrange(n)] for _ in range(n)] if n else fixed
        for key, value in run_on_documents(config, drawn).items():
            out[key].append(value)
    return dict(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_resample.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/stats/resample.py tests/harness/stats/test_resample.py
git commit -m "feat: bootstrap engine (item-level + holistic) with jackknife"
```

---

### Task 9: Order-permutation robustness — order_spread

**Files:**
- Create: `src/lattice/harness/stats/permutation.py`
- Test: `tests/harness/stats/test_permutation.py`

**Interfaces:**
- Consumes: `run_on_documents`, `ExperimentConfig` (Task 3); `instantiate`, `Dataset`.
- Produces:
  - `SpreadResult(values: list[float], min: float, max: float, range: float, std: float)` — frozen dataclass.
  - `order_spread(config: ExperimentConfig, *, permutations: int, seed: int, fixed_prefix: int = 0) -> dict[str, SpreadResult]` — run the pipeline on K seeded document orderings (holding the first `fixed_prefix` fixed), report per-flattened-key spread.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_permutation.py
from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.permutation import SpreadResult, order_spread

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gold-mentions", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "mention-clusters", "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"}},
    "metrics": [{"name": "redundancy"}, {"name": "hierarchy-sanity"}],
})


def test_order_spread_reports_stats_and_is_deterministic():
    a = order_spread(CFG, permutations=6, seed=2)
    b = order_spread(CFG, permutations=6, seed=2)
    assert a == b
    key = "redundancy.concept-count"
    assert isinstance(a[key], SpreadResult)
    assert a[key].range == a[key].max - a[key].min
    assert a[key].min <= a[key].max
    # exact-label graph accretes commutatively -> concept count is order-invariant
    assert a[key].range == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_permutation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lattice.harness.stats.permutation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lattice/harness/stats/permutation.py
"""Order-permutation robustness. Runs the pipeline over K seeded document
orderings and reports the spread of each metric. A robustness report, not a
hypothesis test; for the glossary-first M4 protocol use fixed_prefix=1 so the
term-inventory document stays first and only the corpus order varies."""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pstdev

from lattice.config.factory import instantiate
from lattice.harness.runner import ExperimentConfig, run_on_documents
from lattice.ports import Dataset


@dataclass(frozen=True)
class SpreadResult:
    values: list[float]
    min: float
    max: float
    range: float
    std: float


def order_spread(
    config: ExperimentConfig, *, permutations: int, seed: int, fixed_prefix: int = 0
) -> dict[str, SpreadResult]:
    documents = list(instantiate(Dataset, config.dataset).documents())
    fixed = documents[:fixed_prefix]
    pool = documents[fixed_prefix:]
    rng = random.Random(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(permutations):
        order = pool[:]
        rng.shuffle(order)
        for key, value in run_on_documents(config, fixed + order).items():
            collected[key].append(value)
    return {
        key: SpreadResult(
            values=values, min=min(values), max=max(values),
            range=max(values) - min(values), std=pstdev(values),
        )
        for key, values in collected.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_permutation.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/stats/permutation.py tests/harness/stats/test_permutation.py
git commit -m "feat: order-permutation robustness report"
```

---

### Task 10: Report assembly + CLI

**Files:**
- Create: `src/lattice/harness/stats/report.py`
- Create: `src/lattice/harness/stats/__main__.py`
- Test: `tests/harness/stats/test_report_cli.py`

**Interfaces:**
- Consumes: everything above; `load_config`, `ExperimentConfig`, `SweepConfig`/`expand`.
- Produces:
  - `analyze(config, *, samples, seed, level, thresholds, permutations, paired, fixed_prefix) -> dict` — build the interval-report dict: for each item-level bundle, `run_experiment_detailed` once → `bootstrap` + `jackknife` → per-key `{estimate, bca, percentile}`; for holistic metrics, `bootstrap_holistic`; optional `paired` (two resolver configs) → `paired_delta`; optional `thresholds` → per-threshold point + CI; optional `permutations` → `order_spread`. Returns a JSON-serializable dict with `seed`, `level`, `samples`.
  - `write_report(report: dict, out_dir: str | Path) -> Path` — writes `<out_dir>/interval-report.json` (`json.dumps(..., indent=2, sort_keys=True)`).
  - `__main__.py`: `python -m lattice.harness.stats <config.toml> <out_dir> [--samples N] [--seed S] [--level L] [--holistic] [--permutations K] [--fixed-prefix P]`. `--holistic` routes to `bootstrap_holistic` (M5); default is item-level. Parse with `argparse`.

The full `analyze` composition is exercised end-to-end by Task 11; this task's tests cover the report dict shape and JSON writing on a toy config.

- [ ] **Step 1: Write the failing test**

```python
# tests/harness/stats/test_report_cli.py
import json

from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.report import analyze, write_report

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {"name": "gazetteer", "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"}},
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {"name": "union", "params": {"members": [{"name": "hearst"}, {"name": "compound"}]}},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {"name": "taxonomy", "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"}},
    "metrics": [{"name": "edge-f1"}],
})


def test_analyze_item_level_shape_and_centering():
    report = analyze(CFG, samples=300, seed=5, level=0.95)
    assert report["seed"] == 5 and report["level"] == 0.95 and report["samples"] == 300
    f1 = report["metrics"]["edge-f1"]["f1"]
    # the interval is centered on the point estimate, and the estimate is the real F1 (1.0 on toy)
    assert f1["estimate"] == 1.0
    assert set(f1) >= {"estimate", "bca", "percentile"}
    assert set(f1["bca"]) == {"lo", "hi", "method"}


def test_write_report_is_json_and_sorted(tmp_path):
    report = analyze(CFG, samples=50, seed=1, level=0.95)
    path = write_report(report, tmp_path)
    assert path.name == "interval-report.json"
    loaded = json.loads(path.read_text())
    assert loaded["metrics"]["edge-f1"]["f1"]["estimate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_report_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lattice.harness.stats.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lattice/harness/stats/report.py
"""Assemble the interval report from bundles + engine + intervals, and write
it as regenerable JSON. Item-level metrics bootstrap from one pipeline run;
holistic metrics re-run the pipeline per resample."""

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import Interval, bca_interval, percentile_interval
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife


def _iv(estimate: float, resamples: list[float], jack: list[float], level: float) -> dict:
    bca = bca_interval(estimate, resamples, jack, level)
    pct = percentile_interval(estimate, resamples, level)
    return {
        "estimate": estimate,
        "bca": {"lo": bca.lo, "hi": bca.hi, "method": bca.method},
        "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
    }


def analyze(
    config: ExperimentConfig, *, samples: int, seed: int, level: float = 0.95,
    holistic: bool = False, fixed_prefix: int = 0, fixed_doc_ids: Sequence[str] = (),
) -> dict:
    metrics: dict[str, dict] = {}
    if holistic:
        dists = bootstrap_holistic(config, samples=samples, seed=seed, fixed_prefix=fixed_prefix)
        # holistic point estimates: one clean run over the full stream
        from lattice.harness.runner import run_on_documents
        from lattice.config.factory import instantiate
        from lattice.ports import Dataset
        documents = list(instantiate(Dataset, config.dataset).documents())
        estimates = run_on_documents(config, documents)
        for flat_key, resamples in dists.items():
            metric, key = flat_key.split(".", 1)
            # holistic BCa acceleration would need pipeline jackknife; use percentile
            pct = percentile_interval(estimates[flat_key], resamples, level)
            metrics.setdefault(metric, {})[key] = {
                "estimate": estimates[flat_key],
                "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
            }
    else:
        report_full, bundles = run_experiment_detailed(config)
        for name, bundle in bundles.items():
            dists = bootstrap(bundle, samples=samples, seed=seed, fixed_doc_ids=fixed_doc_ids)
            jacks = jackknife(bundle, fixed_doc_ids=fixed_doc_ids)
            for key, resamples in dists.items():
                estimate = report_full.metrics[name][key]
                metrics.setdefault(name, {})[key] = _iv(estimate, resamples, jacks[key], level)
    return {"seed": seed, "level": level, "samples": samples, "config": config.model_dump(), "metrics": metrics}


def write_report(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "interval-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path
```

```python
# src/lattice/harness/stats/__main__.py
"""CLI: python -m lattice.harness.stats <config.toml> <out_dir> [flags]."""

import argparse

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.report import analyze, write_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m lattice.harness.stats")
    parser.add_argument("config")
    parser.add_argument("out_dir")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--holistic", action="store_true")
    parser.add_argument("--fixed-prefix", type=int, default=0)
    args = parser.parse_args()
    samples = args.samples if args.samples is not None else (1000 if args.holistic else 10000)
    config = load_config(args.config, model=ExperimentConfig)
    report = analyze(
        config, samples=samples, seed=args.seed, level=args.level,
        holistic=args.holistic, fixed_prefix=args.fixed_prefix,
    )
    path = write_report(report, args.out_dir)
    print(f"interval report: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/stats/test_report_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lattice/harness/stats/report.py src/lattice/harness/stats/__main__.py tests/harness/stats/test_report_cli.py
git commit -m "feat: interval report assembly + CLI"
```

---

### Task 11: Run the full matrix and write the committed results doc

**Files:**
- Create: `docs/results/2026-07-14-interval-analysis.md` (COMMITTED)
- (Regenerable, gitignored) `reports/intervals/**/interval-report.json`

**Interfaces:**
- Consumes: the CLI (Task 10), the existing sweep configs under `configs/`, the datasets under `data/`, and (for M2/M3/M5 standard runs) the ml venv + cached models. This task RUNS the analysis on real corpora — it needs the same environment as the original sweeps (see the venv rebuild recipe in memory if the environment is broken).

This task has no unit test; its deliverable is the committed numbers doc. Each number is produced by a recorded command so it is regenerable. Preconditions: `data/conel2`, `data/ecbplus`, `data/inspec`, and the M4 TExEval corpora present; `scripts/fetch_models.py` run if a config uses `sentence-transformer`/`noun-chunk`.

- [ ] **Step 1: Generate item-level interval reports (M2, M3, M4)**

For each single-config run needed, materialize a one-config TOML from the sweep (or point the CLI at a config whose axes are collapsed to the target adapter) and run:

```bash
mkdir -p reports/intervals
# M3 exact-label and nn@0.90 (item-level, B=10000)
python -m lattice.harness.stats configs/m3-conel2-exact.toml   reports/intervals/m3-conel2-exact  --seed 0
python -m lattice.harness.stats configs/m3-conel2-nn090.toml   reports/intervals/m3-conel2-nn090  --seed 0
# M2 Inspec f1@k
python -m lattice.harness.stats configs/m2b-f1atk.toml         reports/intervals/m2b              --seed 0
# M4: one per gold, glossary held fixed via --fixed-prefix 1
for g in env-eurovoc food food-wordnet science science-eurovoc science-wordnet; do
  python -m lattice.harness.stats configs/m4-$g-union.toml reports/intervals/m4-$g --seed 0 --fixed-prefix 1
done
```

Note: create the collapsed single-config TOMLs (e.g. `configs/m3-conel2-nn090.toml`) by copying the sweep `base` and setting the one target adapter; these are config files, committed like the existing `configs/*.toml`. The `--fixed-prefix 1` holds the M4 glossary document (stream position 0) fixed while resampling the corpus.

- [ ] **Step 2: Generate the M3 paired delta and threshold curve**

Create a committed `scripts/interval_analysis.py` that composes the tested library primitives (`run_experiment_detailed`, `bootstrap`, `jackknife`, `paired_delta`, `percentile_interval`, `order_spread`) — no new algorithms, only orchestration. For the paired delta: run `bootstrap` on the M3 clustering bundle for the exact-label config and the nn@0.90 config with the SAME seed (identical document draws → paired), then `paired_delta(nn090_resamples["b3-f1"], exact_resamples["b3-f1"], est_nn090, est_exact)`. Record estimate, 95% CI, `prob_positive`. For the threshold curve: loop the grid (0.65/0.70/0.75/0.80/0.85/0.90/0.95), building a config per threshold, and tabulate `b3-f1` estimate + BCa CI, marking 0.90.

- [ ] **Step 3: Generate holistic (M5) and permutation reports**

```bash
# M5 intrinsic, holistic, B=1000
python -m lattice.harness.stats configs/m5-conel2-nn090.toml reports/intervals/m5-conel2 --holistic --seed 0
```

Order-permutation stability (K=40) is produced by `scripts/interval_analysis.py` calling `order_spread` on the M3 and M5 configs (`fixed_prefix=0`) and on the M4 union configs (`fixed_prefix=1`, so the glossary document stays first — expect ~0 spread, validating the glossary-first design).

- [ ] **Step 4: Write the committed results doc**

Create `docs/results/2026-07-14-interval-analysis.md` containing: the headline M3 paired-delta verdict (nn@0.90 − exact-label b3-f1, CI, `prob_positive`, and the honest interpretation of whether "beats" survives); the per-milestone CI tables (M2 f1@k, M3 b3-f1 per config, M4 P/R/F1 per gold, M5 intrinsic); the threshold-sensitivity table with 0.90 marked and the note that 0.90 is not b3-f1-optimal (chosen for M5's tradeoff); the order-permutation spreads (M4 ~0 validates glossary-first); the explicit statement of the edge-set-pooled bootstrap semantics (multiplicity collapses by construction → the CI reflects corpus-composition variance); and the exact commands that regenerate every number.

- [ ] **Step 5: Commit**

```bash
git add docs/results/2026-07-14-interval-analysis.md configs/m3-conel2-*.toml configs/m2b-f1atk.toml configs/m4-*-union.toml configs/m5-conel2-nn090.toml scripts/  # only newly-created config/script files
git commit -m "docs: interval analysis results across M2-M5"
```

---

## Self-Review

**Spec coverage:**
- §3 protocol changes → Tasks 1, 4–7. §4.1 kernels → realized as per-metric `_aggregate` driven by the generic `bootstrap` (Task 8) — a faithful refinement of "one recompute per kind" (each metric owns its recompute; the engine is generic). Documented in Architecture. §4.2 intervals → Task 2. §4.3 permutation → Task 9. §4.4 `run_experiment_detailed` → Task 3. §4.5 CLI → Task 10 covers the single-config item-level and `--holistic` runs (plus `--fixed-prefix`); the `--paired`/`--thresholds`/`--permutations` compositions from §4.5 are realized in the committed `scripts/interval_analysis.py` (Task 11) from the same tested primitives, rather than as argparse flags — a faithful refinement, since paired/threshold/permutation are orchestrations of `bootstrap`/`paired_delta`/`order_spread`, all unit-tested in Tasks 2/8/9. §5 params → Global Constraints + Task 10 defaults. §6 deliverables → Task 11. §7 testing → each task's tests + equivalence in 4/5/6. §8 output → Task 10 (JSON) + Task 11 (committed doc). §9 success criteria → equivalence (4/5/6), reproducible (2/8/9/10), headline (11), full matrix (11), no regression (every task re-runs the neighboring e2e suite).
- Divergence from spec letter, flagged: the "three kernels" live inside each metric's `_aggregate` rather than in the engine, because pooled edge-F1 and pooled B³ need different recompute formulas — a single engine-side pooled kernel would couple the engine to metric internals. Behavior, interfaces, and equivalence are unchanged.

**Placeholder scan:** No TBD/TODO. Task 11 is an analysis/execution task (like the M4 fetch grind) with no unit test by nature; its steps carry exact commands. All code steps contain complete code.

**Type consistency:** `EvaluationContext`, `ResampleBundle(kind, per_document, aggregate, global_context)`, `Interval(lo,hi,method)`, `DeltaResult`, `SpreadResult`, `bootstrap`/`jackknife`/`bootstrap_holistic`/`order_spread` signatures are consistent across Tasks 1–10. `_aggregate(records, ctx)` signature is uniform (macro/clustering ignore `ctx`).

## Execution Amendments

Authored by the observer during execution; each resolves a defect found in the
plan's own text (not an implementer deviation).

1. **Task 2 percentile literal (typo fix).** The Step-1 test asserted the 97.5th
   percentile of `range(100)` is `97.525`; the correct linear-interpolation value is
   `96.525` (position `0.975*99 = 96.525` → `96 + 0.525`). The lower bound `2.475` was
   correct. The implementer corrected the literal; the reviewer re-derived it against
   numpy. No algorithm changed.
2. **ruff cleanliness.** The plan's literal code blocks (long single-line fixture-config
   dicts, some unsorted imports) are not ruff-clean under this repo's `line-length = 100`,
   `select = ["E","F","I","UP"]`. A cross-task formatting-only fix (`c7a32e2`) cleaned
   Tasks 1–3 with no logic change; ruff-check-before-commit is now explicit in every
   dispatch from Task 4 on. Implementers should format to the repo's ruff config even
   where the plan's literal text does not.
3. **Duplicate-metric-name guard (Task 3).** `run_on_documents` and
   `run_experiment_detailed` now carry the same guard `run_experiment` has — raise
   `ValueError(f"metric name(s) used by both families: {sorted(duplicates)}")` when a
   name appears in both `config.metrics` and `config.document_metrics`. The plan's
   original Task 3 code omitted it (a silent divergence flagged Important by the Task 3
   reviewer); the guard is added above, plus a regression test per function. The
   `test_detailed_returns_bundles_for_resamplable_only` test also switches from
   `model_copy(update=...).model_dump()` (which emits a harmless
   `PydanticSerializationUnexpectedValue` warning) to mutating `M5.model_dump()` before
   `model_validate`.
4. **Task 6 multiplicity test was a tautology.** The plan's original
   `test_clustering_equivalence_and_multiplicity` asserted `set(dup) == set(report.metrics
   ["clustering"])` — but `_aggregate` always returns the same four keys, so that assertion
   passes even if the index-prefixing (multiplicity) logic is missing or broken. Split into
   `test_clustering_equivalence` (the real equivalence check, kept) and
   `test_clustering_aggregate_respects_multiplicity`, a hand-built case where duplicating a
   document changes b3-precision from 2/3 to 3/4 — values verified against the real
   `_aggregate`; the test fails iff the unique re-keying is removed. Also documents the
   emit_records precondition (amendment 5).
5. **emit_records precondition (guard parity).** `emit_records` on `F1AtK`, `EdgeF1`, and
   `ClusteringMetric` deliberately does NOT re-validate inputs that `evaluate`/
   `evaluate_documents` already check — its sole caller `run_experiment_detailed` runs
   evaluate first on the same inputs, so duplicating those guards would be unreachable dead
   code plus verbatim-duplicated validation. Each `emit_records` carries a one-line
   precondition docstring stating this. (Contrast the Task 3 duplicate-name guard, amendment
   3, which was genuinely reachable and silently corrupting.)
6. **Task 8 holistic-bootstrap test half-covered the property.**
   `test_bootstrap_holistic_runs_and_is_deterministic` asserted same-seed determinism but
   not that a different seed differs (the sibling `bootstrap` test checks both). Added
   `assert a != bootstrap_holistic(HCFG, samples=5, seed=4)`; verified against the real
   implementation (seed 3 → concept-count [2,2,2,2,2], seed 4 → [1,2,1,2,1]).
