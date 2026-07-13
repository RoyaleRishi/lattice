# M5 Integration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Intrinsic graph-quality metrics — `redundancy`, `coherence`, `hierarchy-sanity` — plus the first all-real pipeline sweep over ConEL-2 transcripts.

**Architecture:** Per `docs/2026-07-12-m5-integration-design.md` (zero core changes): three metric adapters behind the existing `Metric`/`DocumentMetric` ports, one runner change so metric instantiation gets shared-dep injection (only `coherence` consumes the embedder), one sweep config over the resolver axis. No new dataset work — ConEL-2 is on disk from M3.

**Tech Stack:** Python 3.13 (uv), stdlib-only metrics (`lattice.core.vectors.cosine`, iterative Tarjan), pytest, existing ml group for the real run only.

## Global Constraints

- `pyproject.toml` is FROZEN — no new dependencies.
- No network access, no model loads in tests or adapters (`@pytest.mark.ml` tests use `importorskip` + OSError-skip; everything else runs on `hashing`/`token`).
- `data/`, `reports/`, `.superpowers/` are gitignored — never commit them.
- Registered names are load-bearing: `redundancy`, `coherence`, `hierarchy-sanity`.
- Every new adapter module gets an import line in `src/lattice/adapters/__init__.py`.
- Metric return values are ALL floats (Metric convention); metric keys use hyphens (`duplicate-rate`), matching `clustering`'s style.
- Run tests with `uv run --no-sync pytest <path> -q`; if imports fail first run `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null`.
- `uv run --no-sync ruff check .` before every commit; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Sweeps: `uv run --no-sync python -m lattice.harness --sweep <toml> <out_dir>`.

## File Structure

```
src/lattice/harness/runner.py                       Task 1 (modify: metric shared-dep injection)
src/lattice/adapters/metric/redundancy.py           Task 2  "redundancy"
src/lattice/adapters/metric/hierarchy_sanity.py     Task 3  "hierarchy-sanity"
src/lattice/adapters/document_metric/coherence.py   Task 4  "coherence"
configs/m5-conel2-sweep.toml                        Task 5
tests/harness/test_metric_injection.py              Task 1
tests/adapters/test_redundancy_metric.py            Task 2
tests/adapters/test_hierarchy_sanity_metric.py      Task 3
tests/adapters/test_coherence_metric.py             Task 4
tests/harness/test_m5_e2e.py                        Task 5
```

Task 6 is the exit-criteria run (ml sweep + spec §8 adjudication); no committed code.

**Suggested implementers:** Task 2, 4 — haiku (complete code below). Tasks 1, 3, 5 — sonnet (harness surgery / Tarjan intricacy / e2e). Task 6 — orchestrator. All reviewers sonnet. Verified before plan commit: the Tarjan/longest-path/shortcut code below was executed against every planted test case (2-cycle, scc+tail, diamond, 2000-deep chain), and the hashing-embedder cosine bounds in Task 4's tests are measured (0.8386 / 0.1336).

---

### Task 1: Runner shared-dep injection for metrics

**Files:**
- Modify: `src/lattice/harness/runner.py`
- Test: `tests/harness/test_metric_injection.py`

**Interfaces:**
- Consumes: `instantiate(port, spec, shared)` from `lattice.config.factory` (already injects only when the constructor names the param), `Embedder` port.
- Produces: `run_experiment` instantiates every `Metric` and `DocumentMetric` with `shared = {"embedder": <instance built from config.embedder>}`. Task 4's `coherence` relies on this. Existing metrics (no `embedder` param) are unaffected.

- [ ] **Step 1: Write the failing test**

Create `tests/harness/test_metric_injection.py`:

```python
from lattice.core.types import GraphSnapshot
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.ports import Embedder, Metric
from lattice.registry.registry import register


@register(Metric, "test-embedder-probe")
class EmbedderProbe(Metric):
    """Test-only metric: proves the runner injects the shared embedder."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        [vector] = self.embedder.embed(["probe"])
        return {"embedder-dim": float(len(vector))}


def _config(metrics: list[dict]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token"},
            "scorer": {"name": "frequency"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing", "params": {"dim": 16}},
            "dataset": {"name": "toy"},
            "metrics": metrics,
        }
    )


def test_metric_with_embedder_param_receives_the_configured_embedder():
    report = run_experiment(_config([{"name": "test-embedder-probe"}]))
    assert report.errors == ()
    # dim=16 proves the injected instance was built from config.embedder,
    # not a default.
    assert report.metrics["test-embedder-probe"]["embedder-dim"] == 16.0


def test_metric_without_embedder_param_is_unaffected():
    report = run_experiment(_config([{"name": "label-f1"}]))
    assert set(report.metrics["label-f1"]) == {"precision", "recall", "f1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_metric_injection.py -q`
Expected: FAIL — `TypeError: EmbedderProbe.__init__() missing 1 required positional argument: 'embedder'` (the runner currently instantiates metrics without shared deps).

- [ ] **Step 3: Modify the runner**

In `src/lattice/harness/runner.py`, change the ports import line and `run_experiment`:

```python
from lattice.ports import Dataset, DocumentMetric, Embedder, Metric
```

and inside `run_experiment`, replace the two metric-instantiation dict comprehensions with:

```python
    # Intrinsic metrics may consume the embedder (M5 spec §3): a second
    # instance from the same spec is deterministic, so pipeline and metric
    # embeddings agree; instantiate() injects only where the constructor
    # names the param.
    metric_shared: dict[str, object] = {
        "embedder": instantiate(Embedder, config.embedder)
    }
    metric_results = {
        spec.name: instantiate(Metric, spec, metric_shared).evaluate(
            snapshot, ground_truth
        )
        for spec in config.metrics
    }
    document_results = {
        spec.name: instantiate(DocumentMetric, spec, metric_shared).evaluate_documents(
            deltas, ground_truth
        )
        for spec in config.document_metrics
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/harness/ -q`
Expected: PASS (new tests plus all existing harness tests — the runner change must not break m2a/m2b/m3/m4 e2e).

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/harness/runner.py tests/harness/test_metric_injection.py
git commit -m "feat: inject shared embedder into metric instantiation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `redundancy` Metric

**Files:**
- Create: `src/lattice/adapters/metric/redundancy.py`
- Modify: `src/lattice/adapters/__init__.py` (metric import line)
- Test: `tests/adapters/test_redundancy_metric.py`

**Interfaces:**
- Consumes: `Metric` port, `GraphSnapshot`, `lattice.core.vectors.cosine`, `MetricContract`.
- Produces: registered `("redundancy", Metric)`, constructor `Redundancy(threshold: float = 0.9)`, keys `duplicate-rate`, `near-duplicate-pairs`, `concept-count`. Module-level `_normalize(label: str) -> str` (unit-tested directly). Task 5's config references `name = "redundancy"`.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_redundancy_metric.py`:

```python
from lattice.adapters.metric.redundancy import Redundancy, _normalize
from lattice.core.types import Concept, GraphSnapshot
from tests.contracts.metric_contract import MetricContract


class TestRedundancyContract(MetricContract):
    def make_metric(self):
        return Redundancy()

    def make_ground_truth(self):
        return {}


def _concept(cid: str, label: str, embedding: tuple[float, ...]) -> Concept:
    return Concept(
        id=cid, label=label, embedding=embedding, first_seen="d1", updated_at="d1"
    )


def _snapshot(*concepts: Concept) -> GraphSnapshot:
    return GraphSnapshot(concepts=tuple(concepts), relations=())


def test_normalize_rules():
    assert _normalize("The Beatles") == "beatle"
    assert _normalize("beatles") == "beatle"
    assert _normalize("an apple") == "apple"
    assert _normalize("glass") == "glass"  # 'ss' guard: no plural strip
    assert _normalize("gas") == "gas"  # too short to strip
    assert _normalize("glas") == "gla"


def test_embedding_near_duplicates_counted():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "alpha", (1.0, 0.0)),
            _concept("c2", "beta", (1.0, 0.0)),
            _concept("c3", "gamma", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 1.0
    assert result["duplicate-rate"] == 2.0 / 3.0
    assert result["concept-count"] == 3.0


def test_label_collision_counts_even_with_orthogonal_embeddings():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "the beatles", (1.0, 0.0)),
            _concept("c2", "beatles", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 1.0
    assert result["duplicate-rate"] == 1.0


def test_ss_guard_prevents_false_plural_collision():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "glass", (1.0, 0.0)),
            _concept("c2", "glas", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 0.0
    assert result["duplicate-rate"] == 0.0


def test_threshold_is_respected():
    # cosine of these is ~0.9487: above 0.9, below 0.99
    a = (3.0, 1.0)
    b = (1.0, 0.0)
    snapshot = _snapshot(_concept("c1", "x", a), _concept("c2", "y", b))
    assert Redundancy(threshold=0.9).evaluate(snapshot, {})["near-duplicate-pairs"] == 1.0
    assert Redundancy(threshold=0.99).evaluate(snapshot, {})["near-duplicate-pairs"] == 0.0


def test_zero_vectors_never_match_by_embedding():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "x", (0.0, 0.0)),
            _concept("c2", "y", (0.0, 0.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 0.0


def test_empty_snapshot_is_all_zeros():
    assert Redundancy().evaluate(_snapshot(), {}) == {
        "duplicate-rate": 0.0,
        "near-duplicate-pairs": 0.0,
        "concept-count": 0.0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_redundancy_metric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.metric.redundancy'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/metric/redundancy.py`:

```python
import re

from lattice.core.types import GraphSnapshot
from lattice.core.vectors import cosine
from lattice.ports import Metric
from lattice.registry.registry import register

_ARTICLE = re.compile(r"^(?:a|an|the)\s+")


def _normalize(label: str) -> str:
    """casefold -> strip one leading article -> strip one trailing 's'
    when the result stays >= 3 chars and the label doesn't end in 'ss'
    (M5 spec §4.1: "beatles"->"beatle", "glass"->"glass")."""
    norm = _ARTICLE.sub("", label.casefold().strip())
    if len(norm) > 3 and norm.endswith("s") and not norm.endswith("ss"):
        norm = norm[:-1]
    return norm


@register(Metric, "redundancy")
class Redundancy(Metric):
    """Intrinsic near-duplicate detection over the accreted graph (M5 spec
    §4.1): what the resolver failed to merge. Two concepts are
    near-duplicates when their stored embeddings' cosine >= threshold or
    their normalized labels collide. O(n²) pairwise scan — fine at this
    scale (top-k selection bounds concepts to the low thousands)."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        concepts = snapshot.concepts
        count = len(concepts)
        norms = [_normalize(concept.label) for concept in concepts]
        pairs = 0
        has_duplicate = [False] * count
        for i in range(count):
            for j in range(i + 1, count):
                near = (
                    norms[i] == norms[j]
                    or cosine(concepts[i].embedding, concepts[j].embedding)
                    >= self.threshold
                )
                if near:
                    pairs += 1
                    has_duplicate[i] = has_duplicate[j] = True
        return {
            "duplicate-rate": (sum(has_duplicate) / count) if count else 0.0,
            "near-duplicate-pairs": float(pairs),
            "concept-count": float(count),
        }
```

In `src/lattice/adapters/__init__.py`, change the metric import line to:

```python
from lattice.adapters.metric import edge_f1, label_f1, redundancy  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_redundancy_metric.py -q`
Expected: PASS (10 tests: 2 contract + 8 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/metric/redundancy.py src/lattice/adapters/__init__.py tests/adapters/test_redundancy_metric.py
git commit -m "feat: add intrinsic redundancy metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `hierarchy-sanity` Metric

**Files:**
- Create: `src/lattice/adapters/metric/hierarchy_sanity.py`
- Modify: `src/lattice/adapters/__init__.py` (metric import line)
- Test: `tests/adapters/test_hierarchy_sanity_metric.py`

**Interfaces:**
- Consumes: `Metric` port, `GraphSnapshot`, `Relation`, `MetricContract`, `tests.helpers.make_concept`.
- Produces: registered `("hierarchy-sanity", Metric)`, no-arg constructor `HierarchySanity()`, keys `cycle-components`, `cycle-nodes`, `self-loops`, `max-depth`, `transitive-shortcuts`, `is-a-edges`. Task 5's config references `name = "hierarchy-sanity"`.

All graph code below was executed against these exact planted cases before this plan was committed (including a 2000-node chain for stack safety).

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_hierarchy_sanity_metric.py`:

```python
from lattice.adapters.metric.hierarchy_sanity import HierarchySanity
from lattice.core.types import GraphSnapshot, Relation
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


class TestHierarchySanityContract(MetricContract):
    def make_metric(self):
        return HierarchySanity()

    def make_ground_truth(self):
        return {}


def _edge(source: str, target: str, type: str = "IS_A") -> Relation:
    return Relation(
        type=type, source_id=source, target_id=target, confidence=1.0,
        provenance="d1",
    )


def _snapshot(*relations: Relation) -> GraphSnapshot:
    node_ids = sorted({r.source_id for r in relations} | {r.target_id for r in relations})
    return GraphSnapshot(
        concepts=tuple(make_concept(id=n, label=n) for n in node_ids),
        relations=tuple(relations),
    )


def _evaluate(*relations: Relation) -> dict[str, float]:
    return HierarchySanity().evaluate(_snapshot(*relations), {})


def test_empty_snapshot_is_all_zeros():
    result = HierarchySanity().evaluate(GraphSnapshot(concepts=(), relations=()), {})
    assert result == {
        "cycle-components": 0.0,
        "cycle-nodes": 0.0,
        "self-loops": 0.0,
        "max-depth": 0.0,
        "transitive-shortcuts": 0.0,
        "is-a-edges": 0.0,
    }


def test_two_cycle_detected():
    result = _evaluate(_edge("a", "b"), _edge("b", "a"))
    assert result["cycle-components"] == 1.0
    assert result["cycle-nodes"] == 2.0
    assert result["self-loops"] == 0.0


def test_self_loop_counted_separately_not_as_cycle_component():
    result = _evaluate(_edge("a", "a"), _edge("a", "b"))
    assert result["self-loops"] == 1.0
    assert result["cycle-components"] == 0.0
    assert result["max-depth"] == 1.0


def test_chain_depth_counts_edges():
    result = _evaluate(_edge("a", "b"), _edge("b", "c"))
    assert result["max-depth"] == 2.0
    assert result["transitive-shortcuts"] == 0.0


def test_triangle_shortcut_detected():
    result = _evaluate(_edge("a", "b"), _edge("b", "c"), _edge("a", "c"))
    assert result["transitive-shortcuts"] == 1.0


def test_diamond_has_exactly_one_shortcut():
    result = _evaluate(
        _edge("a", "b"), _edge("a", "c"), _edge("a", "d"),
        _edge("b", "d"), _edge("c", "d"),
    )
    assert result["transitive-shortcuts"] == 1.0  # only a->d


def test_edge_out_of_a_cycle_is_not_a_shortcut():
    # b->c's only alternative "path" re-uses the b->c edge itself via the
    # a<->b cycle; the mid-walk edge guard must reject it.
    result = _evaluate(_edge("a", "b"), _edge("b", "a"), _edge("b", "c"))
    assert result["transitive-shortcuts"] == 0.0


def test_depth_excludes_cycle_nodes():
    # a<->b is a cycle; the acyclic remainder is c->d (depth 1)
    result = _evaluate(
        _edge("a", "b"), _edge("b", "a"), _edge("b", "c"), _edge("c", "d")
    )
    assert result["cycle-nodes"] == 2.0
    assert result["max-depth"] == 1.0


def test_non_is_a_relations_are_ignored():
    result = _evaluate(
        _edge("a", "b", type="CO_OCCURS"), _edge("b", "a", type="CO_OCCURS")
    )
    assert result["is-a-edges"] == 0.0
    assert result["cycle-components"] == 0.0


def test_deep_chain_does_not_blow_the_stack():
    edges = [_edge(str(i), str(i + 1)) for i in range(2000)]
    result = _evaluate(*edges)
    assert result["max-depth"] == 2000.0
    assert result["cycle-components"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_hierarchy_sanity_metric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.metric.hierarchy_sanity'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/metric/hierarchy_sanity.py`:

```python
from lattice.core.types import GraphSnapshot
from lattice.ports import Metric
from lattice.registry.registry import register


def _tarjan_sccs(
    nodes: list[str], adjacency: dict[str, list[str]]
) -> list[list[str]]:
    """Iterative Tarjan (recursion-free: real IS_A chains can be deep)."""
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index_of:
            continue
        work = [(root, iter(adjacency.get(root, ())))]
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, neighbours = work[-1]
            advanced = False
            for neighbour in neighbours:
                if neighbour not in index_of:
                    index_of[neighbour] = lowlink[neighbour] = counter
                    counter += 1
                    stack.append(neighbour)
                    on_stack.add(neighbour)
                    work.append((neighbour, iter(adjacency.get(neighbour, ()))))
                    advanced = True
                    break
                if neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[neighbour])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                sccs.append(component)
    return sccs


def _longest_path(allowed: set[str], adjacency: dict[str, list[str]]) -> int:
    """Longest path in edges over an acyclic subgraph, iterative post-order."""
    depth: dict[str, int] = {}
    for root in sorted(allowed):
        if root in depth:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if not expanded and node in depth:
                continue
            children = [c for c in adjacency.get(node, ()) if c in allowed]
            if expanded:
                depth[node] = 1 + max((depth[c] for c in children), default=-1)
            else:
                stack.append((node, True))
                stack.extend((c, False) for c in children if c not in depth)
    return max(depth.values(), default=0)


def _is_shortcut(
    source: str, target: str, adjacency: dict[str, list[str]]
) -> bool:
    """True when target is reachable from source without the direct edge —
    i.e. the edge duplicates a >= 2-step path. The direct edge must be
    excluded everywhere in the walk, not only at the first step: a cycle
    can revisit `source` mid-path and would otherwise re-offer the very
    edge under test (a<->b with b->c must NOT make b->c a shortcut)."""
    stack = [n for n in adjacency.get(source, ()) if n != target]
    seen = set(stack)
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for child in adjacency.get(node, ()):
            if node == source and child == target:
                continue
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return False


@register(Metric, "hierarchy-sanity")
class HierarchySanity(Metric):
    """Structural sanity of the induced IS_A hierarchy (M5 spec §4.3), in
    the spirit of TExEval-2's structural analysis: cycles, self-loops,
    depth, transitive shortcuts. No gold needed; all stdlib."""

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        edges = [
            (r.source_id, r.target_id)
            for r in snapshot.relations
            if r.type == "IS_A"
        ]
        self_loops = sum(1 for a, b in edges if a == b)
        proper = [(a, b) for a, b in edges if a != b]
        nodes = sorted({n for edge in proper for n in edge})
        adjacency: dict[str, list[str]] = {}
        for a, b in proper:
            adjacency.setdefault(a, []).append(b)
        cycle_components = [
            c for c in _tarjan_sccs(nodes, adjacency) if len(c) >= 2
        ]
        cycle_nodes = {n for component in cycle_components for n in component}
        allowed = {n for n in nodes if n not in cycle_nodes}
        acyclic = {
            node: [c for c in children if c in allowed]
            for node, children in adjacency.items()
            if node in allowed
        }
        shortcuts = sum(1 for a, b in proper if _is_shortcut(a, b, adjacency))
        return {
            "cycle-components": float(len(cycle_components)),
            "cycle-nodes": float(len(cycle_nodes)),
            "self-loops": float(self_loops),
            "max-depth": float(_longest_path(allowed, acyclic)),
            "transitive-shortcuts": float(shortcuts),
            "is-a-edges": float(len(edges)),
        }
```

In `src/lattice/adapters/__init__.py`, change the metric import line to:

```python
from lattice.adapters.metric import (  # noqa: F401
    edge_f1,
    hierarchy_sanity,
    label_f1,
    redundancy,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_hierarchy_sanity_metric.py -q`
Expected: PASS (12 tests: 2 contract + 10 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/metric/hierarchy_sanity.py src/lattice/adapters/__init__.py tests/adapters/test_hierarchy_sanity_metric.py
git commit -m "feat: add hierarchy-sanity structural metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `coherence` DocumentMetric

**Files:**
- Create: `src/lattice/adapters/document_metric/coherence.py`
- Modify: `src/lattice/adapters/__init__.py` (document_metric import line)
- Test: `tests/adapters/test_coherence_metric.py`

**Interfaces:**
- Consumes: `DocumentMetric` port (`evaluate_documents(deltas, ground_truth) -> dict[str, float]`), `Embedder` port, `lattice.core.vectors.cosine`, `HashingEmbedder`, `GraphDelta`/`Resolution` construction via `tests.helpers.make_concept/make_resolution`.
- Produces: registered `("coherence", DocumentMetric)`, constructor `Coherence(embedder: Embedder)` — receives the embedder via Task 1's injection. Keys `coherence`, `multi-surface-concepts`, `singleton-fraction`.
- Does NOT join `DocumentMetricContract` (spec §7 as amended: the contract's `test_unknown_document_raises` is gold-anchored; intrinsic metrics ignore ground truth). Own suite below.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_coherence_metric.py`:

```python
from collections.abc import Sequence

from lattice.adapters.document_metric.coherence import Coherence
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.core.types import GraphDelta
from tests.helpers import make_concept, make_resolution


class CountingEmbedder(HashingEmbedder):
    """Asserts the batched-embed contract: one call for the whole run."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        return super().embed(texts)


def _delta(document_id: str, resolutions) -> GraphDelta:
    return GraphDelta(
        document_id=document_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=(),
        resolutions=tuple(resolutions),
    )


def _resolutions(concept_id: str, surfaces: list[str]):
    concept = make_concept(id=concept_id, label=surfaces[0].casefold())
    return [make_resolution(concept=concept, surface=s) for s in surfaces]


def test_coherent_merge_scores_high_incoherent_low():
    # hashing-embedder cosines, measured: ("beatles","the beatles")=0.8386,
    # ("beatles","kid rock")=0.1336 — assert against safe bounds, not exact.
    embedder = HashingEmbedder()
    coherent = Coherence(embedder).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles", "the beatles"]))], {}
    )
    incoherent = Coherence(embedder).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles", "kid rock"]))], {}
    )
    assert coherent["coherence"] > 0.5
    assert incoherent["coherence"] < 0.5
    assert coherent["multi-surface-concepts"] == 1.0


def test_surfaces_dedupe_casefolded_within_a_concept():
    result = Coherence(HashingEmbedder()).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["Beatles", "beatles", "BEATLES"]))], {}
    )
    # one distinct surface -> not a multi-surface concept -> vacuous 1.0
    assert result["multi-surface-concepts"] == 0.0
    assert result["coherence"] == 1.0


def test_vacuous_coherence_is_one_with_zero_multi_surface():
    result = Coherence(HashingEmbedder()).evaluate_documents(
        [_delta("d1", _resolutions("c1", ["beatles"]))], {}
    )
    assert result["coherence"] == 1.0
    assert result["multi-surface-concepts"] == 0.0


def test_singleton_fraction():
    deltas = [
        _delta("d1", _resolutions("c1", ["beatles", "the beatles"])),
        _delta("d2", _resolutions("c2", ["kid rock"])),
    ]
    result = Coherence(HashingEmbedder()).evaluate_documents(deltas, {})
    # c1 has 2 resolutions, c2 has 1 -> half the concepts are singletons
    assert result["singleton-fraction"] == 0.5


def test_grouping_spans_documents():
    concept = make_concept(id="c1", label="beatles")
    deltas = [
        _delta("d1", [make_resolution(concept=concept, surface="beatles")]),
        _delta("d2", [make_resolution(concept=concept, surface="the beatles")]),
    ]
    result = Coherence(HashingEmbedder()).evaluate_documents(deltas, {})
    assert result["multi-surface-concepts"] == 1.0
    assert result["singleton-fraction"] == 0.0


def test_single_batched_embed_call():
    embedder = CountingEmbedder()
    deltas = [
        _delta("d1", _resolutions("c1", ["beatles", "the beatles"])),
        _delta("d2", _resolutions("c2", ["kid rock", "kid rock songs"])),
    ]
    Coherence(embedder).evaluate_documents(deltas, {})
    assert embedder.calls == 1


def test_no_deltas_is_vacuous():
    result = Coherence(HashingEmbedder()).evaluate_documents([], {})
    assert result == {
        "coherence": 1.0,
        "multi-surface-concepts": 0.0,
        "singleton-fraction": 0.0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_coherence_metric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.document_metric.coherence'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/document_metric/coherence.py`:

```python
from collections.abc import Sequence
from itertools import combinations

from lattice.core.types import GraphDelta
from lattice.core.vectors import cosine
from lattice.ports import DocumentMetric, Embedder
from lattice.registry.registry import register


@register(DocumentMetric, "coherence")
class Coherence(DocumentMetric):
    """Intrinsic merge quality (M5 spec §4.2): what the resolver wrongly
    merged — the counterweight to the redundancy metric. For each concept
    that accumulated >= 2 distinct casefolded mention surfaces across the
    run, coherence is the mean pairwise cosine of the surface embeddings;
    the reported value is the mean over those concepts, and 1.0 when there
    are none (vacuous perfection, made visible by multi-surface-concepts).
    Ground truth is ignored — this metric is intrinsic (spec §7 explains
    why it does not join DocumentMetricContract). One batched embed call
    covers every distinct surface in the run."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]:
        surfaces_by_concept: dict[str, set[str]] = {}
        resolution_counts: dict[str, int] = {}
        for delta in deltas:
            for resolution in delta.resolutions:
                concept_id = resolution.concept.id
                surface = resolution.mention.mention.surface.casefold()
                surfaces_by_concept.setdefault(concept_id, set()).add(surface)
                resolution_counts[concept_id] = (
                    resolution_counts.get(concept_id, 0) + 1
                )
        multi = {
            concept_id: surfaces
            for concept_id, surfaces in surfaces_by_concept.items()
            if len(surfaces) >= 2
        }
        distinct = sorted({s for surfaces in multi.values() for s in surfaces})
        embeddings = (
            dict(zip(distinct, self.embedder.embed(distinct))) if distinct else {}
        )
        per_concept: list[float] = []
        for concept_id in sorted(multi):
            pairs = list(combinations(sorted(multi[concept_id]), 2))
            per_concept.append(
                sum(cosine(embeddings[a], embeddings[b]) for a, b in pairs)
                / len(pairs)
            )
        concept_count = len(surfaces_by_concept)
        singletons = sum(1 for n in resolution_counts.values() if n == 1)
        return {
            "coherence": (
                sum(per_concept) / len(per_concept) if per_concept else 1.0
            ),
            "multi-surface-concepts": float(len(multi)),
            "singleton-fraction": (
                singletons / concept_count if concept_count else 0.0
            ),
        }
```

In `src/lattice/adapters/__init__.py`, change the document_metric import line to:

```python
from lattice.adapters.document_metric import (  # noqa: F401
    clustering,
    coherence,
    f1_at_k,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_coherence_metric.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/document_metric/coherence.py src/lattice/adapters/__init__.py tests/adapters/test_coherence_metric.py
git commit -m "feat: add intrinsic coherence document metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Sweep config + M5 e2e tests

**Files:**
- Create: `configs/m5-conel2-sweep.toml`
- Test: `tests/harness/test_m5_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4 by registered name; `tests/fixtures/mini_clusters_conel` (M3 fixture, 3 conversations); `ExperimentConfig`/`run_experiment`/`SweepConfig`/`expand`/`load_config` as in `tests/harness/test_m3_e2e.py`.
- Produces: the config Task 6 sweeps.

- [ ] **Step 1: Write the failing test**

Create `tests/harness/test_m5_e2e.py`:

```python
import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ROOT = "tests/fixtures/mini_clusters_conel"
REDUNDANCY_KEYS = {"duplicate-rate", "near-duplicate-pairs", "concept-count"}
SANITY_KEYS = {
    "cycle-components", "cycle-nodes", "self-loops",
    "max-depth", "transitive-shortcuts", "is-a-edges",
}
COHERENCE_KEYS = {"coherence", "multi-surface-concepts", "singleton-fraction"}


def _config(resolver: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token"},
            "scorer": {"name": "embedding-cosine"},
            "resolver": resolver,
            "relation_inducer": {
                "name": "union",
                "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
            },
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {
                "name": "mention-clusters",
                "params": {"root": ROOT, "split": "test"},
            },
            "metrics": [{"name": "redundancy"}, {"name": "hierarchy-sanity"}],
            "document_metrics": [{"name": "coherence"}],
        }
    )


@pytest.mark.parametrize(
    "resolver",
    [
        {"name": "exact-label"},
        {"name": "embedding-nn", "params": {"threshold": 0.8}},
    ],
)
def test_m5_intrinsic_pipeline_pure(resolver):
    """Full real-shape pipeline (token extractor standing in for spaCy) with
    all three intrinsic metrics — proves wiring without the ml stack."""
    report = run_experiment(_config(resolver))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["redundancy"]) == REDUNDANCY_KEYS
    assert set(report.metrics["hierarchy-sanity"]) == SANITY_KEYS
    assert set(report.metrics["coherence"]) == COHERENCE_KEYS
    assert 0.0 <= report.metrics["redundancy"]["duplicate-rate"] <= 1.0
    assert 0.0 <= report.metrics["coherence"]["singleton-fraction"] <= 1.0
    assert report.metrics["hierarchy-sanity"]["self-loops"] == 0.0


def test_m5_run_is_reproducible():
    config = _config({"name": "embedding-nn", "params": {"threshold": 0.8}})
    assert run_experiment(config) == run_experiment(config)


def test_m5_sweep_config_expands_to_four_resolver_rows():
    sweep = load_config("configs/m5-conel2-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    assert [c.resolver.name for c in configs] == [
        "exact-label", "embedding-nn", "embedding-nn", "embedding-nn",
    ]
    assert [
        c.resolver.params.get("threshold") for c in configs
    ] == [None, 0.90, 0.75, 0.65]
    for config in configs:
        assert config.extractor.name == "noun-chunk"
        assert config.embedder.name == "sentence-transformer"
        assert {m.name for m in config.metrics} == {"redundancy", "hierarchy-sanity"}
        assert [m.name for m in config.document_metrics] == ["coherence"]


@pytest.mark.ml
def test_m5_real_pipeline_path():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    config = _config({"name": "embedding-nn", "params": {"threshold": 0.8}})
    data = config.model_dump()
    data["extractor"] = {"name": "noun-chunk"}
    data["embedder"] = {"name": "sentence-transformer"}
    try:
        report = run_experiment(ExperimentConfig.model_validate(data))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert set(report.metrics["coherence"]) == COHERENCE_KEYS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_m5_e2e.py -q`
Expected: FAIL — the sweep-config test errors on missing `configs/m5-conel2-sweep.toml`; the fixture-backed tests should already PASS (if any fail, STOP and report — do not patch other tasks' code).

- [ ] **Step 3: Write the config**

Create `configs/m5-conel2-sweep.toml`:

```toml
# M5 integration sweep (spec §5): the first all-real pipeline configuration.
# Intrinsic metrics judge the accreted graph; the resolver axis exposes the
# redundancy/coherence tension so an operating point can be chosen without
# gold. Requires the ml extras and cached models (scripts/fetch_models.py)
# plus data/conel2 (scripts/fetch_conel2.py).

[base.segmenter]
name = "block"

[base.extractor]
name = "noun-chunk"

[base.scorer]
name = "embedding-cosine"

[base.resolver]
name = "exact-label"

[base.relation_inducer]
name = "union"
[base.relation_inducer.params]
members = [{ name = "hearst" }, { name = "compound" }]

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
name = "mention-clusters"
[base.dataset.params]
root = "data/conel2"
split = "test"

[[base.metrics]]
name = "redundancy"

[[base.metrics]]
name = "hierarchy-sanity"

[[base.document_metrics]]
name = "coherence"

[axes]
resolver = [
  { name = "exact-label" },
  { name = "embedding-nn", params = { threshold = 0.90 } },
  { name = "embedding-nn", params = { threshold = 0.75 } },
  { name = "embedding-nn", params = { threshold = 0.65 } },
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/harness/test_m5_e2e.py -q`
Expected: PASS (4 tests + 1 ml skip or pass). Then the whole suite: `uv run --no-sync pytest -q` — no new failures anywhere.

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add configs/m5-conel2-sweep.toml tests/harness/test_m5_e2e.py
git commit -m "feat: add M5 intrinsic sweep config and end-to-end tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Exit criteria — real sweep + spec §8 adjudication

Operational task (orchestrator/observer; nothing committed — reports/ and data/ are gitignored). Spec §8 is the contract.

- [ ] **Step 1: Preconditions**

```bash
ls data/conel2/test.jsonl  # from M3; if missing: SSL export + uv run --no-sync python scripts/fetch_conel2.py
uv run --no-sync python -c "import spacy, sentence_transformers" || echo "ml group missing"
```

Models are cached from M2–M4. If spaCy/sentence-transformers imports fail, run `uv sync --group ml` first; if model loads fail, `uv run --group ml python scripts/fetch_models.py`.

- [ ] **Step 2: Run the sweep**

```bash
uv run --no-sync python -m lattice.harness --sweep configs/m5-conel2-sweep.toml reports/m5-conel2
```

Expected: 4 rows, `errors` 0 everywhere. 58 documents/row through spaCy + MiniLM on CPU — allow ~10–30 min total; use a generous Bash timeout or a background run.

- [ ] **Step 3: Adjudicate against spec §8 and record in the ledger**

1. Mechanical: 4/4 rows zero errors; full suite + ruff clean.
2. Discrimination (hard): `duplicate-rate` exact-label strictly greatest; `concept-count` strictly decreasing exact-label → nn@0.90 → nn@0.75 → nn@0.65 (adjacent-nn ties adjudicable).
3. Tension (adjudicable): `coherence` non-increasing as threshold loosens; a violation needs a recorded, evidence-backed explanation before close.
4. Hierarchy: `self-loops` == 0 all rows; `is-a-edges` > 0; cycle/shortcut counts inspected.
5. Qualitative cross-check (non-gating): for the chosen operating row, list top-10 concepts by resolution count with member surfaces; eyeball against ConEL-2 gold entities; record in ledger.

- [ ] **Step 4: Full verification**

```bash
uv run --no-sync pytest -q && uv run --no-sync ruff check . && git status
```

Expected: suite green, lint clean, tree clean.

---

## Self-Review Notes (already applied)

- Spec coverage: §3→T1, §4.1→T2, §4.3→T3, §4.2→T4, §5→T5, §6 (degenerate-input behavior tested per metric; FileNotFoundError/OSError paths pre-exist), §7→each task's tests (coherence deliberately outside DocumentMetricContract per amended spec), §8→T6.
- All graph algorithms and test constants machine-verified pre-commit: Tarjan (2-cycle, scc+tail, 2000-chain), longest-path (chain=2, tail=1), shortcuts (triangle=1, diamond=1, cycle-with-tail=0, shortcut-through-a-cycle=1), normalize table, hashing cosines 0.8386/0.1336 → the >0.5/<0.5 bounds in Task 4.
- Verification caught and fixed one defect pre-commit: `_is_shortcut` originally excluded the direct edge only at the walk's first step, so a cycle revisiting the source falsely counted `b→c` (in a↔b, b→c) as a shortcut; the mid-walk edge guard plus a planted regression test now cover it.
- Type consistency: `Coherence(embedder)` matches Task 1's injected key `"embedder"`; metric key spellings in Task 5's e2e match Tasks 2–4 exactly; `union` member-dict shape matches M4's adapter.
- Float-equality lesson (M4): no exact float assertions on computed cosines anywhere — only bounds and exact-by-construction values (counts, 0.0/1.0/0.5 fractions with exact binary representations).
