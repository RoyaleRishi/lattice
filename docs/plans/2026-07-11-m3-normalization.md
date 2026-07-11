# M3 — Normalization Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embedding-NN resolver + B³/ARI clustering metric + ECB+ and ConEL-2 gold-mention benchmarks, compared against the exact-label baseline in threshold sweeps.

**Architecture:** One new field on `GraphDelta` (`resolutions`), five new adapters behind
existing ports (resolver `embedding-nn`, scorer `passthrough`, extractor `gold-mentions`,
dataset `mention-clusters`, document-metric `clustering`), two stdlib fetch/convert scripts,
two sweep configs. Spec: `docs/2026-07-11-m3-normalization-design.md` (as pinned at b7cf2d1).

**Tech Stack:** Python stdlib only (xml.etree, zipfile, urllib, json, math). Zero new
dependencies. The ml group is touched only by ml-marked e2e tests and the real sweeps.

## Global Constraints

- `pyproject.toml` is **FROZEN**. M3 adds zero dependencies. Do not edit it for any reason.
- macOS quirk: every test/run command must be
  `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync <cmd>`
- Lean suite stays green without the ml group; ml tests use `pytest.importorskip` +
  `@pytest.mark.ml` + `except OSError: pytest.skip(...)`.
- No downloads inside adapters or tests. `data/`, `reports/`, `docs/papers/` stay gitignored.
  Fetch scripts download only when run explicitly (exit-criteria step).
- `uv run --no-sync ruff check` passes before every commit (line length 100, rules E/F/I/UP).
- Follow existing idioms: `@register(Port, "name")`, sorted determinism, contract-suite
  subclassing, hard errors that name the fetch script / invariant.
- Do not redesign anything. If reality contradicts the plan, STOP and report.

## Paper/corpus fidelity (context for reviewers)

- Conversion rules were pinned against the **downloaded** corpora on 2026-07-11 (spec §5):
  ECB+ entity-tag prefixes, chain-id assignment, 3 non-contiguous skips, duplicate-span
  dedupe; ConEL-2 three splits, span-correction rule (1 known case), no-newline invariant.
- B³: mention-wise precision/recall averaged over mentions; F1 = harmonic mean of the
  averages (Bagga & Baldwin 1998). ARI: standard adjusted Rand over the mention partition;
  degenerate `max_index == expected_index` → 1.0 (both partitions trivial and identical).
- Recommended implementer models: T1/T2/T5/T6 haiku; T3/T4/T7/T8/T9 sonnet. Reviewers
  sonnet; final review opus.

---

### Task 1: `GraphDelta.resolutions`

**Files:**
- Modify: `src/lattice/core/types.py` (GraphDelta dataclass)
- Modify: `src/lattice/orchestrator/orchestrator.py` (success-path delta construction)
- Test: `tests/orchestrator/test_delta_resolutions.py` (new file)

**Interfaces:**
- Consumes: existing `Resolution` type (already defined above `GraphDelta` in types.py).
- Produces: `GraphDelta.resolutions: tuple[Resolution, ...] = ()` — Task 3's metric and
  Task 9's e2e rely on it being populated with the resolver's output on the success path.

- [ ] **Step 1: Write the failing test**

`tests/orchestrator/test_delta_resolutions.py`:

```python
from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.extractor.token import TokenExtractor
from lattice.adapters.graph_integrator.in_memory import InMemoryGraphIntegrator
from lattice.adapters.relation_inducer.co_occurrence import CoOccurrenceInducer
from lattice.adapters.resolver.exact_label import ExactLabelResolver
from lattice.adapters.scorer.frequency import FrequencyScorer
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.core.types import Document
from lattice.orchestrator.orchestrator import Orchestrator


def _orchestrator() -> Orchestrator:
    return Orchestrator(
        segmenter=BlockSegmenter(),
        extractor=TokenExtractor(min_length=4),
        scorer=FrequencyScorer(top_k=5),
        resolver=ExactLabelResolver(
            embedder=HashingEmbedder(dim=16), concept_store=InMemoryConceptStore()
        ),
        relation_inducer=CoOccurrenceInducer(),
        graph_integrator=InMemoryGraphIntegrator(),
    )


def test_delta_carries_resolutions():
    delta = _orchestrator().process(
        Document(id="d1", kind="note", text="alpha beta alpha", timestamp=1.0)
    )
    assert len(delta.resolutions) == len(delta.selected_mentions) > 0
    assert {r.mention.mention.surface for r in delta.resolutions} == {"alpha", "beta"}
    assert all(r.concept.id for r in delta.resolutions)


def test_default_is_empty_tuple():
    from lattice.core.types import GraphDelta

    delta = GraphDelta(
        document_id="d", concepts_added=(), concepts_updated=(), relations_added=()
    )
    assert delta.resolutions == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/orchestrator/test_delta_resolutions.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'resolutions'` /
`AttributeError: ... no attribute 'resolutions'`

- [ ] **Step 3: Implement**

In `src/lattice/core/types.py`, add one field to `GraphDelta` (after
`selected_mentions`) and extend the docstring's last sentence:

```python
    selected_mentions: tuple[ScoredMention, ...] = ()
    resolutions: tuple[Resolution, ...] = ()
```

Docstring: append `` `resolutions` is the resolver's mention→concept assignment for this
document, the unit of normalization evaluation (M3 spec §3).``

In `src/lattice/orchestrator/orchestrator.py`, the success-path `GraphDelta(...)`
construction gains one argument after `selected_mentions=tuple(selected),`:

```python
            resolutions=tuple(resolutions),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/orchestrator/test_delta_resolutions.py -q`
Expected: 2 passed

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/core/types.py src/lattice/orchestrator/orchestrator.py tests/orchestrator/test_delta_resolutions.py
git commit -m "feat: expose resolver output on GraphDelta.resolutions"
```

---

### Task 2: Passthrough scorer

**Files:**
- Create: `src/lattice/adapters/scorer/passthrough.py`
- Modify: `src/lattice/adapters/__init__.py` (scorer import line)
- Test: `tests/adapters/test_passthrough_scorer.py`

**Interfaces:**
- Consumes: `Scorer` port, `ScoredMention`.
- Produces: registered scorer `"passthrough"` (no params). Tasks 9's configs use it.

- [ ] **Step 1: Write the failing test**

`tests/adapters/test_passthrough_scorer.py`:

```python
from lattice.adapters.scorer.passthrough import PassthroughScorer
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class TestPassthroughScorer(ScorerContract):
    def make_scorer(self) -> PassthroughScorer:
        return PassthroughScorer()

    def test_everything_selected_at_salience_one(self):
        unit = make_unit(id="d:u0", text="alpha beta gamma")
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="gamma", unit_id="d:u0", span=(11, 16)),
        ]
        scored = self.make_scorer().score(mentions, [unit])
        assert all(sm.selected and sm.salience == 1.0 for sm in scored)
        assert len(scored) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_passthrough_scorer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer.passthrough'`

- [ ] **Step 3: Implement**

`src/lattice/adapters/scorer/passthrough.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.ports import Scorer
from lattice.registry.registry import register


@register(Scorer, "passthrough")
class PassthroughScorer(Scorer):
    """Evaluation-protocol scorer (M3 spec §4.3): selects every mention at
    salience 1.0. Paired with the gold-mentions extractor so resolution
    metrics see every gold mention — never use it for salience experiments."""

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        return [
            ScoredMention(mention=m, salience=1.0, selected=True) for m in mentions
        ]
```

In `src/lattice/adapters/__init__.py`, change the scorer import line to:

```python
from lattice.adapters.scorer import (  # noqa: F401
    embedding_cosine,
    frequency,
    hcuke,
    mderank,
    passthrough,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_passthrough_scorer.py -q`
Expected: 4 passed (3 contract + 1 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/scorer/passthrough.py src/lattice/adapters/__init__.py tests/adapters/test_passthrough_scorer.py
git commit -m "feat: add passthrough scorer for gold-mention evaluation"
```

---

### Task 3: Clustering metric (B³ + ARI)

**Files:**
- Create: `src/lattice/adapters/document_metric/clustering.py`
- Modify: `src/lattice/adapters/__init__.py` (document_metric import line)
- Test: `tests/adapters/test_clustering_metric.py`

**Interfaces:**
- Consumes: `GraphDelta.resolutions` (Task 1); `DocumentMetric` port
  (`evaluate_documents(deltas, ground_truth) -> dict[str, float]`).
- Produces: registered document-metric `"clustering"` (no params) with output keys
  `"b3-precision"`, `"b3-recall"`, `"b3-f1"`, `"ari"`. Ground-truth shape consumed:
  `{"clusters_by_mention": {"<doc_id>:<start>-<end>": cluster_id}}`. Task 9 relies on the
  key names.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_clustering_metric.py`:

```python
import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.core.types import Concept, GraphDelta, Mention, Resolution, ScoredMention
from tests.contracts.document_metric_contract import DocumentMetricContract


def _delta(document_id: str, rows: list[tuple[str, tuple[int, int], str]]) -> GraphDelta:
    """Delta whose resolutions assign mention (surface, span) -> concept id."""
    resolutions = []
    for surface, span, concept_id in rows:
        mention = Mention(
            surface=surface, unit_id=f"{document_id}:u0", span=span, context=surface
        )
        scored = ScoredMention(mention=mention, salience=1.0, selected=True)
        concept = Concept(
            id=concept_id, label=surface, embedding=(1.0, 0.0),
            first_seen=document_id, updated_at=document_id,
        )
        resolutions.append(Resolution(concept=concept, mention=scored, is_new=True))
    return GraphDelta(
        document_id=document_id, concepts_added=(), concepts_updated=(),
        relations_added=(), resolutions=tuple(resolutions),
    )


class TestClusteringMetricContract(DocumentMetricContract):
    def make_metric(self) -> ClusteringMetric:
        return ClusteringMetric()

    def make_ground_truth(self) -> dict:
        return {"clusters_by_mention": {"d1:0-5": "g1", "d1:6-9": "g1"}}

    def make_deltas(self) -> list[GraphDelta]:
        return [_delta("d1", [("alpha", (0, 5), "k1"), ("beta", (6, 9), "k1")])]


class TestClusteringMetricValues:
    # Textbook example: gold G1={a,b,c}, G2={d,e}; predicted P1={a,b}, P2={c,d,e}.
    # B3 precision per mention: a=1, b=1, c=1/3, d=2/3, e=2/3 -> mean 11/15.
    # B3 recall per mention:    a=2/3, b=2/3, c=1/3, d=1, e=1 -> mean 11/15.
    # ARI: index=2, sum_pred=4, sum_gold=4, total=C(5,2)=10, expected=1.6,
    #      max=4 -> (2-1.6)/(4-1.6) = 1/6.
    GOLD = {
        "clusters_by_mention": {
            "d1:0-1": "G1", "d1:2-3": "G1", "d1:4-5": "G1", "d2:0-1": "G2", "d2:2-3": "G2",
        }
    }

    def _deltas(self) -> list[GraphDelta]:
        return [
            _delta("d1", [("a", (0, 1), "P1"), ("b", (2, 3), "P1"), ("c", (4, 5), "P2")]),
            _delta("d2", [("d", (0, 1), "P2"), ("e", (2, 3), "P2")]),
        ]

    def test_hand_computed_b3_and_ari(self):
        result = ClusteringMetric().evaluate_documents(self._deltas(), self.GOLD)
        assert result["b3-precision"] == pytest.approx(11 / 15)
        assert result["b3-recall"] == pytest.approx(11 / 15)
        assert result["b3-f1"] == pytest.approx(11 / 15)
        assert result["ari"] == pytest.approx(1 / 6)

    def test_perfect_clustering_scores_one(self):
        deltas = [
            _delta("d1", [("a", (0, 1), "X"), ("b", (2, 3), "X"), ("c", (4, 5), "X")]),
            _delta("d2", [("d", (0, 1), "Y"), ("e", (2, 3), "Y")]),
        ]
        result = ClusteringMetric().evaluate_documents(deltas, self.GOLD)
        assert result == {
            "b3-precision": 1.0, "b3-recall": 1.0, "b3-f1": 1.0, "ari": 1.0,
        }

    def test_single_predicted_cluster_has_perfect_recall(self):
        # gold {a,b} + {c}; predicted one cluster {a,b,c}:
        # precision = (2/3 + 2/3 + 1/3)/3 = 5/9; recall = 1; ari = 0.
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G1", "d1:4-5": "G2"}}
        deltas = [
            _delta("d1", [("a", (0, 1), "P"), ("b", (2, 3), "P"), ("c", (4, 5), "P")])
        ]
        result = ClusteringMetric().evaluate_documents(deltas, gold)
        assert result["b3-precision"] == pytest.approx(5 / 9)
        assert result["b3-recall"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(0.0)

    def test_all_singletons_on_singleton_gold_is_perfect(self):
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G2"}}
        deltas = [_delta("d1", [("a", (0, 1), "P1"), ("b", (2, 3), "P2")])]
        result = ClusteringMetric().evaluate_documents(deltas, gold)
        assert result == {
            "b3-precision": 1.0, "b3-recall": 1.0, "b3-f1": 1.0, "ari": 1.0,
        }

    def test_coverage_mismatch_raises_both_directions(self):
        deltas = [_delta("d1", [("a", (0, 1), "P1"), ("z", (8, 9), "P1")])]
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G1"}}
        with pytest.raises(ValueError, match="coverage mismatch"):
            ClusteringMetric().evaluate_documents(deltas, gold)

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="no documents"):
            ClusteringMetric().evaluate_documents([], self.GOLD)

    def test_missing_ground_truth_key_raises(self):
        with pytest.raises(ValueError, match="clusters_by_mention"):
            ClusteringMetric().evaluate_documents(self._deltas(), {"wrong": {}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_clustering_metric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.document_metric.clustering'`

- [ ] **Step 3: Implement**

`src/lattice/adapters/document_metric/clustering.py`:

```python
from collections import Counter
from collections.abc import Sequence
from math import comb

from lattice.core.types import GraphDelta
from lattice.ports import DocumentMetric
from lattice.registry.registry import register


def _b_cubed(pred: dict[str, str], gold: dict[str, str]) -> tuple[float, float, float]:
    """Bagga & Baldwin (1998): mention-wise precision/recall averaged over
    mentions; F1 is the harmonic mean of the two averages."""
    pred_clusters: dict[str, set[str]] = {}
    gold_clusters: dict[str, set[str]] = {}
    for key, cluster in pred.items():
        pred_clusters.setdefault(cluster, set()).add(key)
    for key, cluster in gold.items():
        gold_clusters.setdefault(cluster, set()).add(key)
    precision = recall = 0.0
    for key in pred:
        overlap = len(pred_clusters[pred[key]] & gold_clusters[gold[key]])
        precision += overlap / len(pred_clusters[pred[key]])
        recall += overlap / len(gold_clusters[gold[key]])
    precision /= len(pred)
    recall /= len(pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _ari(pred: dict[str, str], gold: dict[str, str]) -> float:
    """Adjusted Rand index over the mention partition. When max_index equals
    expected_index both partitions are trivial (all-singletons on both sides
    or one cluster on both sides) and identical: defined as 1.0."""
    keys = list(pred)
    contingency = Counter((pred[k], gold[k]) for k in keys)
    pred_sizes = Counter(pred[k] for k in keys)
    gold_sizes = Counter(gold[k] for k in keys)
    index = float(sum(comb(c, 2) for c in contingency.values()))
    sum_pred = sum(comb(c, 2) for c in pred_sizes.values())
    sum_gold = sum(comb(c, 2) for c in gold_sizes.values())
    total = comb(len(keys), 2)
    if total == 0:
        return 1.0
    expected = sum_pred * sum_gold / total
    max_index = (sum_pred + sum_gold) / 2
    if max_index == expected:
        return 1.0
    return (index - expected) / (max_index - expected)


@register(DocumentMetric, "clustering")
class ClusteringMetric(DocumentMetric):
    """Cross-document clustering quality over gold mentions (M3 spec §4.5).
    Predicted clusters group mention keys f"{doc_id}:{start}-{end}" by the
    resolved concept id across ALL deltas; gold comes from
    ground_truth["clusters_by_mention"]. Coverage must match 1:1 in both
    directions — the gold-mention protocol guarantees it, so any mismatch is
    a broken config, never a metric decision (spec §7)."""

    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]:
        by_mention = ground_truth.get("clusters_by_mention")
        if not isinstance(by_mention, dict):
            raise ValueError('clustering requires ground_truth["clusters_by_mention"]')
        deltas = list(deltas)
        if not deltas:
            raise ValueError("no documents to evaluate")
        pred: dict[str, str] = {}
        for delta in deltas:
            for resolution in delta.resolutions:
                start, end = resolution.mention.mention.span
                pred[f"{delta.document_id}:{start}-{end}"] = resolution.concept.id
        gold = {str(k): str(v) for k, v in by_mention.items()}
        missing = sorted(set(gold) - set(pred))
        extra = sorted(set(pred) - set(gold))
        if missing or extra:
            raise ValueError(
                f"mention coverage mismatch: {len(missing)} gold mentions unpredicted "
                f"(e.g. {missing[:3]}), {len(extra)} predictions not in gold "
                f"(e.g. {extra[:3]}) — gold-mention protocol requires 1:1 coverage"
            )
        if not pred:
            raise ValueError("no mentions to evaluate")
        precision, recall, f1 = _b_cubed(pred, gold)
        return {
            "b3-precision": precision,
            "b3-recall": recall,
            "b3-f1": f1,
            "ari": _ari(pred, gold),
        }
```

In `src/lattice/adapters/__init__.py`, change the document_metric import line to:

```python
from lattice.adapters.document_metric import clustering, f1_at_k  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_clustering_metric.py -q`
Expected: 9 passed (2 contract + 7 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/document_metric/clustering.py src/lattice/adapters/__init__.py tests/adapters/test_clustering_metric.py
git commit -m "feat: add B3 + ARI clustering document metric"
```

---

### Task 4: Embedding-NN resolver

**Files:**
- Create: `src/lattice/adapters/resolver/embedding_nn.py`
- Modify: `src/lattice/adapters/__init__.py` (resolver import line)
- Test: `tests/adapters/test_embedding_nn_resolver.py`

**Interfaces:**
- Consumes: `Embedder.embed(texts) -> list[tuple[float, ...]]`;
  `ConceptStore.find_by_label/nearest/upsert`; `nearest` returns
  `list[tuple[Concept, float]]` with cosine similarity, highest first.
- Produces: registered resolver `"embedding-nn"`, constructor
  `(embedder: Embedder, concept_store: ConceptStore, threshold: float = 0.8)`.
  Task 9's configs use it (factory injects embedder/concept_store by parameter name).

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_embedding_nn_resolver.py`:

```python
from lattice.adapters.concept_store.in_memory import InMemoryConceptStore
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.resolver.embedding_nn import EmbeddingNNResolver
from lattice.ports import Embedder
from tests.contracts.resolver_contract import ResolverContract
from tests.helpers import make_document, make_scored_mention


class LookupEmbedder(Embedder):
    """Test double: fixed vector per exact text; `default` otherwise."""

    def __init__(self, mapping: dict[str, tuple[float, ...]], default: tuple[float, ...]):
        self.mapping = mapping
        self.default = default

    @property
    def dim(self) -> int:
        return len(self.default)

    def embed(self, texts):
        return [self.mapping.get(t, self.default) for t in texts]


def _resolver(threshold: float, mapping: dict | None = None) -> EmbeddingNNResolver:
    embedder = (
        LookupEmbedder(mapping, default=(0.0, 1.0)) if mapping is not None
        else HashingEmbedder(dim=16)
    )
    return EmbeddingNNResolver(
        embedder=embedder, concept_store=InMemoryConceptStore(), threshold=threshold
    )


class TestEmbeddingNNResolver(ResolverContract):
    def make_resolver(self) -> EmbeddingNNResolver:
        return EmbeddingNNResolver(
            embedder=HashingEmbedder(dim=16),
            concept_store=InMemoryConceptStore(),
            threshold=0.8,
        )

    def test_merges_exactly_at_threshold(self):
        # cos((1,0), (0.8,0.6)) = 0.8 exactly; threshold 0.8 must merge (>=).
        resolver = _resolver(0.8, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="alphaz")], make_document(id="d2"))
        assert not r2.is_new
        assert r2.concept.id == r1.concept.id
        assert r2.concept.label == "alpha"  # merged concept keeps its own label
        assert r2.concept.updated_at == "d2"

    def test_creates_just_above_threshold(self):
        resolver = _resolver(0.81, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="alphaz")], make_document(id="d2"))
        assert r2.is_new
        assert r2.concept.id != r1.concept.id

    def test_exact_label_short_circuits_regardless_of_threshold(self):
        # threshold 2.0 makes the NN path unreachable; identical strings must
        # still merge via find_by_label.
        resolver = _resolver(2.0)
        [r1] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        [r2] = resolver.resolve([make_scored_mention(surface="Alpha ")], make_document(id="d2"))
        assert not r2.is_new
        assert r2.concept.id == r1.concept.id

    def test_stream_semantics_within_one_document(self):
        # Second mention merges into the concept created earlier in the SAME call.
        resolver = _resolver(0.8, {"alpha": (1.0, 0.0), "alphaz": (0.8, 0.6)})
        resolutions = resolver.resolve(
            [make_scored_mention(surface="alpha"), make_scored_mention(surface="alphaz")],
            make_document(id="d1"),
        )
        assert [r.is_new for r in resolutions] == [True, False]
        assert resolutions[0].concept.id == resolutions[1].concept.id

    def test_empty_store_creates_first_concept(self):
        resolver = _resolver(0.0)  # even threshold 0 cannot merge into nothing
        [r] = resolver.resolve([make_scored_mention(surface="alpha")], make_document(id="d1"))
        assert r.is_new
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_embedding_nn_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.resolver.embedding_nn'`

- [ ] **Step 3: Implement**

`src/lattice/adapters/resolver/embedding_nn.py`:

```python
import uuid
from collections.abc import Sequence
from dataclasses import replace

from lattice.core.types import Concept, Document, Resolution, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@register(Resolver, "embedding-nn")
class EmbeddingNNResolver(Resolver):
    """M3 resolver (spec §4.1): exact-label short-circuit, then embedding
    nearest-neighbour merge at `threshold` cosine similarity, else create a
    new concept. Concept embeddings are fixed at creation (no centroid
    updates in M3 — documented deferral). One embed batch per document;
    mentions resolve in input order, so later mentions can merge into
    concepts created earlier in the same document (stream semantics)."""

    def __init__(
        self, embedder: Embedder, concept_store: ConceptStore, threshold: float = 0.8
    ):
        self.embedder = embedder
        self.concept_store = concept_store
        self.threshold = threshold

    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]:
        if not scored_mentions:
            return []
        labels = [sm.mention.surface.strip().lower() for sm in scored_mentions]
        unique = sorted(set(labels))
        vectors = dict(zip(unique, self.embedder.embed(unique)))
        resolutions: list[Resolution] = []
        for scored_mention, label in zip(scored_mentions, labels):
            existing = self.concept_store.find_by_label(label)
            if existing is None:
                hits = self.concept_store.nearest(vectors[label], k=1)
                if hits and hits[0][1] >= self.threshold:
                    existing = hits[0][0]
            if existing is not None:
                updated = replace(existing, updated_at=document.id)
                self.concept_store.upsert(updated)
                resolutions.append(
                    Resolution(concept=updated, mention=scored_mention, is_new=False)
                )
            else:
                concept = Concept(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{label}")),
                    label=label,
                    embedding=vectors[label],
                    first_seen=document.id,
                    updated_at=document.id,
                )
                self.concept_store.upsert(concept)
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
```

In `src/lattice/adapters/__init__.py`, change the resolver import line to:

```python
from lattice.adapters.resolver import embedding_nn, exact_label  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_embedding_nn_resolver.py -q`
Expected: 8 passed (3 contract + 5 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/resolver/embedding_nn.py src/lattice/adapters/__init__.py tests/adapters/test_embedding_nn_resolver.py
git commit -m "feat: add embedding nearest-neighbour resolver"
```

---

### Task 5: Mention-clusters dataset + mini fixtures

**Files:**
- Create: `src/lattice/adapters/dataset/mention_clusters.py`
- Create: `tests/fixtures/mini_clusters_ecb/test.jsonl`
- Create: `tests/fixtures/mini_clusters_conel/test.jsonl`
- Modify: `src/lattice/adapters/__init__.py` (dataset import line)
- Test: `tests/adapters/test_mention_clusters_dataset.py`

**Interfaces:**
- Consumes: `Dataset` port; JSONL rows
  `{"id", "kind", "text", "mentions": [{"start", "end", "surface", "cluster"}]}`.
- Produces: registered dataset `"mention-clusters"`, constructor
  `(root: str, split: str = "test", limit: int | None = None)`;
  `ground_truth() -> {"clusters_by_mention": {f"{id}:{start}-{end}": cluster}}`.
  Tasks 6 and 9 read the same fixtures; Task 9's configs use the adapter.

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/mini_clusters_ecb/test.jsonl` (exactly 3 lines; spans are hand-verified —
`text[start:end] == surface` for every mention):

```json
{"id": "36_1ecbplus", "kind": "article", "text": "Warren Jeffs was found guilty in San Antonio .", "mentions": [{"start": 0, "end": 12, "surface": "Warren Jeffs", "cluster": "HUM1"}, {"start": 33, "end": 44, "surface": "San Antonio", "cluster": "LOC1"}]}
{"id": "36_2ecbplus", "kind": "article", "text": "Jeffs was convicted .", "mentions": [{"start": 0, "end": 5, "surface": "Jeffs", "cluster": "HUM1"}]}
{"id": "36_3ecbplus", "kind": "article", "text": "A jury deliberated .", "mentions": [{"start": 2, "end": 6, "surface": "jury", "cluster": "36_3ecbplus:m1"}]}
```

`tests/fixtures/mini_clusters_conel/test.jsonl` (exactly 3 lines):

```json
{"id": "conel-1", "kind": "transcript", "text": "My favorite singer is Kid Rock !\nMine too !", "mentions": [{"start": 22, "end": 30, "surface": "Kid Rock", "cluster": "Kid_Rock"}]}
{"id": "conel-2", "kind": "transcript", "text": "i like kid rock better", "mentions": [{"start": 7, "end": 15, "surface": "kid rock", "cluster": "Kid_Rock"}]}
{"id": "conel-3", "kind": "transcript", "text": "The Beatles were bigger", "mentions": [{"start": 0, "end": 11, "surface": "The Beatles", "cluster": "The_Beatles"}]}
```

- [ ] **Step 2: Write the failing tests**

`tests/adapters/test_mention_clusters_dataset.py`:

```python
import pytest

from lattice.adapters.dataset.mention_clusters import MentionClustersDataset
from tests.contracts.dataset_contract import DatasetContract

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"
CONEL_ROOT = "tests/fixtures/mini_clusters_conel"


class TestMentionClustersDataset(DatasetContract):
    def make_dataset(self) -> MentionClustersDataset:
        return MentionClustersDataset(root=ECB_ROOT)

    def test_documents_carry_stored_kind_and_text(self):
        docs = list(MentionClustersDataset(root=CONEL_ROOT).documents())
        assert [d.kind for d in docs] == ["transcript"] * 3
        assert docs[0].text.startswith("My favorite singer")

    def test_ground_truth_keys_and_clusters(self):
        truth = MentionClustersDataset(root=ECB_ROOT).ground_truth()
        assert truth == {
            "clusters_by_mention": {
                "36_1ecbplus:0-12": "HUM1",
                "36_1ecbplus:33-44": "LOC1",
                "36_2ecbplus:0-5": "HUM1",
                "36_3ecbplus:2-6": "36_3ecbplus:m1",
            }
        }

    def test_spans_slice_stored_text(self):
        for root in (ECB_ROOT, CONEL_ROOT):
            import json
            from pathlib import Path

            for line in (Path(root) / "test.jsonl").read_text().splitlines():
                record = json.loads(line)
                for m in record["mentions"]:
                    assert record["text"][m["start"]:m["end"]] == m["surface"]

    def test_limit_truncates(self):
        docs = list(MentionClustersDataset(root=ECB_ROOT, limit=1).documents())
        assert len(docs) == 1

    def test_missing_root_names_fetch_scripts(self):
        with pytest.raises(FileNotFoundError, match="fetch_ecbplus|fetch_conel2"):
            list(MentionClustersDataset(root="data/nowhere").documents())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_mention_clusters_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.dataset.mention_clusters'`

- [ ] **Step 4: Implement**

`src/lattice/adapters/dataset/mention_clusters.py`:

```python
import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "mention-clusters")
class MentionClustersDataset(Dataset):
    """Unified mention-cluster benchmark reader (M3 spec §4.4): one JSONL
    shape serves both ECB+ and ConEL-2, emitted by scripts/fetch_ecbplus.py
    and scripts/fetch_conel2.py. Stdlib-only at runtime. Ground truth maps
    mention keys f"{doc_id}:{start}-{end}" to cluster ids."""

    def __init__(self, root: str, split: str = "test", limit: int | None = None):
        self.path = Path(root) / f"{split}.jsonl"
        self.limit = limit

    def _records(self) -> Iterator[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run `uv run --no-sync python "
                f"scripts/fetch_ecbplus.py` or `scripts/fetch_conel2.py` first"
            )
        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    return
                yield json.loads(line)

    def documents(self) -> Iterator[Document]:
        for i, record in enumerate(self._records()):
            yield Document(
                id=record["id"], kind=record["kind"], text=record["text"],
                timestamp=float(i),
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "clusters_by_mention": {
                f"{record['id']}:{m['start']}-{m['end']}": m["cluster"]
                for record in self._records()
                for m in record["mentions"]
            }
        }
```

In `src/lattice/adapters/__init__.py`, change the dataset import line to:

```python
from lattice.adapters.dataset import inspec, mention_clusters, toy  # noqa: F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_mention_clusters_dataset.py -q`
Expected: 9 passed (4 contract + 5 new)

- [ ] **Step 6: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/dataset/mention_clusters.py src/lattice/adapters/__init__.py tests/fixtures/mini_clusters_ecb tests/fixtures/mini_clusters_conel tests/adapters/test_mention_clusters_dataset.py
git commit -m "feat: add unified mention-clusters dataset adapter + mini fixtures"
```

---

### Task 6: Gold-mentions extractor

**Files:**
- Create: `src/lattice/adapters/extractor/gold_mentions.py`
- Modify: `src/lattice/adapters/__init__.py` (extractor import line)
- Test: `tests/adapters/test_gold_mentions_extractor.py`

**Interfaces:**
- Consumes: the converted JSONL from Task 5's fixture shape (same `root`/`split` params
  as the dataset adapter); `Extractor` port (`extract(units) -> list[Mention]`).
- Produces: registered extractor `"gold-mentions"`, constructor
  `(root: str, split: str = "test")`. NOTE (spec §8 as pinned): this adapter has its
  **own focused suite**, NOT the generic ExtractorContract — the generic contract feeds
  hard-coded units that cannot exist in a sidecar corpus.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_gold_mentions_extractor.py`:

```python
import pytest

from lattice.adapters.extractor.gold_mentions import GoldMentionExtractor
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.core.types import Document
from tests.helpers import make_unit

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"


def _units_for(doc_id: str, text: str):
    return BlockSegmenter().segment(
        Document(id=doc_id, kind="article", text=text, timestamp=0.0)
    )


class TestGoldMentionExtractor:
    def make_extractor(self) -> GoldMentionExtractor:
        return GoldMentionExtractor(root=ECB_ROOT)

    def test_emits_gold_mentions_with_valid_spans(self):
        text = "Warren Jeffs was found guilty in San Antonio ."
        units = _units_for("36_1ecbplus", text)
        assert len(units) == 1  # single-unit invariant (spec §4.2)
        mentions = self.make_extractor().extract(units)
        assert [(m.surface, m.span) for m in mentions] == [
            ("Warren Jeffs", (0, 12)),
            ("San Antonio", (33, 44)),
        ]
        assert all(m.unit_id == units[0].id for m in mentions)
        for m in mentions:
            assert units[0].text[m.span[0]:m.span[1]] == m.surface

    def test_no_units_yields_no_mentions(self):
        assert self.make_extractor().extract([]) == []

    def test_unknown_document_raises(self):
        with pytest.raises(ValueError, match="not in gold mention sidecar"):
            self.make_extractor().extract([make_unit(id="x:u0", document_id="unknown-doc")])

    def test_text_mismatch_raises(self):
        units = _units_for("36_1ecbplus", "Tampered text .")
        with pytest.raises(ValueError, match="differs from stored"):
            self.make_extractor().extract(units)

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="fetch"):
            GoldMentionExtractor(root="data/nowhere")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_gold_mentions_extractor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.extractor.gold_mentions'`

- [ ] **Step 3: Implement**

`src/lattice/adapters/extractor/gold_mentions.py`:

```python
import json
from collections.abc import Sequence
from pathlib import Path

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register


@register(Extractor, "gold-mentions")
class GoldMentionExtractor(Extractor):
    """Evaluation-protocol extractor (M3 spec §4.2): emits the gold mention
    spans stored in the converted corpus JSONL, so resolution metrics are not
    contaminated by extraction errors. Must be configured with the same
    root/split as the mention-clusters dataset, and paired with the block
    segmenter (converters emit single-newline text: one unit per document)."""

    def __init__(self, root: str, split: str = "test"):
        path = Path(root) / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the corpus fetch script "
                "(scripts/fetch_ecbplus.py or scripts/fetch_conel2.py) first"
            )
        self._by_document: dict[str, dict] = {}
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                self._by_document[record["id"]] = record

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            record = self._by_document.get(unit.document_id)
            if record is None:
                raise ValueError(
                    f"document {unit.document_id!r} not in gold mention sidecar — "
                    "configure gold-mentions with the same root/split as the dataset"
                )
            if unit.text != record["text"]:
                raise ValueError(
                    f"unit text differs from stored document text for "
                    f"{unit.document_id!r} — use the block segmenter; converters emit "
                    "single-newline text so each document is exactly one unit"
                )
            for m in record["mentions"]:
                mentions.append(
                    Mention(
                        surface=m["surface"],
                        unit_id=unit.id,
                        span=(m["start"], m["end"]),
                        context=record["text"][max(0, m["start"] - 40):m["end"] + 40],
                    )
                )
        return mentions
```

In `src/lattice/adapters/__init__.py`, change the extractor import line to:

```python
from lattice.adapters.extractor import gold_mentions, noun_chunk, token  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_gold_mentions_extractor.py -q`
Expected: 5 passed

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/extractor/gold_mentions.py src/lattice/adapters/__init__.py tests/adapters/test_gold_mentions_extractor.py
git commit -m "feat: add gold-mentions extractor (M3 evaluation protocol)"
```

---

### Task 7: ConEL-2 fetch script

**Files:**
- Create: `scripts/fetch_conel2.py`
- Test: `tests/scripts/test_fetch_conel2.py`

**Interfaces:**
- Consumes: raw ConEL-2 dialogue dicts (`dialogue_id`, `turns[]`; USER turns carry
  `el_annotations` with utterance-relative `span`).
- Produces: `convert_dialogue(dialogue: dict) -> dict` (pure, tested) emitting the Task 5
  JSONL row shape; running the script downloads the three split files and writes
  `data/conel2/{train,validation,test}.jsonl` + `CHECKSUMS`.

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_fetch_conel2.py`:

```python
import pytest

from scripts.fetch_conel2 import convert_dialogue

SAMPLE = {
    "dialogue_id": "42",
    "turns": [
        {
            "turn_number": 0,
            "speaker": "USER",
            "utterance": "I love the Beatles. so much",
            # span [11, 19] slices "Beatles." — the one known corpus defect;
            # the correction rule must trim it to the mention.
            "el_annotations": [
                {"mention": "Beatles", "span": [11, 19], "entity": "The_Beatles"}
            ],
            "personal_entity_annotations": [
                {"personal_entity_mention": "ignored", "entity": "Ignored"}
            ],
        },
        {"turn_number": 1, "speaker": "SYSTEM", "utterance": "Me too!"},
        {
            "turn_number": 2,
            "speaker": "USER",
            "utterance": "kid rock is fine",
            "el_annotations": [
                {"mention": "kid rock", "span": [0, 8], "entity": "Kid_Rock"}
            ],
            "personal_entity_annotations": [],
        },
    ],
}


def test_convert_dialogue_remaps_spans_and_corrects_the_known_defect():
    row = convert_dialogue(SAMPLE)
    assert row["id"] == "conel-42"
    assert row["kind"] == "transcript"
    assert row["text"] == "I love the Beatles. so much\nMe too!\nkid rock is fine"
    assert row["mentions"] == [
        {"start": 11, "end": 18, "surface": "Beatles", "cluster": "The_Beatles"},
        {"start": 36, "end": 44, "surface": "kid rock", "cluster": "Kid_Rock"},
    ]
    for m in row["mentions"]:
        assert row["text"][m["start"]:m["end"]] == m["surface"]


def test_personal_entity_annotations_are_excluded():
    row = convert_dialogue(SAMPLE)
    assert all(m["cluster"] != "Ignored" for m in row["mentions"])


def test_unfixable_span_raises():
    broken = {
        "dialogue_id": "9",
        "turns": [
            {
                "turn_number": 0,
                "speaker": "USER",
                "utterance": "hello world",
                "el_annotations": [
                    {"mention": "zzz", "span": [0, 3], "entity": "Zzz"}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="dialogue 9"):
        convert_dialogue(broken)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_conel2.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_conel2'`

- [ ] **Step 3: Implement**

`scripts/fetch_conel2.py`:

```python
"""Fetch ConEL-2 (Joko & Hasibi 2022) and convert to lattice's unified
mention-cluster JSONL (M3 spec §4.4/§5). Stdlib only:
    uv run --no-sync python scripts/fetch_conel2.py
Splits: Train/Val/Test JSON -> train/validation/test.jsonl. Personal-entity
annotations are excluded (speaker-relative references, not shared concepts).
Cluster id = the gold Wikipedia entity. One known corpus defect (a span
including a trailing period) is fixed by the prefix-trim rule; anything else
raises."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/informagi/"
    "conversational-entity-linking-2022/main/dataset/"
    "Conversational_Entity_Linking_Annotations"
)
SPLITS = {"Train": "train", "Val": "validation", "Test": "test"}


def convert_dialogue(dialogue: dict) -> dict:
    texts: list[str] = []
    mentions: list[dict] = []
    offset = 0
    for turn in dialogue["turns"]:
        utterance = turn["utterance"]
        for ann in turn.get("el_annotations", []):
            start, end = ann["span"]
            surface = ann["mention"]
            if utterance[start:end] != surface:
                if utterance[start:start + len(surface)] == surface:
                    end = start + len(surface)
                else:
                    raise ValueError(
                        f"unfixable span in dialogue {dialogue['dialogue_id']}: "
                        f"{ann!r}"
                    )
            mentions.append(
                {
                    "start": offset + start,
                    "end": offset + end,
                    "surface": surface,
                    "cluster": ann["entity"],
                }
            )
        texts.append(utterance)
        offset += len(utterance) + 1  # the joining "\n"
    return {
        "id": f"conel-{dialogue['dialogue_id']}",
        "kind": "transcript",
        "text": "\n".join(texts),
        "mentions": mentions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    out_dir = Path(args.root) / "conel2"
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    for raw_split, split in SPLITS.items():
        url = f"{BASE}/ConEL22_EL_{raw_split}.json"
        with urllib.request.urlopen(url) as response:
            dialogues = json.load(response)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for dialogue in dialogues:
                f.write(json.dumps(convert_dialogue(dialogue), sort_keys=True) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        print(f"wrote {out_path} ({len(dialogues)} conversations, {digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_conel2.py -q`
Expected: 3 passed

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green (do NOT run the download in tests)

```bash
git add scripts/fetch_conel2.py tests/scripts/test_fetch_conel2.py
git commit -m "feat: add ConEL-2 fetch/convert script"
```

---

### Task 8: ECB+ fetch script

**Files:**
- Create: `scripts/fetch_ecbplus.py`
- Test: `tests/scripts/test_fetch_ecbplus.py`

**Interfaces:**
- Consumes: CROMER XML text + the validated-sentences set for one document.
- Produces: `convert_document(doc_name: str, xml_text: str, validated_sentences: set[str])
  -> dict | None` (pure, tested; None when no validated sentences) emitting the Task 5
  JSONL row shape; running the script downloads the cltl/ecbPlus archive and writes
  `data/ecbplus/{train,test}.jsonl` + `CHECKSUMS` (train = topics 1–35, test = 36–45).

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_fetch_ecbplus.py`:

```python
from scripts.fetch_ecbplus import convert_document, is_entity_tag

SAMPLE_XML = """<Document doc_name="1_1ecbplus.xml" doc_id="DOC1">
<token t_id="1" sentence="0" number="0">http</token>
<token t_id="2" sentence="1" number="0">Warren</token>
<token t_id="3" sentence="1" number="1">Jeffs</token>
<token t_id="4" sentence="1" number="2">guilty</token>
<token t_id="5" sentence="2" number="0">Jury</token>
<token t_id="6" sentence="2" number="1">decides</token>
<Markables>
<HUMAN_PART_PER m_id="1"><token_anchor t_id="2"/><token_anchor t_id="3"/></HUMAN_PART_PER>
<ACTION_OCCURRENCE m_id="2"><token_anchor t_id="4"/></ACTION_OCCURRENCE>
<HUMAN_PART_ORG m_id="3"><token_anchor t_id="5"/></HUMAN_PART_ORG>
<HUMAN_PART_PER m_id="4"><token_anchor t_id="1"/></HUMAN_PART_PER>
<HUMAN_PART_PER m_id="9" RELATED_TO="" TAG_DESCRIPTOR="jeffs" instance_id="HUM99"/>
</Markables>
<Relations>
<CROSS_DOC_COREF r_id="10" note="HUM99"><source m_id="1"/><target m_id="9"/></CROSS_DOC_COREF>
</Relations>
</Document>"""


def test_entity_tag_rule():
    assert is_entity_tag("HUMAN_PART_PER")
    assert is_entity_tag("NON_HUMAN_PART")
    assert is_entity_tag("LOC_GEO")
    assert is_entity_tag("TIME_DATE")
    assert not is_entity_tag("ACTION_OCCURRENCE")
    assert not is_entity_tag("NEG_ACTION_STATE")
    assert not is_entity_tag("UNKNOWN_INSTANCE_TAG")


def test_convert_document_builds_text_spans_and_clusters():
    row = convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences={"1", "2"})
    assert row["id"] == "1_1ecbplus"
    assert row["kind"] == "article"
    # sentence 0 (the URL junk) is excluded by the validated filter; tokens
    # joined with spaces, sentences with newline.
    assert row["text"] == "Warren Jeffs guilty\nJury decides"
    assert row["mentions"] == [
        {"start": 0, "end": 12, "surface": "Warren Jeffs", "cluster": "HUM99"},
        {"start": 20, "end": 24, "surface": "Jury", "cluster": "1_1ecbplus:m3"},
    ]
    for m in row["mentions"]:
        assert row["text"][m["start"]:m["end"]] == m["surface"]


def test_action_markables_and_unvalidated_sentences_are_excluded():
    row = convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences={"1", "2"})
    surfaces = [m["surface"] for m in row["mentions"]]
    assert "guilty" not in surfaces  # ACTION_OCCURRENCE
    assert "http" not in surfaces  # sentence 0 not validated


def test_no_validated_sentences_returns_none():
    assert convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences=set()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_ecbplus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_ecbplus'`

- [ ] **Step 3: Implement**

`scripts/fetch_ecbplus.py`:

```python
"""Fetch ECB+ (Cybulska & Vossen 2014) and convert to lattice's unified
mention-cluster JSONL (M3 spec §4.4/§5). Stdlib only:
    uv run --no-sync python scripts/fetch_ecbplus.py
Entity chains only (tags starting HUMAN_PART/NON_HUMAN_PART/LOC/TIME);
validated-sentences filter applied; cluster ids: CROSS_DOC_COREF note,
INTRA_DOC_COREF {doc}:r{r_id}, else singleton {doc}:m{m_id}. Non-contiguous
mentions (3 corpus-wide) and duplicate spans are skipped, keeping the lowest
m_id. Splits by topic: train 1-35, test 36-45."""

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://github.com/cltl/ecbPlus/archive/refs/heads/master.zip"
INNER_ZIP = "ecbPlus-master/ECB+_LREC2014/ECB+.zip"
SENTENCES_CSV = "ecbPlus-master/ECB+_LREC2014/ECBplus_coreference_sentences.csv"
ENTITY_PREFIXES = ("HUMAN_PART", "NON_HUMAN_PART", "LOC", "TIME")
TEST_TOPICS = set(range(36, 46))


def is_entity_tag(tag: str) -> bool:
    return tag.startswith(ENTITY_PREFIXES)


def convert_document(
    doc_name: str, xml_text: str, validated_sentences: set[str]
) -> dict | None:
    root = ET.fromstring(xml_text)
    sentences: dict[str, list[tuple[int, str]]] = {}
    for t in root.iter("token"):
        if t.get("sentence") in validated_sentences:
            sentences.setdefault(t.get("sentence"), []).append(
                (int(t.get("t_id")), t.text or "")
            )
    if not sentences:
        return None
    spans: dict[int, tuple[int, int]] = {}
    lines: list[str] = []
    cursor = 0
    for s in sorted(sentences, key=int):
        col = 0
        pieces: list[str] = []
        for t_id, word in sorted(sentences[s]):
            if pieces:
                col += 1  # joining space
            spans[t_id] = (cursor + col, cursor + col + len(word))
            pieces.append(word)
            col += len(word)
        lines.append(" ".join(pieces))
        cursor += col + 1  # the joining "\n"
    text = "\n".join(lines)

    cluster_of: dict[str, str] = {}
    for rel in root.find("Relations") or []:
        if rel.tag == "CROSS_DOC_COREF":
            cluster_id = rel.get("note")
        elif rel.tag == "INTRA_DOC_COREF":
            cluster_id = f"{doc_name}:r{rel.get('r_id')}"
        else:
            continue
        for source in rel.findall("source"):
            cluster_of[source.get("m_id")] = cluster_id

    candidates: list[tuple[int, int, int, str, str]] = []
    for m in root.find("Markables") or []:
        anchors = sorted(int(a.get("t_id")) for a in m.findall("token_anchor"))
        if not anchors or not is_entity_tag(m.tag):
            continue
        if anchors != list(range(anchors[0], anchors[-1] + 1)):
            continue  # 3 non-contiguous mentions corpus-wide — skipped (spec §5)
        if any(t_id not in spans for t_id in anchors):
            continue  # anchored outside validated sentences
        start = spans[anchors[0]][0]
        end = spans[anchors[-1]][1]
        m_id = m.get("m_id")
        candidates.append((start, end, int(m_id), m_id, text[start:end]))

    mentions: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, end, _, m_id, surface in sorted(candidates):
        if (start, end) in seen_spans:
            continue  # duplicate span: keep lowest m_id (spec §5)
        seen_spans.add((start, end))
        mentions.append(
            {
                "start": start,
                "end": end,
                "surface": surface,
                "cluster": cluster_of.get(m_id, f"{doc_name}:m{m_id}"),
            }
        )
    return {"id": doc_name, "kind": "article", "text": text, "mentions": mentions}


def _doc_sort_key(doc_name: str) -> tuple[int, int, str]:
    topic, rest = doc_name.split("_", 1)
    number = int(re.match(r"\d+", rest).group())
    return int(topic), number, rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    out_dir = Path(args.root) / "ecbplus"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = out_dir / "ecbplus-master.zip"
    if not archive_path.exists():
        print(f"downloading {ARCHIVE_URL} …")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    with zipfile.ZipFile(archive_path) as outer:
        validated: dict[str, set[str]] = {}
        with outer.open(SENTENCES_CSV) as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                key = f"{row['Topic']}_{row['File']}"
                validated.setdefault(key, set()).add(row["Sentence Number"])
        inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
        rows: dict[str, list[dict]] = {"train": [], "test": []}
        for name in sorted(inner.namelist()):
            if not name.endswith(".xml") or "__MACOSX" in name:
                continue
            doc_name = Path(name).stem
            if doc_name not in validated:
                continue
            row = convert_document(
                doc_name, inner.read(name).decode("utf-8"), validated[doc_name]
            )
            if row is None:
                continue
            topic = int(doc_name.split("_", 1)[0])
            rows["test" if topic in TEST_TOPICS else "train"].append(row)

    checksums: list[str] = []
    for split, split_rows in rows.items():
        split_rows.sort(key=lambda r: _doc_sort_key(r["id"]))
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        n_mentions = sum(len(r["mentions"]) for r in split_rows)
        print(f"wrote {out_path} ({len(split_rows)} docs, {n_mentions} mentions, {digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_ecbplus.py -q`
Expected: 4 passed

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green (do NOT run the download in tests)

```bash
git add scripts/fetch_ecbplus.py tests/scripts/test_fetch_ecbplus.py
git commit -m "feat: add ECB+ fetch/convert script"
```

---

### Task 9: Sweep configs + e2e tests

**Files:**
- Create: `configs/m3-ecbplus-sweep.toml`
- Create: `configs/m3-conel2-sweep.toml`
- Test: `tests/harness/test_m3_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6 by registered name; `run_experiment`,
  `ExperimentConfig` from `lattice.harness.runner`; `load_config` from
  `lattice.config.loader`; `SweepConfig`, `expand` from `lattice.harness.sweep`;
  the Task 5 mini fixtures.
- Produces: the two exit-criteria sweep configs; e2e regression tests.

- [ ] **Step 1: Write the sweep configs**

`configs/m3-ecbplus-sweep.toml`:

```toml
# M3 normalization sweep (spec §6): exact-label vs embedding-nn thresholds on
# ECB+ gold mentions. The gold-mentions extractor + block segmenter pairing is
# load-bearing: converters emit single-newline text -> one unit per document.

[base.segmenter]
name = "block"

[base.extractor]
name = "gold-mentions"
[base.extractor.params]
root = "data/ecbplus"
split = "test"

[base.scorer]
name = "passthrough"

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
name = "mention-clusters"
[base.dataset.params]
root = "data/ecbplus"
split = "test"

[[base.document_metrics]]
name = "clustering"

[axes]
resolver = [
  { name = "exact-label" },
  { name = "embedding-nn", params = { threshold = 0.65 } },
  { name = "embedding-nn", params = { threshold = 0.75 } },
  { name = "embedding-nn", params = { threshold = 0.80 } },
  { name = "embedding-nn", params = { threshold = 0.85 } },
  { name = "embedding-nn", params = { threshold = 0.90 } },
]
```

`configs/m3-conel2-sweep.toml`:

```toml
# M3 normalization sweep (spec §6): exact-label vs embedding-nn thresholds on
# ConEL-2 gold mentions. The gold-mentions extractor + block segmenter pairing
# is load-bearing: converters emit single-newline text -> one unit per document.

[base.segmenter]
name = "block"

[base.extractor]
name = "gold-mentions"
[base.extractor.params]
root = "data/conel2"
split = "test"

[base.scorer]
name = "passthrough"

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
name = "mention-clusters"
[base.dataset.params]
root = "data/conel2"
split = "test"

[[base.document_metrics]]
name = "clustering"

[axes]
resolver = [
  { name = "exact-label" },
  { name = "embedding-nn", params = { threshold = 0.65 } },
  { name = "embedding-nn", params = { threshold = 0.75 } },
  { name = "embedding-nn", params = { threshold = 0.80 } },
  { name = "embedding-nn", params = { threshold = 0.85 } },
  { name = "embedding-nn", params = { threshold = 0.90 } },
]
```

- [ ] **Step 2: Write the e2e tests**

`tests/harness/test_m3_e2e.py`:

```python
import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"
CONEL_ROOT = "tests/fixtures/mini_clusters_conel"
METRIC_KEYS = {"b3-precision", "b3-recall", "b3-f1", "ari"}


def _config(root: str, resolver: dict, embedder: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gold-mentions", "params": {"root": root}},
            "scorer": {"name": "passthrough"},
            "resolver": resolver,
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": embedder,
            "dataset": {"name": "mention-clusters", "params": {"root": root}},
            "document_metrics": [{"name": "clustering"}],
        }
    )


@pytest.mark.parametrize("root", [ECB_ROOT, CONEL_ROOT])
@pytest.mark.parametrize(
    "resolver",
    [{"name": "exact-label"}, {"name": "embedding-nn", "params": {"threshold": 0.8}}],
)
def test_m3_pipeline_pure(root, resolver):
    """Both corpora fixtures x both resolvers through the full harness with
    the hashing embedder — proves wiring without the ml stack."""
    report = run_experiment(_config(root, resolver, {"name": "hashing"}))
    assert report.errors == ()
    assert report.documents_processed == 3
    metrics = report.metrics["clustering"]
    assert set(metrics) == METRIC_KEYS
    assert all(-1.0 <= v <= 1.0 for v in metrics.values())


def test_mismatched_roots_fail_loudly():
    # Dataset from one corpus, sidecar from the other: the gold-mentions
    # extractor must refuse, not silently under-report.
    config = ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gold-mentions", "params": {"root": CONEL_ROOT}},
            "scorer": {"name": "passthrough"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {"name": "mention-clusters", "params": {"root": ECB_ROOT}},
            "document_metrics": [{"name": "clustering"}],
        }
    )
    with pytest.raises(ValueError, match="not in gold mention sidecar"):
        run_experiment(config)


def test_m3_run_is_reproducible():
    config = _config(
        ECB_ROOT, {"name": "embedding-nn", "params": {"threshold": 0.8}},
        {"name": "hashing"},
    )
    assert run_experiment(config) == run_experiment(config)


@pytest.mark.parametrize(
    "path", ["configs/m3-ecbplus-sweep.toml", "configs/m3-conel2-sweep.toml"]
)
def test_m3_sweep_configs_expand_to_six(path):
    sweep = load_config(path, model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 6
    assert [c.resolver.name for c in configs] == [
        "exact-label"] + ["embedding-nn"] * 5


@pytest.mark.ml
def test_m3_real_embedder_path():
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(
            _config(
                CONEL_ROOT,
                {"name": "embedding-nn", "params": {"threshold": 0.8}},
                {"name": "sentence-transformer"},
            )
        )
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert set(report.metrics["clustering"]) == METRIC_KEYS
```

- [ ] **Step 3: Run the new tests**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m3_e2e.py -q`
Expected: 9 passed (4 pure-parametrized + mismatch + reproducibility + 2 config + 1 ml;
the ml test must pass on this machine — models are cached)

- [ ] **Step 4: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add configs/m3-ecbplus-sweep.toml configs/m3-conel2-sweep.toml tests/harness/test_m3_e2e.py
git commit -m "feat: add M3 sweep configs and end-to-end tests"
```

---

## Exit criteria (orchestrator runs after Task 9 review clears)

1. Full suite + ruff:
   `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q && uv run --no-sync ruff check`
2. Fetch both corpora (real downloads, one-time):
   `uv run --no-sync python scripts/fetch_conel2.py` (expect 174/58/58 conversations)
   `uv run --no-sync python scripts/fetch_ecbplus.py` (expect several hundred docs per
   split; total mentions across splits ≈ 8,270 — the corpus has 8,274 entity mentions in
   validated sentences, minus 3 non-contiguous skips and any duplicate-span dedupes; a
   large deviation means a converter bug — STOP and report)
3. Verify conversion integrity (one-liner):
   `uv run --no-sync python -c "import json,sys; [ [sys.exit(f'bad span {m}') for m in r['mentions'] if r['text'][m['start']:m['end']] != m['surface']] for p in ('data/ecbplus/test.jsonl','data/conel2/test.jsonl') for r in map(json.loads, open(p))]; print('spans OK')"`
4. Run both sweeps (foreground Bash, 600000 timeout each):
   `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync python -m lattice.harness --sweep configs/m3-ecbplus-sweep.toml reports/m3-ecbplus`
   `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync python -m lattice.harness --sweep configs/m3-conel2-sweep.toml reports/m3-conel2`
5. Record both six-row tables in the ledger and check:
   - zero errors in every row;
   - **success criterion (spec §9): embedding-nn beats exact-label on b3-f1 at some
     threshold on at least one corpus** — if not, STOP and message the observer for
     adjudication (no tuning, no retries);
   - report the full threshold curve for both corpora (the shape is a deliverable).
6. Append "M3 tasks complete — awaiting observer final review" and stop. Do NOT merge.
