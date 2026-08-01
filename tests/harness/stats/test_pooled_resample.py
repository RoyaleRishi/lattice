"""Pooled-metric resampling: the tests the suite was missing.

Pooled metrics (`clustering`, `edge-f1`) pool mentions or edges *across*
documents before scoring, so their `_aggregate` is a degree-2 or set-valued
functional of the drawn document multiset, not a per-document mean. Two
consequences the old suite never checked:

1. Equivalence between `_aggregate` and the metric's own independent
   implementation was only ever asserted on the *identity* resample — the full
   document set in its original order — which is the one multiset that cannot
   fail. These tests assert it on strict subsets, the multisets the subsample
   scheme actually draws.
2. Duplicate documents corrupt both metrics, in opposite directions. That is
   why `bootstrap(..., scheme="subsample")` exists; the counterexamples below
   pin the corruption so the scheme cannot be quietly reverted.
"""

from statistics import mean

import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.adapters.metric.edge_f1 import EdgeF1
from lattice.core.types import (
    Concept,
    GraphDelta,
    GraphSnapshot,
    Mention,
    Relation,
    Resolution,
    ScoredMention,
)
from lattice.harness.stats.records import EvaluationContext, ResampleBundle
from lattice.harness.stats.resample import bootstrap

# --- fixture builders ---------------------------------------------------------


def _concept(cid: str, label: str) -> Concept:
    return Concept(id=cid, label=label, embedding=(1.0,), first_seen="d", updated_at="d")


def _resolution(doc_id: str, start: int, surface: str, concept_id: str) -> Resolution:
    return Resolution(
        concept=_concept(concept_id, surface),
        mention=ScoredMention(
            mention=Mention(
                surface=surface,
                unit_id=f"{doc_id}:u0",
                span=(start, start + 1),
                context=surface,
            ),
            salience=1.0,
            selected=True,
        ),
        is_new=True,
    )


def _clustering_delta(doc_id: str, mentions: list[tuple[int, str, str]]) -> GraphDelta:
    """mentions: (span start, surface, predicted concept id). The mention key
    the metric builds is f"{doc_id}:{start}-{start + 1}"."""
    return GraphDelta(
        document_id=doc_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=(),
        resolutions=tuple(_resolution(doc_id, s, surf, cid) for s, surf, cid in mentions),
    )


def _clustering_corpus() -> tuple[list[GraphDelta], dict[str, str]]:
    """Five documents with genuine cross-document structure: concept "shared"
    is predicted for one mention in every document (correctly — those mentions
    share gold cluster "G-shared"), and "over" over-merges one mention from
    docs A, C and E whose gold clusters are distinct. Both directions of error
    are present, so a subset's score genuinely differs from the full corpus's."""
    spec = {
        "A": [(0, "shared", "c-shared"), (2, "alpha", "c-over"), (4, "a3", "c-a3")],
        "B": [(0, "shared", "c-shared"), (2, "b2", "c-b2")],
        "C": [(0, "shared", "c-shared"), (2, "gamma", "c-over"), (4, "c3", "c-c3")],
        "D": [(0, "shared", "c-shared"), (2, "d2", "c-d2"), (4, "d3", "c-d3")],
        "E": [(0, "shared", "c-shared"), (2, "epsilon", "c-over")],
    }
    gold = {
        "A:0-1": "G-shared", "A:2-3": "G-a2", "A:4-5": "G-a3",
        "B:0-1": "G-shared", "B:2-3": "G-b2",
        "C:0-1": "G-shared", "C:2-3": "G-c2", "C:4-5": "G-c3",
        "D:0-1": "G-shared", "D:2-3": "G-d2", "D:4-5": "G-d3",
        "E:0-1": "G-shared", "E:2-3": "G-e2",
    }
    return [_clustering_delta(d, m) for d, m in spec.items()], gold


def _edge_delta(doc_id: str, edges: list[tuple[str, str]]) -> GraphDelta:
    return GraphDelta(
        document_id=doc_id,
        concepts_added=(),
        concepts_updated=(),
        relations_added=tuple(
            Relation(
                type="IS_A",
                source_id=f"c:{hypo}",
                target_id=f"c:{hyper}",
                confidence=1.0,
                provenance=doc_id,
            )
            for hypo, hyper in edges
        ),
    )


def _edge_corpus() -> tuple[list[GraphDelta], tuple[Concept, ...], dict[str, object]]:
    """Four documents. "olive oil -> oil" is proposed by two of them (so the
    union genuinely collapses a duplicate edge), one edge is wrong, and one
    gold edge is never proposed — precision, recall and both counts all move
    when a document is dropped."""
    spec = {
        "W": [("olive oil", "oil"), ("oil", "food")],
        "X": [("olive oil", "oil"), ("butter", "fat")],
        "Y": [("fat", "food"), ("cheese", "mineral")],
        "Z": [("milk", "dairy")],
    }
    labels = [
        "olive oil", "oil", "food", "butter", "fat", "cheese", "mineral",
        "milk", "dairy", "yoghurt",
    ]
    concepts = tuple(_concept(f"c:{label}", label) for label in labels)
    ground_truth = {
        "is_a_edges": [
            ("olive oil", "oil"), ("oil", "food"), ("butter", "fat"),
            ("fat", "food"), ("milk", "dairy"), ("yoghurt", "dairy"),
        ]
    }
    return [_edge_delta(d, e) for d, e in spec.items()], concepts, ground_truth


# --- 1. equivalence on NON-identity document multisets ------------------------

SUBSETS = [["A", "C", "E"], ["B", "D"], ["E", "A"], ["A", "B", "C", "D"], ["C"]]


@pytest.mark.parametrize("subset", SUBSETS)
def test_clustering_aggregate_matches_direct_recomputation_on_a_strict_subset(subset):
    # The regression test the project lacked: _aggregate over a resampled
    # document list must agree with ClusteringMetric's own evaluate_documents()
    # over exactly those documents. Asserting this only on the full set (as
    # test_clustering_resample.py did) exercises the one multiset for which the
    # index re-keying is guaranteed to be a no-op.
    deltas, gold = _clustering_corpus()
    by_id = {d.document_id: d for d in deltas}
    metric = ClusteringMetric()
    bundle = metric.emit_records(
        EvaluationContext(tuple(deltas), GraphSnapshot((), ()), {"clusters_by_mention": gold})
    )
    records = [bundle.per_document[d] for d in subset]
    sub_gold = {k: v for k, v in gold.items() if k.split(":")[0] in subset}
    direct = metric.evaluate_documents(
        [by_id[d] for d in subset], {"clusters_by_mention": sub_gold}
    )
    assert bundle.aggregate(records, bundle.global_context) == direct


def test_clustering_subset_scores_actually_differ_from_the_full_corpus():
    # Guards the guard: if every subset happened to score identically, the
    # equivalence test above would pass on a broken _aggregate too.
    deltas, gold = _clustering_corpus()
    metric = ClusteringMetric()
    bundle = metric.emit_records(
        EvaluationContext(tuple(deltas), GraphSnapshot((), ()), {"clusters_by_mention": gold})
    )
    full = bundle.aggregate(list(bundle.per_document.values()), bundle.global_context)
    scores = {
        tuple(subset): bundle.aggregate(
            [bundle.per_document[d] for d in subset], bundle.global_context
        )["b3-f1"]
        for subset in SUBSETS
    }
    assert len({*scores.values(), full["b3-f1"]}) > 1


EDGE_SUBSETS = [["W", "Y"], ["X", "Z"], ["W", "X"], ["Y", "Z", "W"], ["Z"]]


@pytest.mark.parametrize("subset", EDGE_SUBSETS)
def test_edge_f1_aggregate_matches_direct_recomputation_on_a_strict_subset(subset):
    deltas, concepts, ground_truth = _edge_corpus()
    by_id = {d.document_id: d for d in deltas}
    metric = EdgeF1()
    snapshot = GraphSnapshot(
        concepts=concepts,
        relations=tuple(r for d in deltas for r in d.relations_added),
    )
    bundle = metric.emit_records(EvaluationContext(tuple(deltas), snapshot, ground_truth))
    records = [bundle.per_document[d] for d in subset]
    sub_snapshot = GraphSnapshot(
        concepts=concepts,
        relations=tuple(r for d in subset for r in by_id[d].relations_added),
    )
    assert bundle.aggregate(records, bundle.global_context) == metric.evaluate(
        sub_snapshot, ground_truth
    )


def test_edge_f1_subset_scores_actually_differ_from_the_full_corpus():
    deltas, concepts, ground_truth = _edge_corpus()
    metric = EdgeF1()
    snapshot = GraphSnapshot(
        concepts=concepts,
        relations=tuple(r for d in deltas for r in d.relations_added),
    )
    bundle = metric.emit_records(EvaluationContext(tuple(deltas), snapshot, ground_truth))
    full = bundle.aggregate(list(bundle.per_document.values()), bundle.global_context)
    scores = {
        tuple(subset): bundle.aggregate(
            [bundle.per_document[d] for d in subset], bundle.global_context
        )["f1"]
        for subset in EDGE_SUBSETS
    }
    assert len({*scores.values(), full["f1"]}) > 1


# --- 2. why with-replacement draws are wrong for pooled metrics ---------------


def test_duplicated_document_makes_a_cross_document_error_vanish():
    """The verified two-document counterexample. Documents A and B hold one
    mention each, in separate gold clusters, over-merged by the prediction into
    a single concept — a pure cross-document error. Duplicating either document
    hands _aggregate two instances that share both the predicted concept id and
    the gold cluster id, so they agree with each other by construction and the
    error disappears: 0.6667 -> 1.0000 on b3-f1, 0.0 -> 1.0 on ARI. Averaging
    the four equally likely n=2 draws gives an upward bias of +1/6 on b3-f1 and
    +1/2 on ARI, and the distinct-document fraction converges to 1 - 1/e, so
    this does not shrink with n."""
    a = [("A:0-1", "MERGED", "G-a")]
    b = [("B:0-1", "MERGED", "G-b")]
    agg = ClusteringMetric._aggregate

    point = agg([a, b], {})
    assert point["b3-f1"] == pytest.approx(2 / 3)
    assert point["ari"] == 0.0

    for duplicated in ([a, a], [b, b]):
        vanished = agg(duplicated, {})
        assert vanished["b3-f1"] == 1.0
        assert vanished["ari"] == 1.0

    draws = [agg(d, {}) for d in ([a, a], [a, b], [b, a], [b, b])]
    assert mean(d["b3-f1"] for d in draws) == pytest.approx(5 / 6)   # 2/3 + 1/6
    assert mean(d["ari"] for d in draws) == pytest.approx(0.5)       # 0.0 + 1/2


def test_edge_f1_aggregate_is_blind_to_multiplicity():
    """The mirror-image defect. edge-f1's _aggregate unions per-document edge
    sets, so a document drawn twice contributes exactly as much as a document
    drawn once — a with-replacement replicate silently evaluates only the
    ~63.2% of the corpus it happens to touch, which deflates predicted_edges
    and recall rather than inflating them."""
    ctx = {"gold": frozenset({("a", "b"), ("c", "d"), ("e", "f")})}
    w = frozenset({("a", "b")})
    x = frozenset({("c", "d")})
    once = EdgeF1._aggregate([w, x], ctx)
    twice = EdgeF1._aggregate([w, w, w, x], ctx)
    assert once == twice
    assert once["predicted_edges"] == 2.0
    # dropping a document is what actually moves the metric, so a replicate
    # that duplicates one document has really only measured the rest.
    assert EdgeF1._aggregate([w, w, w, w], ctx)["predicted_edges"] == 1.0


# --- 3. the four-document counterexample under each scheme --------------------


def _over_merge_bundle() -> ResampleBundle:
    """Four documents, four mentions each. Three mentions per document are
    predicted correctly as their own singleton concept; the fourth is swept
    into one corpus-wide concept "MERGED" whose four members belong to four
    distinct gold clusters — the same pure cross-document over-merge as the
    two-document counterexample, sized so the subsample floor of m = 2 applies.

    Point estimate: b3-precision = (12 * 1 + 4 * 1/4) / 16 = 13/16 = 0.8125,
    b3-recall = 1.0, b3-f1 = 0.896552, ARI = 0.0 (the predicted partition adds
    C(4, 2) = 6 pairs, none of which the all-singleton gold agrees with).
    """
    per_document = {
        d: [
            (f"{d}:0-1", f"c-{d}-1", f"G-{d}-1"),
            (f"{d}:2-3", f"c-{d}-2", f"G-{d}-2"),
            (f"{d}:4-5", f"c-{d}-3", f"G-{d}-3"),
            (f"{d}:6-7", "MERGED", f"G-{d}-4"),
        ]
        for d in ("A", "B", "C", "D")
    }
    return ResampleBundle(
        kind="pooled", per_document=per_document, aggregate=ClusteringMetric._aggregate
    )


POINT_B3_F1 = 0.896551724137931
# Stated tolerance for the pooled draw mean. The only bias the subsample scheme
# may show is the intrinsic m-out-of-n scale effect: a size-2 replicate holds a
# MERGED cluster of 2 rather than 4, so those mentions score 1/2 instead of 1/4
# and b3-precision moves 13/16 -> 7/8, i.e. b3-f1 0.896552 -> 0.933333, +0.0368.
# That is what the subsample rescaling exists to pay for. Duplication bias, which
# is unbounded in this fixture (a replicate can reach a *perfect* 1.0 on a
# corpus containing a real error), must be absent entirely.
B3_F1_TOLERANCE = 0.05


def test_subsample_scheme_is_not_biased_upward_by_duplication():
    out = bootstrap(_over_merge_bundle(), samples=1000, seed=0, scheme="subsample")
    assert (out.m, out.n) == (2, 4)
    # ARI is the sharp detector: the over-merge survives every subsample
    # unchanged, so every draw reproduces the point estimate exactly.
    assert set(out.draws["ari"]) == {0.0}
    assert mean(out.draws["ari"]) == 0.0
    # b3-f1 drifts up only by the stated scale effect, and no replicate ever
    # scores the corpus perfect.
    assert mean(out.draws["b3-f1"]) - POINT_B3_F1 == pytest.approx(0.036782, abs=1e-5)
    assert mean(out.draws["b3-f1"]) - POINT_B3_F1 < B3_F1_TOLERANCE
    assert max(out.draws["b3-f1"]) < 1.0


def test_resample_scheme_would_be_biased_upward_on_the_same_fixture():
    # Not a defence of the old behaviour — this pins the defect the subsample
    # scheme removes, so reverting report.py's kind-based selection fails here.
    out = bootstrap(_over_merge_bundle(), samples=1000, seed=0, scheme="resample")
    assert (out.m, out.n) == (4, 4)
    assert mean(out.draws["ari"]) > 0.5          # point estimate is 0.0
    assert max(out.draws["ari"]) == 1.0          # duplicate-only draws score perfect
    assert max(out.draws["b3-f1"]) == 1.0
