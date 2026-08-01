from lattice.core.types import GraphSnapshot
from lattice.harness.stats.records import (
    EvaluationContext,
    Resamplable,
    ResampleBundle,
)
from lattice.ports import Metric
from lattice.registry.registry import register


@register(Metric, "edge-f1")
class EdgeF1(Metric, Resamplable):
    """Edge precision/recall/F1 of the snapshot's deduped IS_A edges —
    expressed as (hyponym label, hypernym label) pairs — against
    ground_truth["is_a_edges"] (M4 spec §4.6; TExEval-2 task paper §4.3).
    Direction matters. predicted_edges/gold_edges counts are returned for
    diagnosis (floats, like every metric value).

    NOT INTERVAL-ESTIMABLE BY DOCUMENT RESAMPLING. This metric is `pooled`, so
    the harness draws it without replacement (see resample.py) — that removes
    the multiplicity collapse in _aggregate, but it does not make the resulting
    band a confidence interval, and no resampling scheme would. edge-f1 is a
    *size-dependent* functional of the document set, on both sides of every
    ratio:

    - The prediction side is a set union over documents (_aggregate unions the
      per-document edge frozensets), so predicted_edges is monotone increasing
      in the number of *distinct* documents in the sample and grows sublinearly
      — a half-corpus draw recovers strictly fewer edges than the full corpus.
    - The recall denominator is the fixed corpus-level gold taxonomy (1587
      edges on TExEval food), which does not shrink with the sample. Recall is
      therefore monotone in corpus size too, and predicted_edges is an unbounded
      raw count rather than a converging statistic.

    So theta(m) != theta(n) systematically for m < n: each key's draws sit on
    one fixed side of the point estimate and never straddle it, at any m, under
    with-replacement and without-replacement draws alike.

    THE SIDE IS NOT THE SAME FOR EVERY KEY, and the two bullets above do not
    determine it for all of them. They fix the direction for predicted_edges
    and recall — both shrink with the sample — and f1 follows recall down.
    Precision is the exception: its numerator and its denominator both shrink,
    so the union argument leaves its direction open, and measurement puts it
    the other way. On M4 food (m=656, n=1311) precision's band sits *above* its
    estimate of 0.6148, while f1 (estimate 0.3233), recall and predicted_edges
    all sit below theirs. Do not restate this paragraph as "the draws sit below
    the estimate": what is universal here is exclusion, not direction.

    The subsample rescaling corrects a variance *rate*; it cannot correct a
    shift in the estimand — and at the harness's m = round(n / 2) it barely
    rescales at all, since the corrected tau = sqrt(m / (n - m)) is ~1.0008
    there. Every key except the constant gold_edges excludes its own estimate.
    The band endpoints once quoted here were computed under the pre-2026-08-01
    sqrt(m / n) = 0.707 and are dropped rather than recomputed by hand; read
    the current ones from a regenerated reports/intervals/m4-food/. That
    exclusion — and each key's side — survives the change of factor by
    construction rather than by luck: the rescaling is about the estimate, so
    sign(lo - estimate) is invariant in tau for every tau > 0 and no change of
    factor can carry a band across an estimate it excludes. Which is why the
    sides above could be stated from the pre-10c artifact without regenerating
    it, while the endpoints could not.

    Read edge-f1's emitted band as a corpus-size sensitivity range, not a
    confidence interval. The report's `brackets_estimate` flag is False for
    exactly this reason and is the machine-readable form of this paragraph.
    """

    kind = "pooled"

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        label_of = {concept.id: concept.label.lower() for concept in snapshot.concepts}
        predicted = {
            (label_of[relation.source_id], label_of[relation.target_id])
            for relation in snapshot.relations
            if relation.type == "IS_A"
        }
        gold = {
            (str(hypo).lower(), str(hyper).lower())
            for hypo, hyper in ground_truth.get("is_a_edges", [])
        }
        true_positives = len(predicted & gold)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_edges": float(len(predicted)),
            "gold_edges": float(len(gold)),
        }

    @staticmethod
    def _aggregate(records: list, ctx: dict) -> dict[str, float]:
        predicted: set = set()
        for record in records:
            predicted |= record
        gold = ctx["gold"]
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_edges": float(len(predicted)),
            "gold_edges": float(len(gold)),
        }

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        """Precondition: only called after evaluate() has validated the same
        inputs (run_experiment_detailed guarantees this ordering); assumes
        ground-truth-complete input and does not re-validate."""
        label_of = {concept.id: concept.label.lower()
                    for concept in context.snapshot.concepts}
        per_document = {
            delta.document_id: frozenset(
                (label_of[r.source_id], label_of[r.target_id])
                for r in delta.relations_added
                if r.type == "IS_A"
            )
            for delta in context.deltas
        }
        gold = frozenset(
            (str(hypo).lower(), str(hyper).lower())
            for hypo, hyper in context.ground_truth.get("is_a_edges", [])
        )
        return ResampleBundle(
            kind="pooled",
            per_document=per_document,
            aggregate=self._aggregate,
            global_context={"gold": gold},
        )
