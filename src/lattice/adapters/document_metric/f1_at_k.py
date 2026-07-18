from collections.abc import Sequence

import snowballstemmer

from lattice.core.types import GraphDelta
from lattice.harness.stats.records import (
    EvaluationContext,
    Resamplable,
    ResampleBundle,
)
from lattice.ports import DocumentMetric
from lattice.registry.registry import register


@register(DocumentMetric, "f1-at-k")
class F1AtK(DocumentMetric, Resamplable):
    """Literature-standard keyphrase evaluation (M2 spec §6.5): per document,
    the top-k selected surfaces (salience desc, ties lexicographic) are
    compared to gold keyphrases as Snowball-stemmed exact phrase matches;
    precision/recall/F1 are macro-averaged over documents.

    Requires the scorer's top_k >= max(ks): rankings are computed over the
    scorer-selected mentions, so metrics at k beyond the scorer's selection
    depth saturate at the selection-depth value."""

    kind = "macro"

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

    def _per_document_scores(
        self, delta: GraphDelta, by_document: dict
    ) -> dict[str, float]:
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
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            scores[f"precision@{k}"] = precision
            scores[f"recall@{k}"] = recall
            scores[f"f1@{k}"] = f1
        return scores

    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]:
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
        """Precondition: only called after evaluate_documents() has validated the
        same inputs (run_experiment_detailed guarantees this ordering); assumes
        non-empty, ground-truth-complete input and does not re-validate."""
        by_document = context.ground_truth.get("keyphrases_by_document")
        if not isinstance(by_document, dict):
            raise ValueError('f1-at-k requires ground_truth["keyphrases_by_document"]')
        return ResampleBundle(
            kind="macro",
            per_document={
                d.document_id: self._per_document_scores(d, by_document)
                for d in context.deltas
            },
            aggregate=self._aggregate,
        )
