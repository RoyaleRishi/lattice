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
