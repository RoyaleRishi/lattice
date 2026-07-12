from lattice.core.types import GraphSnapshot
from lattice.ports import Metric
from lattice.registry.registry import register


@register(Metric, "edge-f1")
class EdgeF1(Metric):
    """Edge precision/recall/F1 of the snapshot's deduped IS_A edges —
    expressed as (hyponym label, hypernym label) pairs — against
    ground_truth["is_a_edges"] (M4 spec §4.6; TExEval-2 task paper §4.3).
    Direction matters. predicted_edges/gold_edges counts are returned for
    diagnosis (floats, like every metric value)."""

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
