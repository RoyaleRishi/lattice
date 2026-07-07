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
