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
