from collections.abc import Sequence
from itertools import combinations

from lattice.core.types import GraphDelta
from lattice.core.vectors import cosine
from lattice.harness.stats.records import Resamplable
from lattice.ports import DocumentMetric, Embedder
from lattice.registry.registry import register


@register(DocumentMetric, "coherence")
class Coherence(DocumentMetric, Resamplable):
    """Intrinsic merge quality (M5 spec §4.2): what the resolver wrongly
    merged — the counterweight to the redundancy metric. For each concept
    that accumulated >= 2 distinct casefolded mention surfaces across the
    run, coherence is the mean pairwise cosine of the surface embeddings;
    the reported value is the mean over those concepts, and 1.0 when there
    are none (vacuous perfection, made visible by multi-surface-concepts).
    Ground truth is ignored — this metric is intrinsic (spec §7 explains
    why it does not join DocumentMetricContract). One batched embed call
    covers every distinct surface in the run."""

    kind = "holistic"

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
