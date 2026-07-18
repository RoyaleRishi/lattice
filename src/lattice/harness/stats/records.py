"""Per-document detail capture for resampling. A resamplable metric declares
its `kind` and emits a ResampleBundle whose `aggregate` recomputes the metric
from a (possibly multiplicity-bearing) list of per-document records — proven
equal to the metric's own evaluate() on the full document set."""

from collections.abc import Callable
from dataclasses import dataclass, field

from lattice.core.types import GraphDelta, GraphSnapshot


@dataclass(frozen=True)
class EvaluationContext:
    deltas: tuple[GraphDelta, ...]
    snapshot: GraphSnapshot
    ground_truth: dict[str, object]


@dataclass(frozen=True)
class ResampleBundle:
    kind: str
    per_document: dict[str, object]
    aggregate: Callable[[list, dict], dict[str, float]]
    global_context: dict = field(default_factory=dict)


class Resamplable:
    """Opt-in marker. `kind` is "macro" | "pooled" | "holistic". macro/pooled
    metrics implement emit_records; holistic metrics do not (the engine
    re-runs the pipeline for them)."""

    kind: str = ""

    def emit_records(self, context: EvaluationContext) -> ResampleBundle:
        raise NotImplementedError
