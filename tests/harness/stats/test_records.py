import pytest

from lattice.core.types import GraphSnapshot
from lattice.harness.stats.records import EvaluationContext, ResampleBundle, Resamplable


def test_evaluation_context_holds_the_three_inputs():
    ctx = EvaluationContext(deltas=(), snapshot=GraphSnapshot(concepts=(), relations=()), ground_truth={"k": 1})
    assert ctx.ground_truth == {"k": 1}
    assert ctx.snapshot.concepts == ()


def test_resample_bundle_drives_its_aggregate():
    bundle = ResampleBundle(
        kind="macro",
        per_document={"d1": {"f1": 1.0}, "d2": {"f1": 0.0}},
        aggregate=lambda records, ctx: {"f1": sum(r["f1"] for r in records) / len(records)},
    )
    picked = [bundle.per_document["d1"], bundle.per_document["d2"]]
    assert bundle.aggregate(picked, bundle.global_context) == {"f1": 0.5}
    assert bundle.global_context == {}


def test_resamplable_default_emit_raises():
    class M(Resamplable):
        pass
    assert M().kind == ""
    with pytest.raises(NotImplementedError):
        M().emit_records(EvaluationContext((), GraphSnapshot((), ()), {}))
