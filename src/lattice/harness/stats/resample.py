"""The resampling engine. Item-level bootstrap drives a bundle's own
aggregate() over resampled document ids (multiplicity-preserving); holistic
bootstrap re-runs the pipeline per resample. One seeded Random throughout."""

import random
from collections import defaultdict
from collections.abc import Sequence

from lattice.config.factory import instantiate
from lattice.harness.runner import ExperimentConfig, run_on_documents
from lattice.harness.stats.records import ResampleBundle
from lattice.ports import Dataset


def _split(
    doc_ids: list[str], fixed_doc_ids: Sequence[str]
) -> tuple[list[str], list[str]]:
    fixed_set = set(fixed_doc_ids)
    fixed = [d for d in doc_ids if d in fixed_set]
    pool = [d for d in doc_ids if d not in fixed_set]
    return fixed, pool


def jackknife(
    bundle: ResampleBundle, fixed_doc_ids: Sequence[str] = ()
) -> dict[str, list[float]]:
    doc_ids = list(bundle.per_document)
    fixed, pool = _split(doc_ids, fixed_doc_ids)
    out: dict[str, list[float]] = defaultdict(list)
    for i in range(len(pool)):
        kept = fixed + pool[:i] + pool[i + 1 :]
        result = bundle.aggregate(
            [bundle.per_document[d] for d in kept], bundle.global_context
        )
        for key, value in result.items():
            out[key].append(value)
    return dict(out)


def bootstrap(
    bundle: ResampleBundle,
    *,
    samples: int,
    seed: int,
    fixed_doc_ids: Sequence[str] = (),
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    doc_ids = list(bundle.per_document)
    fixed, pool = _split(doc_ids, fixed_doc_ids)
    n = len(pool)
    out: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = fixed + [pool[rng.randrange(n)] for _ in range(n)] if n else fixed
        result = bundle.aggregate(
            [bundle.per_document[d] for d in drawn], bundle.global_context
        )
        for key, value in result.items():
            out[key].append(value)
    return dict(out)


def bootstrap_holistic(
    config: ExperimentConfig,
    *,
    samples: int,
    seed: int,
    fixed_prefix: int = 0,
) -> dict[str, list[float]]:
    documents = list(instantiate(Dataset, config.dataset).documents())
    fixed = documents[:fixed_prefix]
    pool = documents[fixed_prefix:]
    n = len(pool)
    rng = random.Random(seed)
    out: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = fixed + [pool[rng.randrange(n)] for _ in range(n)] if n else fixed
        for key, value in run_on_documents(config, drawn).items():
            out[key].append(value)
    return dict(out)
