"""Order-permutation robustness. Runs the pipeline over K seeded document
orderings and reports the spread of each metric. A robustness report, not a
hypothesis test; for the glossary-first M4 protocol use fixed_prefix=1 so the
term-inventory document stays first and only the corpus order varies."""

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import pstdev

from lattice.config.factory import instantiate
from lattice.harness.runner import ExperimentConfig, run_on_documents
from lattice.ports import Dataset


@dataclass(frozen=True)
class SpreadResult:
    values: list[float]
    min: float
    max: float
    range: float
    std: float


def order_spread(
    config: ExperimentConfig, *, permutations: int, seed: int, fixed_prefix: int = 0
) -> dict[str, SpreadResult]:
    documents = list(instantiate(Dataset, config.dataset).documents())
    fixed = documents[:fixed_prefix]
    pool = documents[fixed_prefix:]
    rng = random.Random(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(permutations):
        order = pool[:]
        rng.shuffle(order)
        for key, value in run_on_documents(config, fixed + order).items():
            collected[key].append(value)
    return {
        key: SpreadResult(
            values=values, min=min(values), max=max(values),
            range=max(values) - min(values), std=pstdev(values),
        )
        for key, values in collected.items()
    }
