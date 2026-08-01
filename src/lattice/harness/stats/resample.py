"""The resampling engine. Item-level draws drive a bundle's own aggregate()
over redrawn document ids; holistic bootstrap re-runs the pipeline per
resample. One seeded Random throughout.

Two item-level schemes, chosen by the caller (report.py picks by metric kind):

- "resample" — the classical n-out-of-n bootstrap, with replacement. Correct
  for *macro* metrics, whose aggregate is a plain per-document mean: a
  degree-1 functional of exchangeable documents, for which duplicate draws are
  exactly the point of the method.
- "subsample" — m-out-of-n, *without* replacement (Politis, Romano & Wolf,
  "Subsampling", 1999). Required for *pooled* metrics, which pool mentions or
  edges across documents before scoring and are therefore degree-2 or
  set-valued. A duplicated document is re-keyed into two instances that carry
  the same predicted concept id and the same gold cluster id, so they agree
  with each other by construction in both partitions and inflate B³/ARI;
  edge-F1's per-document union goes the other way and drops the multiplicity
  entirely, evaluating only the ~63.2% of documents that a with-replacement
  draw happens to touch. Drawing without replacement removes duplication at
  the source. The price — a size-m replicate is more variable than the full
  sample — is paid by intervals.subsample_interval()'s rate correction, not
  here. That factor is sqrt(m / (n - m)), not sqrt(m / n): the subsample
  overlaps the full sample in m of its n documents and the finite-population
  term cannot be dropped at the m = subsample_size(n) drawn below. See
  intervals._subsample_tau.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from lattice.config.factory import instantiate
from lattice.harness.runner import ExperimentConfig, run_on_documents
from lattice.harness.stats.records import ResampleBundle
from lattice.ports import Dataset

Scheme = Literal["resample", "subsample"]
SCHEMES: tuple[Scheme, ...] = ("resample", "subsample")


@dataclass(frozen=True)
class BootstrapDraws:
    """Per-key draw distributions plus the scheme that produced them.

    `n` is the size of the sampled pool (fixed documents are excluded from it);
    `m` is how many pool documents each replicate draws — `m == n` under
    "resample", `subsample_size(n)` under "subsample". Interval construction
    needs both to apply the subsampling rate correction, and the report carries
    them so a reader can tell which construction produced an interval.
    """

    draws: dict[str, list[float]]
    scheme: Scheme
    m: int
    n: int


def subsample_size(n: int) -> int:
    """m for m-out-of-n subsampling: half the pool, floored at 2 so every
    replicate still contains a pair (B³ and ARI are pairwise functionals; a
    one-document replicate scores a trivial partition), and capped at n so a
    pool smaller than that floor is still drawable — for n <= 2 every draw is
    the whole pool, which honestly reports zero resampling variability rather
    than inventing some."""
    if n <= 0:
        return 0
    return min(n, max(2, round(n / 2)))


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
    scheme: Scheme = "resample",
) -> BootstrapDraws:
    """Draw `samples` replicate document lists and re-aggregate each one. See
    the module docstring for why pooled metrics must not use "resample".

    `fixed_doc_ids` are present in every replicate and never enter the sampled
    pool, under either scheme.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {list(SCHEMES)}")
    rng = random.Random(seed)
    doc_ids = list(bundle.per_document)
    fixed, pool = _split(doc_ids, fixed_doc_ids)
    n = len(pool)
    m = n if scheme == "resample" else subsample_size(n)
    out: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        if not n:
            drawn = fixed
        elif scheme == "resample":
            drawn = fixed + [pool[rng.randrange(n)] for _ in range(n)]
        else:
            drawn = fixed + rng.sample(pool, m)
        result = bundle.aggregate(
            [bundle.per_document[d] for d in drawn], bundle.global_context
        )
        for key, value in result.items():
            out[key].append(value)
    return BootstrapDraws(draws=dict(out), scheme=scheme, m=m, n=n)


def bootstrap_holistic(
    config: ExperimentConfig,
    *,
    samples: int,
    seed: int,
    fixed_prefix: int = 0,
) -> dict[str, list[float]]:
    """Holistic bootstrap: re-runs the whole pipeline per replicate. Always
    n-out-of-n with replacement, and returns a bare distribution dict rather
    than a BootstrapDraws — here a duplicated document is genuinely re-processed
    by the pipeline rather than re-keyed by an aggregate, so whether it needs
    the same treatment as the pooled item-level path is a separate question and
    deliberately out of this function's scope."""
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
