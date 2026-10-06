---
status: proposed
date: 2026-10-05
decision-makers: Rishi Srikaanth
---

# F1@K dedupes predictions by Snowball stem before truncating to K

Retroactive. Recorded 2026-10-05 for the fix in `5ac23e2` (2026-08-01), which changed the
metric added in `3aa5f04` (2026-07-08).

## Context and Problem Statement

F1@K scores the scorer's ranked keyphrases against gold, both Snowball-stemmed token-wise.
The M2 spec §6.5 (:119-122) defines predicted as "the top *k* unique surfaces … ranked by
salience descending, ties lexicographic", then stemmed for matching. It does not say whether
to dedupe before or after stemming (spec audit M2-2).

The original implementation (`3aa5f04`) truncated by unique surface, then stemmed. When two
top-k surfaces stem alike (e.g. "neural network" / "neural networks"), "a slot is consumed
but the denominator shrinks — inflating precision" (review-findings Task 5, :300-340).

Should F1@K dedupe by surface or by stem, and before or after the top-K cut?

## Decision Drivers

- Precision must not be inflated by near-duplicate predictions.
- Absolute numbers should be comparable to published baseline tables used as sanity
  checks.
- Headline and bootstrap numbers should come from one code path.

## Considered Options

- **A. Dedupe by surface, truncate to K, then stem** (original; matches M2 §6.5 wording).
- **B. Dedupe by stem, then truncate to K.**

## Decision Outcome

Chosen option: **B**.

`f1_at_k.py`: predictions are grouped by stem, keeping the highest salience per stem (:51);
ties sort by (−salience, stem) (:53); the top K stems are taken after dedupe (:64);
precision = tp / len(predicted) (:66), i.e. divides by min(k, n_distinct_stems). Gold is a
set of stemmed phrases (:60).

Stated reasons: it removes the precision inflation, and "dedupe-then-truncate matches the
literature protocol (e.g. HCUKE §4.1), keeping reported numbers comparable to published
baseline tables" (`f1_at_k.py:24-26`; `5ac23e2`). Both evaluate paths share
`_per_document_scores` (`5ac23e2`). A was rejected because of the inflation above.

**Y-statement:** In the context of keyphrase evaluation with stemmed matching, facing
near-duplicate predictions that consumed top-K slots while shrinking the precision
denominator, we decided to dedupe by stem before truncating to K, to achieve precision
comparable to the literature protocol, accepting a metric definition that departs from the
M2 spec's wording.

### Consequences

- Good: no precision inflation from stem collisions.
- Good: matches HCUKE §4.1, so absolutes can be checked against published bands.
- Neutral: system ranking unchanged ("does not reorder systems", review-findings Task 5).
  f1@5 0.2967 → 0.2986; f1@10 0.3547 → 0.3544; f1@15 unchanged at 0.3555
  (post-fix.md §2.1 :80-84, §2.3).
- Bad: requires the scorer's `top_k ≥ max(ks)`, or F1@K saturates (`f1_at_k.py:29-31`).
  The default `top_k` is 10 against `ks` [5, 10, 15], so F1@15 is scored on at most 10
  predictions (spec audit M2-1).
- Bad: the M2 spec §6.5 still describes option A.
- Test: `tests/adapters/test_f1_at_k.py:38`
  (`test_stem_collision_denominator_dedupes_before_truncating`) and :22.
- Follow-up: amend M2 §6.5 to describe B (spec audit T4, M2-2); resolve M2-1.

## Open questions

- **PromptRank at k = 15.** Whether PromptRank's k = 15 saturates the same way is
  unmeasured (promptrank-baseline.md :18-24). The README notes that post-fix.md §8
  inverts the f1@15 sign reading (README :319-335).
- Rationale not recorded — confirm: Snowball rather than Porter stemmer.
- Rationale not recorded — confirm: keep max salience per stem, rather than first-seen.
- Rationale not recorded — confirm: gold-side dedupe policy (currently implicit via set).
- Rationale not recorded — confirm: whether this matches the PromptRank paper's own
  evaluation protocol.

## What I learned

