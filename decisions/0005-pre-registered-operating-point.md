---
status: proposed
date: 2026-10-05
decision-makers: Rishi Srikaanth
---

# Report the pre-registered operating point, not the best point on a threshold grid

Retroactive. Recorded 2026-10-05 for the rule in the statistical-intervals spec §6
(2026-07-14), applied to the README in `b7d127c` (2026-08-01).

## Context and Problem Statement

The resolver threshold is swept over a grid (0.65–0.95, step 0.05; stats spec :168). The
best grid point for b3-f1 is not the shipped one:

- ConEL-2: peak 0.9620 at 0.80; nn@0.90 scores 0.9491; exact-label 0.9386. The peak is
  below stemmed-label's 0.9637 (post-fix.md :522-542).
- ECB+: peak 0.6428 at 0.75; nn@0.90 scores 0.6255; exact-label 0.6084 (post-fix.md
  :534, :552-554).

Earlier README versions quoted the grid maxima (0.962, 0.643); the current README retracts
them as selection bias (README :180-186).

Which point do published comparisons use?

## Decision Drivers

- Published claims must not be inflated by picking the best of many evaluations.
- The threshold should be fixed before the analysis that reports on it.
- The sensitivity curve should still be shown, so readers can see 0.90 is "not a fragile
  peak" (stats spec :13-14).

## Considered Options

- **A. Report at the pre-registered point (nn@0.90); show the full curve for sensitivity.**
- **B. Report the grid maximum.**

## Decision Outcome

Chosen option: **A**.

Stated reason: comparing against the maximum over thresholds "would be selection bias; the
defensible claim uses the threshold we committed to before this analysis" (stats spec
§6 :184-186). 0.90 was chosen for M5's redundancy↔coherence tradeoff, a different objective
from b3-f1 (:187-189; `engine.py:37-39`).

**Y-statement:** In the context of resolver comparisons swept over a threshold grid, facing
grid maxima that sit away from the shipped threshold, we decided to report at the
pre-registered nn@0.90 and show the curve only as sensitivity, to achieve claims free of
selection bias, accepting lower headline numbers than the grid best.

### Consequences

- Good: headline numbers are not selected post hoc.
- Good: ECB+ runs at the same point act as an out-of-sample replication check
  (`configs/m3-ecbplus-exact.toml:5`).
- Bad: lower headline b3-f1 than the grid best on both corpora.
- Bad: enforced only by convention. The threshold lives in config comments
  (`configs/*-nn090.toml`), `engine.py:57`, and two tests
  (`tests/harness/test_m5_cross_corpus_configs.py:75`,
  `tests/scripts/test_interval_analysis.py:46`). Nothing in `src/` stops a report from
  using a grid maximum.
- Scope: the spec names only one pre-registered point (resolver nn@0.90). No scorer or K is
  pre-registered.
- Follow-up: spec audit T2 (selection bias), STAT-6 (M3 §9 success criterion never
  amended), STAT-7 ("beats = CI excludes 0" vs the Holm rule in §8).

## Open questions

- **Is 0.90 actually pre-registered?** Spec audit STAT-5 (high): 0.90 was chosen in M5 on
  the same 58 ConEL-2 test conversations, using gold-free metrics, and the redundancy
  evidence it was chosen on was vacuous (`37e2619`). The audit concludes it "is not
  pre-registered". The audit routes this to an ADR (spec audit :209).
- **PromptRank exception.** The PromptRank comparison picks its incumbent as "the strongest
  included scorer at f1@15" at run time (promptrank spec :137-140, :157-158), an argmax that
  conflicts with §6 (spec audit PR-1).
- Rationale not recorded — confirm: why a gold-free choice on the test set counts as
  pre-registration.
- Rationale not recorded — confirm: policy for arms other than the resolver, and for grids
  sparser than §5 (the M5 sweep has 3 points, STAT-10).

## What I learned

