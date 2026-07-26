# M5 Cross-Corpus Validation — Is the redundancy/coherence tradeoff a method property?

Track 2 deliverable. Runs the M5 intrinsic harness on **MultiWOZ** (new
transcript corpus, 200 dialogues) and **ECB+** (cross-genre, 206 news docs) and
compares against **ConEL-2** (58 conversations, the original M5 corpus). No
number is estimated — each was produced by the recorded command (§5) against
this repository's checked-in code and fetched data. Runs executed 2026-07-26,
seed 0 throughout.

## 1. Headline verdict

**The redundancy/coherence tradeoff is a property of the method, not an artifact
of ConEL-2.** Across three corpora spanning three genres — conversational
entity-linking, task-oriented dialogue, and news — the coherence-vs-threshold
curve is nearly identical (§2), and at the pre-registered `nn@0.90` operating
point the intrinsic metrics land in the same neighbourhood with tight,
overlapping CIs (§3). One honest exception: MultiWOZ's induced IS_A graph is the
only one that is not acyclic — it carries a small cycle that neither ConEL-2 nor
ECB+ exhibits, plus more transitive shortcuts (§4) — a corpus-specific wrinkle in
the *hierarchy* family, not in the redundancy↔coherence tension that Track 2 set
out to test.

## 2. Tradeoff shape across the 4-point resolver axis

`python -m lattice.harness --sweep configs/m5-{corpus}-sweep.toml`. As the
resolver merges less aggressively (threshold 0.65 → 0.75 → 0.90), coherence rises
(fewer wrong merges) and redundancy rises (more true duplicates left unmerged) —
the designed tension. `exact-label` is shown for context but its
`coherence = 1.0` is **degenerate**: it merges only identical label strings, so no
concept has ≥2 surfaces and coherence is pinned at 1.0 by definition
(`multi-surface-concepts = 0`). The meaningful signal is the monotone rise across
the three `embedding-nn` thresholds.

**coherence (mean pairwise cosine over multi-surface concepts):**

| resolver | ConEL-2 | MultiWOZ | ECB+ |
|---|---|---|---|
| embedding-nn@0.65 | 0.7667 | 0.7607 | 0.7545 |
| embedding-nn@0.75 | 0.8338 | 0.8327 | 0.8196 |
| embedding-nn@0.90 | 0.9311 | 0.9355 | 0.9424 |
| exact-label | 1.0000† | 1.0000† | 1.0000† |

**redundancy (duplicate-rate) and concept-count:**

| resolver | ConEL-2 dup / concepts | MultiWOZ dup / concepts | ECB+ dup / concepts |
|---|---|---|---|
| embedding-nn@0.65 | 0.0000 / 347 | 0.0000 / 296 | 0.0000 / 543 |
| embedding-nn@0.75 | 0.0000 / 418 | 0.0000 / 436 | 0.0000 / 670 |
| embedding-nn@0.90 | 0.0153 / 523 | 0.0193 / 724 | 0.0116 / 864 |
| exact-label | 0.1173 / 554 | 0.3537 / 885 | 0.1490 / 933 |

† degenerate — see the paragraph above.

The coherence curve barely moves across corpora: at each `nn` threshold the three
values agree to within ~0.02. The tension direction (coherence↑ as merging
becomes more conservative; duplicate-rate↑ toward `exact-label`) is identical on
all three. This is the core evidence for the §1 verdict.

## 3. Operating-point holistic CIs (nn@0.90, B=50, percentile)

`python -m lattice.harness.stats configs/m5-{corpus}-nn090.toml <out> --holistic
--samples 50 --seed 0`. B=50 (a documented reduction from the plan's 150, uniform
across all three corpora for apples-to-apples: each resample re-runs the *entire*
pipeline, and MultiWOZ's 200 long dialogues cost ~84 s/resample — B=150 would be
~10 h for that corpus alone). Percentile intervals; holistic BCa would need a
pipeline jackknife, not computed (same as Track 1 §4).

| metric.key | ConEL-2 | MultiWOZ | ECB+ |
|---|---|---|---|
| coherence.coherence | 0.9311 [0.9267, 0.9401] | 0.9355 [0.9333, 0.9386] | 0.9424 [0.9376, 0.9487] |
| coherence.multi-surface-concepts | 27 [10, 21]‡ | 116 [68, 85]‡ | 58 [28, 44]‡ |
| coherence.singleton-fraction | 0.7954 [0.3082, 0.5479]‡ | 0.5663 [0.2809, 0.3947]‡ | 0.6991 [0.3628, 0.5117]‡ |
| redundancy.duplicate-rate | 0.0153 [0.0000, 0.0252] | 0.0193 [0.0153, 0.0281] | 0.0116 [0.0034, 0.0163] |
| redundancy.concept-count | 523 [285, 373]‡ | 724 [497, 562]‡ | 864 [580, 655]‡ |
| redundancy.near-duplicate-pairs | 4 [0, 4] | 7 [4, 7] | 5 [1, 5] |
| hierarchy-sanity.is-a-edges | 88 [41, 69]‡ | 163 [98, 126]‡ | 53 [28, 46]‡ |
| hierarchy-sanity.max-depth | 2 [2, 3] | 2 [2, 3] | 2 [1, 2] |
| hierarchy-sanity.cycle-components | 0 [0, 0] | 1 [0, 1] | 0 [0, 0] |
| hierarchy-sanity.cycle-nodes | 0 [0, 0] | 2 [0, 2] | 0 [0, 0] |
| hierarchy-sanity.self-loops | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| hierarchy-sanity.transitive-shortcuts | 0 [0, 0] | 5 [0, 4] | 1 [0, 1] |

‡ **One-sided count CIs (expected).** For count-like keys the point estimate sits
*above* its resample CI. This is the documented holistic-resampling effect (Track 1
§4): with-replacement resampling drops ~37 % of documents on average, so every
resample accretes a smaller graph than the full run — fewer concepts, edges, and
multi-surface concepts, and a lower singleton-fraction (duplicated documents make
their concepts non-singletons). Read these as *where the resamples concentrate*,
not as brackets around the estimate. The `coherence` and `duplicate-rate` CIs are
ordinary (they are ratios, not counts) and bracket their estimates.

## 4. Interpretation — method property, with an honest hierarchy caveat

**Redundancy↔coherence: a method property.** The coherence curve (§2) and the
operating-point `coherence`/`duplicate-rate` CIs (§3) are corpus-invariant to
within noise across three genres. The tension the M5 harness was designed to
expose — merge more → less redundancy but lower coherence; merge less → higher
coherence but more redundancy — reproduces out-of-sample without re-tuning
(`nn@0.90` was fixed on ConEL-2 and applied as-is). The single-corpus M5 result
was not a ConEL-2 artifact.

**Hierarchy-sanity: ConEL-2 is fully clean; the others less so.** ConEL-2 induces
a spotless IS_A forest — zero cycles, self-loops, and transitive shortcuts. ECB+
is close: acyclic and self-loop-free, with a single transitive shortcut
(`transitive-shortcuts = 1 [0,1]`). MultiWOZ is the real outlier and the only
*non-acyclic* graph: one cycle component (2 nodes; `cycle-components = 1 [0,1]`,
`cycle-nodes = 2 [0,2]`) and 5 transitive shortcuts (`[0,4]` — small and
resample-fragile, but the full-corpus graph carries them). The likely cause is
task-dialogue compound phrasing (e.g. "cheap hotel" IS_A "hotel" alongside
"hotel" IS_A "place to stay") producing short redundant chains and the occasional
2-cycle. This is a real corpus difference, confined to the *hierarchy* family; it
does not touch the redundancy↔coherence finding. Two distinctions matter:
**acyclicity** holds on ConEL-2 and ECB+ but fails only on MultiWOZ, while
**shortcut-freeness** holds only on ConEL-2 (ECB+ has one, MultiWOZ five). The
perfectly-clean forest is thus ConEL-2-specific, not universal.

**Bottom line.** Track 2's question — is the redundancy/coherence tradeoff a
method property or a ConEL-2 artifact? — answers **method property**, on two
independent out-of-sample corpora. The one thing that did *not* fully generalize
is hierarchy cleanliness, reported here rather than smoothed over.

## 5. Regeneration

```bash
# Fetch the MultiWOZ slice (200 dialogues, deterministic sorted-by-id). ECB+ and
# ConEL-2 are already on disk (scripts/fetch_ecbplus.py / fetch_conel2.py).
uv run --no-sync python scripts/fetch_multiwoz.py

# Tradeoff-shape sweeps (§2) — point estimates, 4-point resolver axis.
uv run --no-sync python -m lattice.harness --sweep configs/m5-multiwoz-sweep.toml reports/m5-multiwoz-sweep
uv run --no-sync python -m lattice.harness --sweep configs/m5-ecbplus-sweep.toml  reports/m5-ecbplus-sweep
uv run --no-sync python -m lattice.harness --sweep configs/m5-conel2-sweep.toml   reports/m5-conel2

# Operating-point holistic bootstrap CIs (§3), B=50, seed 0. Set HF_HUB_OFFLINE=1
# to avoid HF-Hub network flakiness across the many model reloads.
export HF_HUB_OFFLINE=1
uv run --no-sync python -m lattice.harness.stats configs/m5-conel2-nn090.toml   reports/intervals/m5-conel2-b50 --holistic --samples 50 --seed 0
uv run --no-sync python -m lattice.harness.stats configs/m5-multiwoz-nn090.toml reports/intervals/m5-multiwoz    --holistic --samples 50 --seed 0
uv run --no-sync python -m lattice.harness.stats configs/m5-ecbplus-nn090.toml  reports/intervals/m5-ecbplus     --holistic --samples 50 --seed 0
```

(macOS venv note: prefix each command with `chflags nohidden
.venv/lib/python*/site-packages/*.pth 2>/dev/null;` — the flag re-hides mid-batch
during model loads.)
