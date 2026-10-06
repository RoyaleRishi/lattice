# M5 Cross-Corpus Validation — Design Spec (Credibility Track 2)

**Goal:** Determine whether the M5 redundancy/coherence tradeoff — found on a
single 58-conversation ConEL-2 slice — is a property of the *method* or an
artifact of that *one corpus*, by running the intrinsic harness on two
additional corpora and comparing the tradeoff shape and operating-point numbers.

**Status:** Design approved 2026-07-25. Second of three credibility
sub-projects (Track 1 = statistical intervals, shipped; Track 3 =
excluded-method baseline, pending).

---

## 1. Motivation

M5's intrinsic story rests entirely on ConEL-2 test (58 conversations). The
three metrics form a designed tension pair: **redundancy** measures what the
resolver failed to merge; **coherence** measures what it wrongly merged. As the
resolver threshold rises, both are expected to rise (fewer merges → more
redundancy, but also fewer *wrong* merges → higher coherence). `nn@0.90` was
chosen on ConEL-2 as the balance point. Nothing tells us whether that tension —
or that operating point — generalizes beyond the one corpus it was tuned on.
Track 2 answers that with out-of-sample evidence.

## 2. Decisions (locked with the user, 2026-07-25)

| Decision | Choice | Rationale |
|---|---|---|
| Corpus source | **Both** a new transcript corpus **and** ECB+ | New corpus = faithful same-genre replication; ECB+ = free cross-genre bonus (already on disk). |
| New transcript corpus | **MultiWOZ** (task-oriented dialogue) | Reliable stdlib-fetchable GitHub source; structured entity density (place names, times) gives the metrics real merge decisions; transcript-style. |
| Run scope | **Sweep + holistic CI** | Sweep traces the tradeoff *shape* (the actual thing being tested); holistic bootstrap gives the credibility-consistent CI at the operating point. |
| MultiWOZ slice | **~200 dialogues** (deterministic first-N of the test split) | Comparable to ECB+ (206 docs); enough entity recurrence for real signal. |
| Sweep axis | **4-point** resolver axis (exact-label, nn@0.90/0.75/0.65) | Exact match to the original ConEL-2 sweep → clean apples-to-apples. |
| Holistic B | **150** (percentile intervals) | Matches ConEL-2's documented reduction (each resample re-runs the whole pipeline). |
| Seed | **0** throughout | Consistency with all prior milestones. |

## 3. Architecture

The load-bearing discovery: **M5 is purely intrinsic**, and the existing
`mention-clusters` dataset reader is corpus-agnostic. The three M5 metrics
(`redundancy`, `hierarchy-sanity`, `coherence`) take a `ground_truth` parameter
but reference it nowhere in their bodies — verified during design. The M5
pipeline uses the `noun-chunk` extractor (not gold mentions). Therefore a second
corpus needs **only the document `text`** — no annotations, and **no new dataset
adapter**.

Consequently Track 2 touches **no existing source file**. It is data
acquisition + configs + a run + a committed results doc. The only new code is one
stdlib fetch/convert script and its unit test.

**Reused unchanged:** `mention-clusters` reader, `harness/sweep.py`, the three
M5 metrics, the holistic-bootstrap CLI (`python -m lattice.harness.stats
--holistic`), and Track 1's entire `lattice.harness.stats` layer.

## 4. Components & files

**Create:**
- `scripts/fetch_multiwoz.py` — stdlib-only (`argparse`, `hashlib`, `json`,
  `urllib.request`, `pathlib`), mirroring `scripts/fetch_conel2.py`. Downloads
  MultiWOZ, slices a deterministic subset, converts each dialogue to a transcript
  document, writes `data/multiwoz/test.jsonl` in the `{id, kind, text, mentions}`
  shape with `mentions: []`. Emits a `CHECKSUMS` file like the other fetchers.
- `tests/scripts/test_fetch_multiwoz.py` — drives the pure `convert_dialogue`
  function on an inline MultiWOZ-shaped fixture; no network, no models.
- `configs/m5-multiwoz-sweep.toml`, `configs/m5-multiwoz-nn090.toml`,
  `configs/m5-ecbplus-sweep.toml`, `configs/m5-ecbplus-nn090.toml` — copies of
  the ConEL-2 M5 configs with `dataset.params.root` swapped
  (`data/multiwoz`, `data/ecbplus`).
- `docs/results/2026-07-25-m5-cross-corpus.md` — the committed results doc
  (see §7).

**Not committed:** `data/multiwoz/**` (gitignored, like all `data/**`).

**Unchanged:** every file under `src/`.

## 5. MultiWOZ acquisition & conversion

- **Source/version:** MultiWOZ **2.2** from `github.com/budzianowski/multiwoz`
  (JSON dialogues under the `test` split), fetched via `urllib.request` from raw
  GitHub URLs. The exact raw URL(s) and file layout (2.2 shards the test split
  across `dialogues_*.json` files) are **verified as the first implementation
  step** before the spec's checksum is fixed; fall back to MultiWOZ 2.1's single
  `data.json` if the 2.2 layout proves impractical for a small slice.
- **Conversion (`convert_dialogue`, pure function):** each dialogue → one
  document. Concatenate its turn utterances (in order, both speakers) joined by
  `"\n"`, matching ConEL-2's newline-joined transcript text. Fields:
  `id = f"multiwoz-{dialogue_id}"`, `kind = "transcript"`, `text = <joined
  utterances>`, `mentions = []`.
- **Slice:** deterministic **first ~200 dialogues** of the test split (stable
  ordering by dialogue id). Recorded in the fetch script and the results doc, not
  silently defaulted.

## 6. Run protocol

All runs seed 0.

1. **Tradeoff shape (cheap):** run `harness/sweep.py` on
   `m5-multiwoz-sweep.toml` and `m5-ecbplus-sweep.toml` (4-point resolver axis)
   → point estimates of `redundancy`, `hierarchy-sanity`, `coherence` per
   threshold on each corpus.
2. **Diagnostic gate:** after the first sweep point on MultiWOZ, inspect
   `concept-count` and `multi-surface-concepts`. If concepts are almost all
   singletons (coherence definitionally pinned at 1.0 with no multi-surface
   concepts), the tradeoff is unobservable — **surface this and decide** (larger
   slice / different version) *before* spending ~50 min on the bootstrap.
3. **Headline CIs:** holistic bootstrap at the `nn@0.90` operating point,
   **B=150**, percentile intervals, on MultiWOZ + ECB+ (via
   `python -m lattice.harness.stats <config> <outdir> --holistic --seed 0`).

## 7. Deliverable

One committed results doc, `docs/results/2026-07-25-m5-cross-corpus.md`, with:

- **Tradeoff table:** redundancy and coherence (and hierarchy-sanity) across the
  4 resolver points, side by side for ConEL-2 / MultiWOZ / ECB+ — does the
  monotone redundancy↑/coherence↑ tension appear on all three?
- **Operating-point CI table:** the `nn@0.90` intrinsic metrics with holistic
  percentile CIs, all three corpora.
- **Honest verdict, either way:** "the tradeoff replicates → method property"
  *or* "it is corpus-dependent → here is how it differs." Both are valid,
  reportable outcomes; this is a test, not a confirmation exercise.
- **Regen commands** for every number, matching Track 1's doc style.

## 8. Constraints (inherited, binding)

- Stdlib-only fetch script; **no new dependencies** (pyproject frozen).
- `data/**` gitignored — MultiWOZ data never committed; `docs/results/*.md` **is**
  committed.
- Deterministic, seed-stable slice (no `random` without a fixed seed).
- **No model downloads or network in tests** — fetch/convert tested on inline
  fixtures; ML runs happen only in the manual regeneration step.
- Work on `main`; never merge/tag/push — the user decides.

## 9. Testing

- **Unit (`convert_dialogue`):** inline MultiWOZ-shaped dialogue → assert
  newline-joined utterance order, `id`/`kind`, empty `mentions`. Include a
  multi-turn case and a single-turn edge case.
- **Config validation:** the four new TOMLs load through the existing
  `ExperimentConfig` validation path (same as every other committed config).
- **No new metric logic** → no new metric tests; the M5 metric suite already
  covers `redundancy`/`coherence`/`hierarchy-sanity`.

## 10. Risks

- **Entity-density degeneration (primary):** MultiWOZ's slot entities may recur
  less than ConEL-2's open-web entities; a too-sparse slice yields mostly
  singletons and degenerate metrics. Mitigated by the §6 diagnostic gate.
- **Source/version drift:** 2.2's sharded layout or raw-URL paths may differ from
  expectation; mitigated by verify-first and the 2.1 fallback.
- **Compute:** holistic bootstrap re-runs the whole pipeline per resample; a
  ~200-doc slice is slower per resample than ConEL-2's 58. B=150 keeps each
  corpus near ~1 hr; recorded, not silently defaulted.

## 11. Out of scope (YAGNI)

No new dataset adapter; no changes to any existing source file; no redundancy
auto-repair; no finer threshold grid; no MultiWOZ gold/annotations; no
re-tuning of the operating point (nn@0.90 is applied out-of-sample, by design).

## 12. Success criteria

Track 2 succeeds when: the fetch/convert script is shipped and unit-tested; the
four configs run cleanly; the results doc reports the tradeoff shape and
operating-point CIs across all three corpora with regen commands; and it states
a clear, honest verdict on method-vs-artifact — whichever way the evidence falls.
