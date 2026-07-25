# M5 Cross-Corpus Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the M5 intrinsic harness on two additional corpora (MultiWOZ + ECB+) and produce a committed cross-corpus results doc that states, honestly either way, whether the redundancy/coherence tradeoff is a method property or a ConEL-2 artifact.

**Architecture:** M5 is purely intrinsic (its three metrics ignore `ground_truth`; the pipeline uses the `noun-chunk` extractor, not gold mentions), and the `mention-clusters` reader is corpus-agnostic. So this track touches **no existing source file**: it adds one stdlib MultiWOZ fetch/convert script + tests, four M5 configs (copies of the ConEL-2 configs with the dataset root swapped), and one committed results doc produced by the existing sweep and holistic-bootstrap CLIs. ECB+ is already on disk in the right JSONL shape.

**Tech Stack:** Python 3.13, stdlib only for the fetch script (`argparse`, `hashlib`, `json`, `urllib.request`, `pathlib`); pytest; existing `lattice.harness` sweep + `lattice.harness.stats` holistic bootstrap; ML models (spaCy `en_core_web_sm`, sentence-transformers `all-MiniLM-L6-v2`) for the manual run task only.

## Global Constraints

- **No new dependencies** — the fetch script is stdlib-only; pyproject is frozen.
- **`data/**` is gitignored** — MultiWOZ data is never committed; `docs/results/*.md` **is** committed.
- **No model downloads or network access in tests** — `convert_dialogue`/`select_dialogues` are pure and tested on inline fixtures; ML/network happen only in the manual Task 3.
- **Deterministic, seed-stable slice** — no `random`; the slice is a sorted first-N.
- **Seed 0** for every run; holistic **B=150** (percentile intervals).
- **MultiWOZ 2.2** test split, **first 200 dialogues by id**, 4-point resolver axis (exact-label, nn@0.90/0.75/0.65).
- Work on `main`; **never merge/tag/push** — the user decides.
- macOS venv quirk: prefix every `uv run` in Task 3 with `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null;` (the flag re-hides mid-batch during ML loads).

---

### Task 1: MultiWOZ fetch/convert script + unit tests

**Files:**
- Create: `scripts/fetch_multiwoz.py`
- Test: `tests/scripts/test_fetch_multiwoz.py`

**Interfaces:**
- Produces: `convert_dialogue(dialogue: dict) -> dict` returning `{"id": str, "kind": "transcript", "text": str, "mentions": []}`; `select_dialogues(dialogues: list[dict], limit: int = 200) -> list[dict]` returning a deterministic sorted-by-`dialogue_id` first-`limit` slice; `main()` (network I/O, not unit-tested).
- Consumes: nothing from other tasks. The emitted `data/multiwoz/test.jsonl` is consumed by the `mention-clusters` reader (unchanged) via the Task 2 configs.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_fetch_multiwoz.py`:

```python
from scripts.fetch_multiwoz import convert_dialogue, select_dialogues

SAMPLE = {
    "dialogue_id": "MUL0484.json",
    "services": ["train"],
    "turns": [
        {"speaker": "USER", "turn_id": "0",
         "utterance": "I need a train from norwich to cambridge"},
        {"speaker": "SYSTEM", "turn_id": "1", "utterance": "What time? "},
        {"speaker": "USER", "turn_id": "2", "utterance": "   "},
        {"speaker": "USER", "turn_id": "3", "utterance": "After 10:00  "},
    ],
}


def test_convert_dialogue_joins_turns_and_strips_id_suffix():
    row = convert_dialogue(SAMPLE)
    assert row["id"] == "multiwoz-MUL0484"
    assert row["kind"] == "transcript"
    assert row["text"] == (
        "I need a train from norwich to cambridge\nWhat time?\nAfter 10:00"
    )
    assert row["mentions"] == []


def test_convert_dialogue_text_survives_block_segmenter():
    # BlockSegmenter strips and splits on blank lines; a trailing space or a
    # whitespace-only turn would break the emitted text. rstrip + drop-blank.
    text = convert_dialogue(SAMPLE)["text"]
    assert text == text.strip()
    assert "\n\n" not in text


def test_convert_dialogue_single_turn():
    row = convert_dialogue(
        {"dialogue_id": "SNG01.json",
         "turns": [{"speaker": "USER", "turn_id": "0", "utterance": "hello"}]}
    )
    assert row["text"] == "hello"
    assert row["id"] == "multiwoz-SNG01"


def test_select_dialogues_is_deterministic_and_sorted():
    # 15 dialogues across three id prefixes; sorted-first-4 must be reproducible
    # and independent of input order. 'M' < 'P' < 'S', so MUL* come first.
    dialogues = [
        {"dialogue_id": f"{p}{i}.json", "turns": []}
        for p in ("SNG", "MUL", "PMUL")
        for i in range(5)
    ]
    selected = select_dialogues(dialogues, limit=4)
    ids = [d["dialogue_id"] for d in selected]
    assert ids == sorted(ids)
    assert len(selected) == 4
    assert all(i.startswith("MUL") for i in ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_multiwoz.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_multiwoz'`.

- [ ] **Step 3: Write the script**

Create `scripts/fetch_multiwoz.py`:

```python
"""Fetch MultiWOZ 2.2 (Zang et al. 2020) test split and convert to lattice's
unified mention-cluster JSONL for M5 cross-corpus validation (credibility
Track 2). Stdlib only:
    uv run --no-sync python scripts/fetch_multiwoz.py

M5 is intrinsic — the three metrics judge the accreted graph and ignore
ground_truth, and the pipeline uses the noun-chunk extractor, not gold
mentions — so only the transcript text is needed; `mentions` is emitted empty.
A deterministic slice of the first 200 dialogues (sorted by id) keeps the
holistic bootstrap tractable and comparable to ECB+ (~200 docs). CHECKSUMS
pins the exact bytes so source drift on the (unpinned) master branch is
detectable."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/budzianowski/multiwoz/master/"
    "data/MultiWOZ_2.2/test"
)
SHARDS = ("dialogues_001.json", "dialogues_002.json")
LIMIT = 200


def convert_dialogue(dialogue: dict) -> dict:
    """One MultiWOZ dialogue -> one transcript document. Utterances are
    rstripped and blank turns dropped so the text survives BlockSegmenter's
    strip() (no trailing whitespace, no blank lines). No gold mentions."""
    texts: list[str] = []
    for turn in dialogue["turns"]:
        utterance = turn["utterance"].rstrip()
        if utterance:
            texts.append(utterance)
    did = dialogue["dialogue_id"].removesuffix(".json")
    return {
        "id": f"multiwoz-{did}",
        "kind": "transcript",
        "text": "\n".join(texts),
        "mentions": [],
    }


def select_dialogues(dialogues: list[dict], limit: int = LIMIT) -> list[dict]:
    """Deterministic slice: sort by dialogue_id, take the first `limit`.
    Independent of shard order, so the emitted corpus is reproducible."""
    ordered = sorted(dialogues, key=lambda d: d["dialogue_id"])
    return ordered[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()
    out_dir = Path(args.root) / "multiwoz"
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues: list[dict] = []
    for shard in SHARDS:
        with urllib.request.urlopen(f"{BASE}/{shard}") as response:
            dialogues.extend(json.load(response))
    selected = select_dialogues(dialogues, args.limit)
    out_path = out_dir / "test.jsonl"
    with out_path.open("w") as f:
        for dialogue in selected:
            f.write(json.dumps(convert_dialogue(dialogue), sort_keys=True) + "\n")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (out_dir / "CHECKSUMS").write_text(f"{digest}  {out_path.name}\n")
    print(f"wrote {out_path} ({len(selected)} dialogues, {digest[:12]}…)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/scripts/test_fetch_multiwoz.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `uv run --no-sync ruff check scripts/fetch_multiwoz.py tests/scripts/test_fetch_multiwoz.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_multiwoz.py tests/scripts/test_fetch_multiwoz.py
git commit -m "feat: stdlib MultiWOZ 2.2 fetch/convert for M5 cross-corpus (track 2)"
```

---

### Task 2: Four M5 configs + validation test

**Files:**
- Create: `configs/m5-multiwoz-sweep.toml`, `configs/m5-multiwoz-nn090.toml`, `configs/m5-ecbplus-sweep.toml`, `configs/m5-ecbplus-nn090.toml`
- Test: `tests/harness/test_m5_cross_corpus_configs.py`

**Interfaces:**
- Consumes: the `mention-clusters` dataset reader (unchanged); `data/multiwoz/test.jsonl` (Task 1 output, produced in Task 3) and `data/ecbplus/test.jsonl` (already on disk). Config validation does **not** require the data files to exist (validation is schema-only).
- Produces: four config paths the Task 3 run commands invoke.

The four configs are byte-for-byte copies of `configs/m5-conel2-sweep.toml` and `configs/m5-conel2-nn090.toml` with only `dataset.params.root` changed. The sweep configs keep the same 4-point `[axes].resolver` list; the nn090 configs keep `resolver = embedding-nn, threshold = 0.90`.

- [ ] **Step 1: Write the failing test**

Create `tests/harness/test_m5_cross_corpus_configs.py`:

```python
from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig, expand


def test_multiwoz_sweep_config_expands_to_the_four_point_axis():
    sweep = load_config("configs/m5-multiwoz-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    assert [c.resolver.name for c in configs] == [
        "exact-label", "embedding-nn", "embedding-nn", "embedding-nn",
    ]
    assert [c.resolver.params.get("threshold") for c in configs] == [
        None, 0.90, 0.75, 0.65,
    ]
    for config in configs:
        assert config.dataset.params["root"] == "data/multiwoz"


def test_ecbplus_sweep_config_expands_to_the_four_point_axis():
    sweep = load_config("configs/m5-ecbplus-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    for config in configs:
        assert config.dataset.params["root"] == "data/ecbplus"


def test_nn090_configs_load_at_the_operating_point():
    for root in ("multiwoz", "ecbplus"):
        config = load_config(
            f"configs/m5-{root}-nn090.toml", model=ExperimentConfig
        )
        assert config.resolver.name == "embedding-nn"
        assert config.resolver.params["threshold"] == 0.90
        assert config.dataset.params["root"] == f"data/{root}"
        metric_names = {m.name for m in config.metrics}
        assert {"redundancy", "hierarchy-sanity"} <= metric_names
        assert any(dm.name == "coherence" for dm in config.document_metrics)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m5_cross_corpus_configs.py -q`
Expected: FAIL — `FileNotFoundError` / config-not-found for `configs/m5-multiwoz-sweep.toml`.

- [ ] **Step 3: Create the four configs**

Create `configs/m5-multiwoz-sweep.toml` — copy `configs/m5-conel2-sweep.toml` verbatim, then change only the dataset block to:

```toml
[base.dataset]
name = "mention-clusters"
[base.dataset.params]
root = "data/multiwoz"
split = "test"
```

Create `configs/m5-ecbplus-sweep.toml` — same, with `root = "data/ecbplus"`.

Create `configs/m5-multiwoz-nn090.toml` — copy `configs/m5-conel2-nn090.toml` verbatim, then change only:

```toml
[dataset]
name = "mention-clusters"
[dataset.params]
root = "data/multiwoz"
split = "test"
```

Create `configs/m5-ecbplus-nn090.toml` — same, with `root = "data/ecbplus"`.

Leave every other section (segmenter, extractor `noun-chunk`, scorer, resolver/axes, relation_inducer `union{hearst,compound}`, embedder `sentence-transformer`, metrics `redundancy`/`hierarchy-sanity`, document_metrics `coherence`, `[run] seed = 0`) exactly as in the ConEL-2 originals.

- [ ] **Step 4: Run the test to verify it passes**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m5_cross_corpus_configs.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Full suite + lint (no regressions)**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q && uv run --no-sync ruff check src/ tests/`
Expected: all pass, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add configs/m5-multiwoz-sweep.toml configs/m5-multiwoz-nn090.toml \
        configs/m5-ecbplus-sweep.toml configs/m5-ecbplus-nn090.toml \
        tests/harness/test_m5_cross_corpus_configs.py
git commit -m "feat: M5 cross-corpus configs (multiwoz + ecbplus, sweep + nn090)"
```

---

### Task 3: Fetch data, run sweeps + diagnostic gate + holistic bootstrap, write results doc

This is the manual, ML-heavy, non-committed-data task (mirrors Track 1's Task 11). It requires the ml extras + cached models. It produces gitignored reports under `reports/intervals/` and the **committed** results doc. No fabricated numbers: every table cell is populated from an on-disk JSON produced by the recorded command.

**Files:**
- Create: `docs/results/2026-07-25-m5-cross-corpus.md` (committed)
- Produces (gitignored): `reports/m5-multiwoz-sweep/`, `reports/m5-ecbplus-sweep/`, `reports/intervals/m5-multiwoz/`, `reports/intervals/m5-ecbplus/`, `data/multiwoz/test.jsonl`

- [ ] **Step 1: Ensure env (models + venv quirks)**

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())")
uv run --group ml python scripts/fetch_models.py   # no-op if already cached
```

- [ ] **Step 2: Fetch the MultiWOZ slice**

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python scripts/fetch_multiwoz.py
```
Expected: `wrote data/multiwoz/test.jsonl (200 dialogues, <digest>…)`. Confirm 200 lines: `wc -l data/multiwoz/test.jsonl`.

- [ ] **Step 3: DIAGNOSTIC GATE — confirm the corpus has merge signal before spending compute**

Run the single-config point once and inspect concept structure:

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python -m lattice.harness configs/m5-multiwoz-nn090.toml \
  | python3 -c "import sys,json; d=json.load(sys.stdin); \
      print('concept-count:', d['metrics']['redundancy']['concept-count']); \
      print('multi-surface-concepts:', d['metrics']['coherence']['multi-surface-concepts']); \
      print('coherence:', d['metrics']['coherence']['coherence'])"
```

Decision rule: if `multi-surface-concepts` is `0` (coherence definitionally pinned at 1.0 — no concept has ≥2 surfaces to score), the tradeoff is **unobservable on this slice**. STOP and report to the user with the numbers; options are a larger `--limit` or MultiWOZ 2.1. Do **not** proceed to the ~1 hr bootstrap on a degenerate corpus. If `multi-surface-concepts` > 0, continue.

- [ ] **Step 4: Run the tradeoff sweeps (both corpora)**

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python -m lattice.harness --sweep configs/m5-multiwoz-sweep.toml reports/m5-multiwoz-sweep
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python -m lattice.harness --sweep configs/m5-ecbplus-sweep.toml reports/m5-ecbplus-sweep
```
Each writes a JSON + markdown report with per-resolver-point `redundancy`, `hierarchy-sanity`, `coherence` values. These are the tradeoff-shape point estimates.

- [ ] **Step 5: Run the holistic bootstrap at the operating point (both corpora)**

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python -m lattice.harness.stats configs/m5-multiwoz-nn090.toml reports/intervals/m5-multiwoz --holistic --samples 150 --seed 0
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
uv run --no-sync python -m lattice.harness.stats configs/m5-ecbplus-nn090.toml reports/intervals/m5-ecbplus --holistic --samples 150 --seed 0
```
Each writes `reports/intervals/m5-{corpus}/interval-report.json` with percentile CIs per metric key. (~50+ min each; ECB+ has 206 docs, MultiWOZ 200.)

- [ ] **Step 6: Write the results doc**

Create `docs/results/2026-07-25-m5-cross-corpus.md`. Reuse ConEL-2's numbers from Track 1's `reports/intervals/m5-conel2/interval-report.json` and `docs/results/2026-07-14-interval-analysis.md` §4 for the third column. Structure:

```markdown
# M5 Cross-Corpus Validation — Is the redundancy/coherence tradeoff a method property?

Track 2 deliverable. Runs the M5 intrinsic harness on MultiWOZ (new
transcript corpus, 200 dialogues) and ECB+ (cross-genre, 206 news docs) and
compares against ConEL-2 (58 conversations). No number is estimated — each is
produced by the recorded command against checked-in code and fetched data.

## 1. Headline verdict
[One paragraph: does the monotone redundancy↑/coherence↑ tension replicate on
both corpora? State it plainly, either way. Cite the §2 table.]

## 2. Tradeoff shape across the 4-point resolver axis
[For each corpus, a table with rows exact-label / nn@0.90 / nn@0.75 / nn@0.65
and columns redundancy(duplicate-rate), coherence, hierarchy-sanity(is-a-edges).
Populate every cell from reports/m5-{corpus}-sweep JSON. ConEL-2 column from
Track 1. Note whether redundancy and coherence both move monotonically with
threshold on each corpus.]

## 3. Operating-point holistic CIs (nn@0.90, B=150, percentile)
[Table: metric key | ConEL-2 est [CI] | MultiWOZ est [CI] | ECB+ est [CI].
Populate MultiWOZ/ECB+ from reports/intervals/m5-{corpus}/interval-report.json;
ConEL-2 from reports/intervals/m5-conel2. Note the count-like keys' one-sided
CIs (holistic resampling drops documents — same effect documented in Track 1 §4).]

## 4. Interpretation — method property or artifact?
[Honest reading. If the tension holds on all three: evidence it is a method
property, not a ConEL-2 artifact. If it differs: say how (e.g. MultiWOZ's slot
entities behave differently) and what that implies for the nn@0.90 operating
point applied out-of-sample. Do not overclaim.]

## 5. Regeneration
[The exact commands from Steps 2, 4, 5 — fetch, both sweeps, both holistic runs,
with seeds and B.]
```

Sanity checks before finalizing: (a) each holistic CI's `method` field is `percentile`; (b) every cell traces to a real JSON path; (c) if the diagnostic gate (Step 3) forced any change (larger slice), record the actual `--limit` used.

- [ ] **Step 7: Commit the results doc**

```bash
git add docs/results/2026-07-25-m5-cross-corpus.md
git commit -m "docs: M5 cross-corpus validation results (multiwoz + ecbplus vs conel-2)"
```

---

## Self-Review

**1. Spec coverage.** Spec §3 (no new adapter, intrinsic) → Task 1/2 architecture notes. §4 (fetch script, tests, 4 configs, results doc) → Tasks 1–3. §5 (MultiWOZ 2.2, ~200 sorted slice, convert) → Task 1. §6 (sweep, diagnostic gate, holistic B=150) → Task 3 Steps 3–5. §7 (deliverable doc structure, either-way verdict) → Task 3 Step 6. §8 (constraints) → Global Constraints. §9 (unit + config tests) → Task 1/2 tests. §10 (risks: diagnostic gate) → Task 3 Step 3. §11 (YAGNI: no source changes) → confirmed, no `src/` edits in any task. Full coverage.

**2. Placeholder scan.** No "TBD"/"handle errors" in code steps. The results-doc skeleton's bracketed notes are populate-from-JSON instructions for a data-generation task (numbers cannot exist before the run) — the same pattern as Track 1's Task 11, not code placeholders. All code steps carry complete code.

**3. Type consistency.** `convert_dialogue`/`select_dialogues` signatures and the `{id,kind,text,mentions}` dict shape are identical across Task 1's script, its tests, and the Task 2 reader contract. Config keys (`resolver.name`, `resolver.params.threshold`, `dataset.params.root`, `metrics`, `document_metrics`) match the ConEL-2 configs and the `expand()`/`ExperimentConfig` API verified in `tests/harness/test_m5_e2e.py`. CLI flags (`--sweep`, `--holistic --samples --seed`) match `src/lattice/harness/__main__.py` and `src/lattice/harness/stats/__main__.py`.
