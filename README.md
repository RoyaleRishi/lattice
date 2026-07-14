# lattice

**A concept-memory engine.** lattice ingests a stream of documents — LLM
session transcripts, notes, any unit of text — and maintains an accreting,
normalized concept graph that preserves concept identity **across
documents** over time. Its value is not per-document keyphrase extraction;
it is cross-document identity: recognizing that "vector store" in session 5
and "vector database" in session 200 are the same concept, and letting a
graph of concepts and their relations grow as documents arrive.

Every algorithmic stage is a swappable adapter behind a port, and every
default was chosen by benchmark (see the results table below).

## Install

```bash
uv add lattice            # core: dependency-light "lite" profile
uv add "lattice[ml]"      # + spaCy & sentence-transformers for "standard"
uv run python scripts/fetch_models.py   # one-time model download (standard)
```

## Quickstart

```python
from lattice import Engine

engine = Engine()  # "lite" profile — dependency-free; see the toggle below

engine.ingest("Olive oil is a fat prized in Mediterranean cooking.")
engine.ingest("Mediterranean groves grow olive trees for oil.")

view = engine.view()
olive = view.find_concept("olive")

engine.save("memory.json")            # versioned JSON, survives restarts
restored = Engine.load("memory.json")
restored.ingest("Olive presses yield fresh oil each autumn.")
```

`Engine()` defaults to the **lite** profile: the same pipeline topology as
the real thing, with a toy tokenizer and hashing embedder — instant,
dependency-free, right for smoke tests and CI. Note its limits: it only
sees single words of ≥ 4 letters ("olive", not "olive oil").

**For real use, flip one switch:**

```
engine = Engine(profile="standard")
```

## Profiles

| stage | lite | standard | why (evidence) |
|---|---|---|---|
| extractor | token (words ≥ 4 chars) | spaCy noun chunks | real noun phrases (M2 spec) |
| embedder | hashing trigrams | all-MiniLM-L6-v2 | semantic identity (M2/M3 sweeps) |
| scorer | embedding-cosine | embedding-cosine | best salience F1 on Inspec (M2) |
| resolver | embedding-NN @ 0.90 | embedding-NN @ 0.90 | M5's recorded operating point |
| relations | hearst + compound | hearst + compound | above published TExEval-2 band on 2/6 golds (M4) |

Both profiles share one topology — switching changes quality, never
behavior shape. Full control: `Engine.from_config(path_or_dict)` with the
same TOML schema the experiment harness uses.

## Benchmark evidence

| track | benchmark | headline |
|---|---|---|
| salience | Inspec | embedding-cosine F1@10 0.355 vs frequency 0.240 |
| identity | ECB+ / ConEL-2 | embedding-NN beats exact-label B³ F1 on both (0.643 vs 0.608; 0.962 vs 0.939) |
| hierarchy | TExEval-2 (6 English golds) | hearst+compound union ≥ members on 6/6; above the published band on food & food-wordnet |
| integration | ConEL-2 intrinsic | nn@0.90: duplicate-rate 0.117 → 0.015 at coherence 0.931 |

Specs and sweeps: `docs/` (start at
`docs/2026-07-05-lattice-architecture-design.md`).

## Persistence

`engine.save(path)` writes versioned JSON (`format_version: 1`) holding the
fully resolved config, the graph, and the document counter.
`Engine.load(path)` rebuilds the engine and **resumes exactly**: processing
A, B, save, load, C equals processing A, B, C in one run (test-enforced).

## Stability

Pre-1.0: the public contract is `lattice.__all__` — `Engine`, `GraphView`,
`Document`, `Concept`, `Relation`, `GraphDelta`, `GraphSnapshot`,
`__version__`. Minor versions may break it with a changelog note.
Everything below the top level is internal. Save files carry
`format_version` and are readable by any lattice that understands it.
