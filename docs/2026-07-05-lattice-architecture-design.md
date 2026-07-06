# lattice — Architecture Design Spec

**Date:** 2026-07-05
**Status:** Draft for review
**Repo:** `~/Desktop/Projects/lattice` (independent; NeuroNote is a future downstream consumer)

---

## 1. What lattice is

lattice is a **concept-memory engine**. It ingests a stream of *documents* — an LLM session
transcript, a note, any unit of text — and maintains an **accreting, normalized concept graph**
that preserves concept identity **across documents** over time.

The engine's value is not per-document keyphrase extraction (that is a solved, commodity step);
it is **cross-document identity** — recognizing that "vector store" in session 5 and "vector
database" in session 200 are the same concept, and letting a structured graph of concepts and
their relations grow as more documents arrive. NeuroNote's notes are simply one instance of
"document"; lattice is designed to serve any such consumer.

lattice is simultaneously:
- a **research instrument** — every algorithmic step is swappable and benchmarkable, so competing
  approaches can be compared objectively; and
- a **reusable engine** — a stable `document → concept-graph` API a downstream app depends on.

It is built API-first (because it is meant to be the shared engine), but the **immediate
deliverable is the experiment harness** that proves the algorithm choices.

## 2. Framing constraints

- **C1 — Total independence.** Standalone repo. lattice never imports or depends on NeuroNote;
  NeuroNote will depend on lattice. No naming, vocabulary, or design inherited from NeuroNote.
- **C2 — No anchoring.** Designed fresh from the problem, not as a reproduction of any existing
  pipeline. Convergent design is acceptable; inherited design is not.
- **C3 — SOLID** is the guiding design discipline.

## 3. Research grounding

The design targets the *deterministic / unsupervised* frontier of concept extraction (no
generative LLM on the critical path), because that is the intended operating constraint.

- **Candidate generation** at frontier quality is PoS-bounded noun-phrase extraction — the basis
  for the `Extractor` port's default adapter.
- **Salience** is where the frontier has moved and where experiments will focus. Reference
  approaches to implement as `Scorer` adapters: embedding-cosine (KeyBERT/SIFRank baseline),
  **masked-document ranking (MDERank, 2022)** — fixes the length-bias and context-free-embedding
  flaws of phrase-document cosine — and **hierarchical significance (HCUKE, 2024)** — document →
  segment → candidate significance, which exploits document structure the baselines ignore.
- **Cross-document identity** is evaluated with cross-document coreference metrics; **ECB+**
  (clustering: B³/CEAF/MUC) and **ConEL-2** (conversational entity+concept linking — closest to
  the LLM-session framing) back the normalization benchmarks.
- **Hierarchy (`IS_A`)** is evaluated against **SemEval-2016 Task 13 (TExEval-2)** and
  hypernym-discovery sets (edge P/R/F).

Key references: MDERank (arXiv 2110.06651), PromptRank (ACL 2023), HCUKE (Knowledge-Based Systems
2024), LLM-empowered KG construction survey (arXiv 2510.20345), SemEval-2016 Task 13, ConEL-2
(informagi/conversational-entity-linking-2022), ECB+ cross-document coreference.

## 4. Architecture: ports & adapters (hexagonal) with a thin orchestrator

A stable **core** — the domain model, typed contracts, and orchestration logic — depends only on
**ports** (abstract interfaces). Every swappable thing is an **adapter** behind a port: each
algorithm stage, the embedding model, the vector index, the graph store, *and* the benchmark
datasets and metrics. This is the only structure that satisfies all three constraints at once:
swappability is the literal definition of adapters behind ports; streaming-native memory lives
behind stateful ports the core treats identically; and SOLID falls out (each port SRP + ISP,
new algorithms open/closed, core depends on abstractions = DIP, adapters honor contracts = LSP).

Rejected alternatives: a **linear pipeline** (leaks stateful memory through parameters and couples
datasets to algorithms) and a **DAG framework** (over-engineered — per-document flow is linear;
YAGNI).

```
                      ┌──────────────────────────────────────────┐
   Document  ───────▶ │  Orchestrator:  process(document) → delta │ ──────▶ GraphDelta
                      └──────────────────────────────────────────┘
                          │ depends only on ports ▼
   Segmenter → Extractor → Scorer → Resolver → RelationInducer → GraphIntegrator
                                       │                              │
                                (ConceptStore)                  (graph state)
                             shared: Embedder
   Harness ports:  Dataset ─▶ [fold process] ─▶ snapshot ─▶ Metric ─▶ report
```

### 4.1 Runtime model (streaming-paramount)

The orchestrator's fundamental unit of work is `process(document, memory_state) → GraphDelta`.
**Batch is a fold over the stream** — "replay the whole stream at once." Batch has **no privileged
code path**; this invariant is what keeps the streaming case first-class rather than bolted on.
Only the first *implementations* of the stateful stores run batch-style; their ports are designed
streaming-native from day one.

### 4.2 Memory representation

The stateful ports (`ConceptStore`, `GraphIntegrator`) **hold** the accreting state. `process()`
mutates them through their interfaces and returns only the `GraphDelta`. Memory is **not** threaded
as an immutable value (copying an ever-growing graph per document is unrealistic at streaming
volume). Reproducibility is preserved by an explicit `snapshot()` contract on the stateful ports,
plus `reset()` between experiment runs.

## 5. Domain contracts (`core/`)

Pure data types, zero external dependencies:

| Type | Carries | Produced by |
|---|---|---|
| `Document` | `id`, `kind`, payload (text or turns), `timestamp` (stream ordering), metadata | input |
| `Unit` | `id`, `document_id`, `text`, `order`, `kind` (turn/block/sentence), optional `speaker` | `Segmenter` |
| `Mention` | `surface`, `unit_id`, char `span`, `context`, `head`/`lemma` | `Extractor` |
| `ScoredMention` | `Mention` + `salience` + `selected` | `Scorer` |
| `Concept` | canonical node: stable `id`, `label`, `embedding`, `first_seen`, `updated_at` | `Resolver` |
| `Relation` | typed edge: `type`, `source_id`, `target_id`, `confidence`, `provenance` (doc id) | `RelationInducer` |
| `GraphDelta` | concepts added, concepts merged/updated, relations added/updated, per-doc errors | orchestrator |

## 6. Ports

**Stage ports** (each SRP, an adapter is chosen by config):

| Port | Responsibility | Stateful | Initial adapters |
|---|---|---|---|
| `Segmenter` | document → ordered `Unit`s | no | turn · block · sentence |
| `Extractor` | `Unit`s → candidate `Mention`s | no | noun-chunk · PoS-pattern · statistical |
| `Scorer` | `Mention`s → `ScoredMention`s + selection | no | embedding-cosine · masked-doc (MDERank) · hierarchical (HCUKE) · graph-centrality |
| `Resolver` | `ScoredMention`s + memory → canonical `Concept`s | **yes** | embedding-NN threshold · clustering · linking |
| `RelationInducer` | `Concept`s + context → `Relation`s | no | Hearst patterns · compound head-modifier · co-occurrence |
| `GraphIntegrator` | apply concepts + relations into accreting graph | **yes** | in-memory · (future) persistent |

**Cross-cutting ports:** `Embedder` (embedding model, used by `Scorer` + `Resolver`),
`ConceptStore` (vector index / memory backing the `Resolver`; in-memory default, pluggable).

**Harness ports:** `Dataset` (yields `Document`s + ground truth), `Metric` (scores a snapshot
against ground truth).

## 7. Assembly: registry + declarative config + factory

1. **Registry** — adapters self-register under `(port, name)` via a decorator
   (`@register(Scorer, "mderank")`); the registry is `{port → {name → adapter_class}}`. A new
   algorithm is a new decorated class — nothing else in the system changes (open/closed).
2. **Declarative config** (pydantic-validated) — a run is one adapter `name` + `params` per port,
   loadable from a TOML file or constructed in code. Example:

   ```toml
   [segmenter]         name = "turn"
   [extractor]         name = "noun-chunk"
   [scorer]            name = "mderank"
     [scorer.params]     encoder = "all-MiniLM-L6-v2"
   [resolver]          name = "embedding-nn"
     [resolver.params]   threshold = 0.88
   [relation_inducer]  name = "hearst"
   [graph_integrator]  name = "in-memory"
   [run]               on_error = "fail"   # D15
   ```

3. **Factory** — validated config → registry lookup → instantiate with params → inject shared deps
   (`Embedder`, `ConceptStore`) → wired orchestrator. This is the single DIP composition root and
   the only place concrete classes are named.

**Reproducibility.** The fully resolved config — adapter names, params, model versions, and a seed —
is serialized and stamped onto every run's output. Re-running a config reproduces the result.

## 8. Failure semantics

The orchestrator takes an `on_error: fail | skip` policy from config (same code path either way):
- **`fail`** (experiment default) — any adapter error aborts the run and surfaces the exception, so a
  crash never silently shrinks the scored corpus.
- **`skip`** (future production/streaming default) — the failing document is skipped and the stream
  continues, so one poison document can't halt the memory.

Regardless of policy, the error is **always** recorded in the `GraphDelta` and the run report —
skips are never silent.

## 9. Experiment harness & evaluation

**Run flow:** load a `Dataset` adapter → fold `process()` over its documents → `snapshot()` the
resulting graph → score with the run's `Metric` adapter(s) → emit a report keyed by the stamped
config.

**Sweeps** are declarative matrices: a sweep spec lists the axes (e.g. 4 scorers × 2 resolvers); the
harness expands the cartesian product into individual configs and runs each, collecting a comparison
table. Experiments are data (git-diffable), not bespoke scripts; programmatic construction remains
available underneath.

**Evaluation strategy — public benchmark per port + deferred integration harness:**
- Extraction + Salience: Inspec, SemEval-2017, SemEval-2010, DUC-2001 (F1@5/10/15).
- Normalization / cross-doc identity: ECB+ (B³/CEAF/MUC), ConEL-2.
- Hierarchy / `IS_A`: SemEval-2016 Task 13 (TExEval-2), hypernym-discovery (edge P/R/F).
- Integration (accreting-graph quality): a small bespoke **intrinsic** harness — redundancy rate,
  cluster purity, hierarchy sanity — built **last**, deferred until components prove out.

Datasets and metrics are adapters (`Dataset`, `Metric` ports), so the harness plugs into the same
architecture rather than being separate scaffolding.

## 10. Project layout

```
lattice/
  pyproject.toml            # uv-managed
  src/lattice/
    core/         # domain model + contracts (§5) — pure, zero external deps
    ports/        # abstract interfaces (§6)
    adapters/     # implementations grouped by port (segmenter/ extractor/ scorer/ resolver/ ...)
    registry/     # registration + lookup (§7.1)
    config/       # pydantic schema + loader (§7.2)
    orchestrator/ # process() fold, error policy (§8), snapshot (§4.2)
    harness/      # dataset + metric adapters, runner, sweep expander, reporting (§9)
  configs/        # declarative run + sweep configs
  tests/          # per-port contract tests + adapter units + integration
  docs/           # this spec
```

## 11. Testing strategy (TDD, tests-first)

- **Contract tests per port** — one shared test suite that *every* adapter for a port must pass.
  This enforces LSP (any adapter is substitutable) and is the backbone that makes swapping safe.
- Core contracts pure-unit-tested.
- Orchestrator integration-tested on a tiny fixture corpus (verifies the streaming fold + delta).
- Benchmark adapters tested against small fixtures before running full datasets.

## 12. SOLID mapping

- **S** — each port has one responsibility; each adapter one implementation.
- **O** — new algorithms are added as adapters without touching the core or other adapters.
- **L** — the per-port contract test suite guarantees adapter substitutability.
- **I** — narrow per-stage ports; no adapter depends on methods it doesn't use.
- **D** — the core and orchestrator depend only on ports; the factory is the single composition root.

## 13. Milestones (build order)

1. **Walking skeleton** — core contracts + all ports + one trivial adapter each + factory +
   orchestrator fold + in-memory stores. `process()` runs end-to-end on a fixture document.
2. **Extraction + salience track** — noun-chunk `Extractor`; `Scorer` adapters (cosine baseline,
   then MDERank, then HCUKE); Inspec/SemEval `Dataset` + F1 `Metric`; first sweep comparing scorers.
3. **Normalization track** — `Resolver` (embedding-NN) + `ConceptStore`; ECB+/ConEL-2 datasets +
   clustering metrics.
4. **Hierarchy track** — `RelationInducer` (Hearst, compound) ; TExEval-2 dataset + edge metrics.
5. **Integration harness** — bespoke intrinsic metrics over a small transcript corpus.
6. **Engine API hardening** — stabilize the public `document → graph` surface for downstream
   consumers (NeuroNote).

## 14. Open questions / deferred

- Concrete embedding model default (likely `all-MiniLM-L6-v2` for parity with the literature) —
  confirm at implementation.
- Whether `PromptRank` (local deterministic seq2seq) is admitted as a `Scorer` adapter, or excluded
  as crossing the "no generative model" line — decide when the salience track is underway.
- Persistent `GraphIntegrator` / `ConceptStore` backends (for real streaming deployment) — deferred
  until the batch engine proves out; the ports are designed to accommodate them.
- Distribution/packaging name if published (`lattice` may be taken on PyPI; namespace then).
```
