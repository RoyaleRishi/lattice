# lattice M6 — Engine API Hardening Design Spec

**Date:** 2026-07-13
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` §13 milestone 6
**Status:** approved design, pre-plan

Milestone 6, the last in the parent spec: stabilize the public
`document → graph` surface for downstream consumers (NeuroNote). Everything
a consumer needs comes from `import lattice`; the accreted graph survives a
process restart; the README's quickstart is executable truth.

## 1. Goal

Today a consumer must reach into `lattice.config.factory`,
`lattice.config.loader`, and `lattice.core.types`, drive `Orchestrator`
directly, and loses the entire graph when the process exits. M6 delivers:
one facade (`lattice.Engine`), two documented profiles, a read-optimized
`GraphView`, versioned save/load with a resume-equivalence guarantee, a
README whose quickstart is executed by the test suite, typing/packaging
polish (`py.typed`, `lattice[ml]` extra, 0.2.0), and a consumer-simulation
test suite that never imports below the top level.

## 2. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Persistence in scope | Yes — `Engine.save`/`Engine.load`, versioned JSON | A memory engine that forgets on restart is unusable for NeuroNote. Still not a persistent *backend* (no DB; parent §14 deferral stands). User decision 2026-07-13. |
| Default stack | `profile="lite"` (dependency-free) with a one-argument toggle to `profile="standard"` (benchmark-validated) | User decision 2026-07-13: dependency-free default, visible toggle. Both profiles share the same pipeline topology (embedding-NN resolver @ 0.90, hearst+compound union) and differ only in extractor and embedder — switching profiles changes quality, never behavior shape. |
| Default error policy | `on_error="fail"` in both profiles | Least surprise for a library: exceptions propagate loudly. Consumers who want poison-document tolerance opt into `"skip"` via `from_config` (documented, including the partial-mutation caveat from the orchestrator docstring). |
| Packaging of ml deps | Mirror the `ml` dependency group into `[project.optional-dependencies]` as the `ml` extra | Dependency groups are dev-only and do not ship; consumers need `lattice[ml]` installable. Mirror only — the pyproject dependency freeze (no NEW deps) is respected. |
| Query surface | Thin `GraphView` wrapper, not methods on core types | Core dataclasses stay pure/dependency-free (parent §5); the view carries lazy indexes and can evolve without touching the domain model. |
| Restore mechanism | New `GraphIntegrator.restore(snapshot)` port method; `ConceptStore` restores via existing `upsert` | Smallest port surface that makes load possible. The resolver keeps no private state (both resolvers lean on the store), so store + integrator restoration is complete. |

## 3. Core/port changes

- `GraphIntegrator` port gains `restore(self, snapshot: GraphSnapshot) -> None`
  — replace internal state with the snapshot's contents. The in-memory
  adapter implements it; the port contract suite gains a
  restore-then-snapshot round-trip test. No other port or core type changes.
- `src/lattice/__init__.py` becomes the public contract:

  ```python
  __all__ = [
      "Concept", "Document", "Engine", "GraphDelta", "GraphSnapshot",
      "GraphView", "Relation", "__version__",
  ]
  ```

  with `__version__ = "0.2.0"` (pyproject bumped to match; a test asserts
  they agree).
- `src/lattice/py.typed` marker ships in the wheel (PEP 561; hatchling
  includes package files by default).

## 4. New modules

### 4.1 `src/lattice/engine.py` — the facade

```python
class Engine:
    def __init__(self, profile: str = "lite") ...
    @classmethod
    def from_config(cls, config: RunConfig | dict | str | Path) -> "Engine"
    @classmethod
    def load(cls, path: str | Path) -> "Engine"

    profile: str | None      # None when built via from_config
    config: RunConfig        # fully resolved, always set

    def ingest(self, document: Document | str, *, id: str | None = None,
               kind: str = "note", timestamp: float | None = None,
               metadata: dict[str, str] | None = None) -> GraphDelta
    def ingest_all(self, documents: Iterable[Document | str]) -> list[GraphDelta]
    def snapshot(self) -> GraphSnapshot
    def view(self) -> GraphView
    def save(self, path: str | Path) -> None
    def reset(self) -> None
```

- Construction: profile name → a `RunConfig` dict built in-module → the
  existing factory (`build_orchestrator`). Unknown profile → `ValueError`
  listing the known ones. `from_config` accepts a `RunConfig`, a plain
  dict, or a TOML path (reusing `load_config`).
- String ingestion: wraps the text in a `Document` with auto id `doc-N`
  (monotonic per-engine counter, persisted by save/load) and
  `timestamp=float(N)`; every field overridable by keyword. Passing a
  `Document` together with any override kwarg → `ValueError` (ambiguous).
- Model laziness: `standard`'s spaCy/sentence-transformer loads happen at
  Engine construction (factory behavior today); `import lattice` itself
  stays light because `engine.py` imports no ml packages. A missing-model
  OSError surfaces at construction with the existing
  `scripts/fetch_models.py` guidance.

**Profiles** (single source of truth: a `_PROFILES` dict in `engine.py`):

| port | lite | standard |
|---|---|---|
| segmenter | block | block |
| extractor | token | noun-chunk |
| scorer | embedding-cosine | embedding-cosine |
| resolver | embedding-nn @ 0.90 | embedding-nn @ 0.90 |
| relation_inducer | union(hearst, compound) | union(hearst, compound) |
| graph_integrator | in-memory | in-memory |
| embedder | hashing | sentence-transformer |
| concept_store | in-memory | in-memory |
| run.on_error | fail | fail |

Every `standard` choice traces to sweep evidence (M2 cosine baseline; M5's
recorded operating point nn@0.90; M4 union). `lite` is documented as
functional-but-unvalidated: same shape, toy extractor/embedder, for smoke
tests, CI, and first contact.

### 4.2 `src/lattice/graph_view.py` — read surface

```python
class GraphView:
    def __init__(self, snapshot: GraphSnapshot)
    def concepts(self) -> tuple[Concept, ...]
    def find_concept(self, label: str) -> Concept | None          # casefolded exact match
    def relations(self, type: str | None = None) -> tuple[Relation, ...]
    def neighbors(self, concept_id: str, type: str | None = None
                  ) -> tuple[tuple[Relation, Concept], ...]
```

Immutable over one snapshot; label and adjacency indexes built lazily on
first use. `neighbors` returns `(relation, other_concept)` pairs for every
relation touching the id (direction readable from the relation itself),
sorted by `(relation.type, other.id)` for determinism. Unknown concept_id →
empty tuple (not an error: queries are reads, not assertions).

### 4.3 Persistence format (v1)

`Engine.save(path)` writes JSON:

```json
{
  "format_version": 1,
  "lattice_version": "0.2.0",
  "profile": "lite",
  "config": { ...fully resolved RunConfig dump... },
  "document_counter": 2,
  "concepts": [{"id", "label", "embedding", "first_seen", "updated_at"}, ...],
  "relations": [{"type", "source_id", "target_id", "confidence", "provenance"}, ...]
}
```

`Engine.load(path)`: `format_version != 1` → `ValueError` naming the found
version. Otherwise rebuild the engine from the stored config
(`RunConfig.model_validate`), `restore` the integrator with the
reconstructed `GraphSnapshot`, `upsert` every concept into the concept
store, restore the document counter, set `profile` from the file. A stored
config naming unregistered adapters fails with the registry's normal error.
Concept/relation field values round-trip exactly (JSON floats round-trip;
tuples reconstructed).

**Resume-equivalence guarantee (the contract, test-enforced):**
`ingest(A); ingest(B); save; load; ingest(C)` produces a snapshot equal
(`==` on the frozen dataclasses) to `ingest(A); ingest(B); ingest(C)` in
one process — including auto-generated ids, because the counter persists.

## 5. README.md

First README for the repo. Sections: what lattice is (from parent §1, three
sentences); install (`uv add lattice` framing plus the `lattice[ml]` extra
and `scripts/fetch_models.py` for `standard`); **quickstart** (lite,
five-ish lines, executed verbatim by a test — so it only uses what lite
actually produces: the token extractor emits single words, so the example
queries a single-word concept and the text explains that `standard` yields
real noun phrases); the profile toggle table with one-line rationale per
choice and pointers to the M2–M5 specs and their headline results;
persistence example (save/load three-liner); stability policy (pre-1.0:
`__all__` is the contract, minor versions may break it with a changelog
note, `format_version` guards saves); pointer to
`docs/2026-07-05-lattice-architecture-design.md` for the architecture.

## 6. Error handling

- Unknown profile → `ValueError` listing known profiles.
- `Document` + override kwargs to `ingest` → `ValueError`.
- Save-file `format_version` mismatch → `ValueError` naming found/expected.
- Missing ml deps for `standard` → the underlying ImportError propagates
  from adapter construction, with the README documenting
  `pip install "lattice[ml]"`; missing models → existing OSError +
  `fetch_models.py` guidance.
- Corrupt save JSON → `json.JSONDecodeError` propagates (a broken file
  should look broken, not half-load).

## 7. Testing strategy

- **Consumer simulation** (`tests/api/test_public_surface.py`): imports
  ONLY top-level `lattice`. Lite session flow (ingest strings → deltas →
  view queries); `ingest` auto-id/timestamp and override semantics
  (including the Document+kwargs ValueError); `ingest_all`; profile
  attribute; unknown-profile ValueError; `from_config` with a dict and with
  a TOML file (tmp_path).
- **Persistence** (`tests/api/test_persistence.py`): resume-equivalence
  exact-equality test on lite; counter persistence (post-load auto ids
  continue, no collisions); format_version mismatch ValueError; saved-file
  shape (keys, format_version 1); save → load → save byte-identical second
  file (idempotence).
- **GraphView** (`tests/api/test_graph_view.py`): find_concept casefold,
  relations type filter, neighbors direction + determinism + unknown-id
  empty result; built against a hand-made snapshot, no pipeline.
- **README executability** (`tests/api/test_readme.py`): extract the first
  fenced `python` block from README.md and `exec` it in a tmp cwd; assert
  it runs and defines the names it claims.
- **Port contract:** restore-then-snapshot round-trip added to the
  GraphIntegrator contract suite (in-memory adapter).
- **Version/packaging** (`tests/api/test_packaging.py`): `lattice.__version__`
  == pyproject version (read via tomllib); `src/lattice/py.typed` exists;
  `__all__` names all importable.
- **ml path** (`@pytest.mark.ml`): `Engine(profile="standard")` constructs
  and ingests one string; importorskip + OSError-skip (M3 pattern).
- Existing 385 tests must pass unchanged.

## 8. Success criteria

1. **Consumer suite green** importing only top-level `lattice`; the
   resume-equivalence test passes with exact `==`.
2. **README quickstart executes verbatim** via the extraction test.
3. **Packaging:** `uv build` succeeds; the wheel contains `py.typed`;
   versions agree (0.2.0).
4. **No regression:** full suite (existing 385 + new) passes; ruff clean;
   no changes to any benchmark adapter, config, or recorded result.
5. **Standard profile** constructs and ingests under the ml stack
   (ml-marked test, skipped where models are absent).

## 9. Explicitly deferred

- Persistent backends (DB/vector-index stores) — parent §14, unchanged.
- PyPI publication and the name-collision question — parent §14.
- Async/streaming-push API, CLI entry point.
- Save-format migrations (the `format_version` field exists so v2 can).
- Graph mutation via the API (delete/merge concepts) — the engine accretes;
  curation is a future consumer-driven feature.
- M7+: there is no M7 — the parent spec's milestone list ends here; next
  steps belong to consumer-driven iteration.
