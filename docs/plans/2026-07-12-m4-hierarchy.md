# M4 Hierarchy Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give lattice its first real `IS_A` edges — compound head-modifier and Hearst-pattern `RelationInducer` adapters plus a union combinator, benchmarked on all six TExEval-2 English golds with an edge-P/R/F snapshot metric.

**Architecture:** Everything sits behind existing ports (M4 spec `docs/2026-07-12-m4-hierarchy-design.md`; zero core changes). A `taxonomy` dataset yields a glossary document first (all gold terms co-resident, so compound closure is intra-document and the port stays stateless), then one Wikipedia-summary document per term. A `gazetteer` extractor dictionary-matches the term list; `exact-label` resolves; the relation inducer is the sweep axis; `edge-f1` scores the final snapshot against gold edges.

**Tech Stack:** Python 3.13 (uv-managed), pydantic config, pytest, stdlib-only fetch script. No new dependencies.

## Global Constraints

- `pyproject.toml` is FROZEN — no new dependencies of any kind.
- No network access and no model/dataset downloads inside tests or adapters. Only `scripts/fetch_texeval.py` touches the network, and only when run explicitly.
- `data/`, `reports/`, and `.superpowers/` are gitignored — never commit datasets, sweep reports, or ledgers. Never commit anything under `data/texeval/`.
- Registered adapter names are load-bearing (configs reference them): `compound`, `hearst`, `union`, `gazetteer`, `taxonomy`, `edge-f1`.
- Every new adapter module must be imported in `src/lattice/adapters/__init__.py` (import-time registration) — each task adds its own line.
- Run tests with `uv run --no-sync pytest <path> -q`. If imports mysteriously fail (site-packages .pth hidden-flag quirk on this Mac), first run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null`.
- Lint with `uv run --no-sync ruff check .` before every commit.
- Sweeps run as `uv run --no-sync python -m lattice.harness --sweep <toml> <out_dir>` (the `--sweep` flag on `lattice.harness`; the `lattice.harness.sweep` module path is a silent no-op).
- Commit messages: `feat:`/`test:`/`docs:` prefix, ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
src/lattice/adapters/relation_inducer/compound.py    Task 1  "compound"
src/lattice/adapters/relation_inducer/hearst.py      Task 2  "hearst"
src/lattice/adapters/relation_inducer/union.py       Task 3  "union"
src/lattice/adapters/extractor/gazetteer.py          Task 4  "gazetteer"
src/lattice/adapters/dataset/taxonomy.py             Task 5  "taxonomy"
src/lattice/adapters/metric/edge_f1.py               Task 6  "edge-f1"
scripts/fetch_texeval.py                             Task 7
configs/m4-<key>-sweep.toml (6 files)                Task 8
tests/adapters/test_compound_inducer.py              Task 1
tests/adapters/test_hearst_inducer.py                Task 2
tests/adapters/test_union_inducer.py                 Task 3
tests/adapters/test_gazetteer_extractor.py           Task 4
tests/adapters/test_taxonomy_dataset.py              Task 5
tests/adapters/test_edge_f1_metric.py                Task 6
tests/scripts/test_fetch_texeval.py                  Task 7
tests/harness/test_m4_e2e.py                         Task 8
tests/fixtures/mini_texeval/contract/terms.txt       Task 4
tests/fixtures/mini_texeval/toy/{terms.txt,gold_edges.jsonl,documents.jsonl}  Task 5
```

Task 9 is the exit-criteria run (real fetch + six sweeps + adjudication); it produces no committed code.

**Suggested implementers** (standing arrangement): Tasks 1, 3, 4, 5, 6 — haiku (complete code below, transcription + verify). Tasks 2, 7, 8 — sonnet (algorithmic subtlety / network handling / harness integration). Task 9 — orchestrator/observer. All task reviews: sonnet.

---

### Task 1: `compound` RelationInducer

**Files:**
- Create: `src/lattice/adapters/relation_inducer/compound.py`
- Modify: `src/lattice/adapters/__init__.py` (add import)
- Test: `tests/adapters/test_compound_inducer.py`

**Interfaces:**
- Consumes: `RelationInducer` port (`induce(resolutions, units, document) -> list[Relation]`), `Relation`, `tests/helpers.make_resolution/make_unit/make_document`, `tests/contracts/relation_inducer_contract.RelationInducerContract`.
- Produces: registered adapter `("compound", RelationInducer)`, constructor `CompoundInducer(longest_only: bool = True)`. Task 3's union instantiates it by name; Task 8's configs reference `name = "compound"`.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_compound_inducer.py`:

```python
from lattice.adapters.relation_inducer.compound import CompoundInducer
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestCompoundContract(RelationInducerContract):
    def make_inducer(self):
        return CompoundInducer()


def _induce(surfaces: list[str], longest_only: bool = True):
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text=" ".join(surfaces))]
    resolutions = [make_resolution(surface=s, unit_id="d1:u0") for s in surfaces]
    inducer = CompoundInducer(longest_only=longest_only)
    relations = inducer.induce(resolutions, units, document)
    label = {r.concept.id: r.concept.label for r in resolutions}
    return [(label[rel.source_id], label[rel.target_id], rel) for rel in relations]


def test_multiword_label_links_to_its_head_suffix():
    edges = _induce(["olive oil", "oil"])
    assert [(s, t) for s, t, _ in edges] == [("olive oil", "oil")]
    relation = edges[0][2]
    assert relation.type == "IS_A"
    assert relation.confidence == 1.0
    assert relation.provenance == "d1"


def test_longest_matching_suffix_wins_by_default():
    edges = _induce(["extra virgin olive oil", "olive oil", "oil"])
    pairs = {(s, t) for s, t, _ in edges}
    # the compound links only to its longest matching suffix; the shorter
    # compound links to its own head.
    assert pairs == {
        ("extra virgin olive oil", "olive oil"),
        ("olive oil", "oil"),
    }


def test_longest_only_false_emits_every_matching_suffix():
    edges = _induce(["extra virgin olive oil", "olive oil", "oil"],
                    longest_only=False)
    pairs = {(s, t) for s, t, _ in edges}
    assert pairs == {
        ("extra virgin olive oil", "olive oil"),
        ("extra virgin olive oil", "oil"),
        ("olive oil", "oil"),
    }


def test_suffix_must_be_whole_word():
    assert _induce(["pineapple", "apple"]) == []


def test_single_word_labels_are_inert():
    assert _induce(["oil", "fat"]) == []


def test_missing_head_yields_no_edge():
    assert _induce(["olive oil", "fat"]) == []


def test_duplicate_resolutions_of_one_concept_emit_one_edge():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="olive oil oil olive oil")]
    resolutions = [
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
        make_resolution(surface="olive oil", unit_id="d1:u0"),
    ]
    relations = CompoundInducer().induce(resolutions, units, document)
    assert len(relations) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_compound_inducer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.relation_inducer.compound'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/relation_inducer/compound.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Concept, Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register


@register(RelationInducer, "compound")
class CompoundInducer(RelationInducer):
    """Head-modifier IS_A induction (M4 spec §4.1): a multiword concept is a
    kind of its head — "olive oil" IS_A "oil" — whenever a word-suffix of its
    label is itself a concept resolved in the same document. On the taxonomy
    benchmark's glossary document every gold term is co-resident, so this
    computes the complete compound closure of the term list there. Suffixes
    are whole words, never substrings ("pineapple" is not an "apple")."""

    def __init__(self, longest_only: bool = True):
        self.longest_only = longest_only

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        by_label: dict[str, Concept] = {}
        for resolution in resolutions:
            by_label.setdefault(resolution.concept.label, resolution.concept)
        relations: list[Relation] = []
        for label, concept in sorted(by_label.items()):
            words = label.split()
            for i in range(1, len(words)):
                target = by_label.get(" ".join(words[i:]))
                if target is None:
                    continue
                relations.append(
                    Relation(
                        type="IS_A",
                        source_id=concept.id,
                        target_id=target.id,
                        confidence=1.0,
                        provenance=document.id,
                    )
                )
                if self.longest_only:
                    break
        return relations
```

In `src/lattice/adapters/__init__.py`, change the relation_inducer import line to:

```python
from lattice.adapters.relation_inducer import co_occurrence, compound  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_compound_inducer.py -q`
Expected: PASS (10 tests: 3 contract + 7 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/relation_inducer/compound.py src/lattice/adapters/__init__.py tests/adapters/test_compound_inducer.py
git commit -m "feat: add compound head-modifier IS_A inducer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `hearst` RelationInducer

**Files:**
- Create: `src/lattice/adapters/relation_inducer/hearst.py`
- Modify: `src/lattice/adapters/__init__.py` (extend import)
- Test: `tests/adapters/test_hearst_inducer.py`

**Interfaces:**
- Consumes: `RelationInducer` port, `Relation`, `Mention`, `ScoredMention`, `Resolution`, `tests/helpers.make_concept/make_document/make_unit`, `RelationInducerContract`.
- Produces: registered adapter `("hearst", RelationInducer)`, constructor `HearstInducer(patterns: list[str] | None = None, copula: bool = True)`. Pattern names: `such-as`, `such-np-as`, `including`, `especially`, `and-other`, `or-other`, `copula`. Task 3's union instantiates it by name; Task 8's configs reference `name = "hearst"`.

**Key design constraints (spec §4.2):** anchors are resolved mention spans; only **adjacent** anchor pairs within a unit are examined; the connector text between the two spans must **fullmatch** a pattern regex (this is what provides locality — a cross-sentence `". And other"` connector fails); forward coordination walking applies to hypernym-left patterns only; `tests/helpers.make_resolution` has a fake `(0, 7)` span, so tests build real-span resolutions with a local helper. Note the overlapping-span guard is load-bearing: the shared `RelationInducerContract` fixture resolves two mentions with identical spans, and the guard is what keeps the contract green.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_hearst_inducer.py`:

```python
import pytest

from lattice.adapters.relation_inducer.hearst import HearstInducer
from lattice.core.types import Mention, Resolution, ScoredMention
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_concept, make_document, make_unit


class TestHearstContract(RelationInducerContract):
    def make_inducer(self):
        return HearstInducer()


def _resolution(surface: str, unit_id: str, start: int) -> Resolution:
    mention = Mention(
        surface=surface,
        unit_id=unit_id,
        span=(start, start + len(surface)),
        context=surface,
    )
    return Resolution(
        concept=make_concept(id=f"c:{surface.lower()}", label=surface.lower()),
        mention=ScoredMention(mention=mention, salience=1.0, selected=True),
        is_new=True,
    )


def _edges(text: str, surfaces: list[str], **kwargs) -> set[tuple[str, str]]:
    """Resolve each surface at its first occurrence in `text`, run the
    inducer, and return (hyponym label, hypernym label) pairs."""
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text=text)]
    resolutions = []
    cursor: dict[str, int] = {}
    for surface in surfaces:
        start = text.index(surface, cursor.get(surface, 0))
        cursor[surface] = start + 1
        resolutions.append(_resolution(surface, "d1:u0", start))
    relations = HearstInducer(**kwargs).induce(resolutions, units, document)
    label = {r.concept.id: r.concept.label for r in resolutions}
    return {(label[rel.source_id], label[rel.target_id]) for rel in relations}


def test_such_as():
    assert _edges("fats such as olive oil are prized.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_such_as_with_comma():
    assert _edges("fats, such as olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_such_np_as():
    assert _edges("such fats as olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_bare_as_without_such_prefix_does_not_match():
    assert _edges("fats as olive oil.", ["fats", "olive oil"]) == set()


def test_including():
    assert _edges("fats, including olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_especially():
    assert _edges("fats, especially olive oil.", ["fats", "olive oil"]) == {
        ("olive oil", "fats")
    }


def test_and_other():
    assert _edges("olive oil and other fats.", ["olive oil", "fats"]) == {
        ("olive oil", "fats")
    }


def test_or_other():
    assert _edges("olive oil or other fats.", ["olive oil", "fats"]) == {
        ("olive oil", "fats")
    }


def test_copula_variants():
    for text in [
        "olive oil is a fat.",
        "olive oil is an fat.",
        "olive oil is a kind of fat.",
        "olive oil is a type of fat.",
    ]:
        assert _edges(text, ["olive oil", "fat"]) == {("olive oil", "fat")}, text


def test_copula_flag_off_drops_copula_edges():
    assert _edges("olive oil is a fat.", ["olive oil", "fat"], copula=False) == set()


def test_explicit_patterns_select_a_subset():
    # AMENDED during execution (plan defect): the original fixture text
    # ("fats such as olive oil. canola is a fat.") anchors "fat" via
    # text.index inside "fats" at position 0; the overlap guard then eats
    # the copula pair. "fat" must occur before "fats" in the text. Same
    # intent: copula-only selection finds the copula edge and excludes the
    # such-as edge.
    text = "canola is a fat. fats such as olive oil."
    surfaces = ["canola", "fat", "fats", "olive oil"]
    assert _edges(text, surfaces, patterns=["copula"]) == {("canola", "fat")}


def test_unknown_pattern_name_raises():
    with pytest.raises(ValueError, match="unknown hearst pattern"):
        HearstInducer(patterns=["cherry-picked"])


def test_coordination_walking():
    text = "fats, such as olive oil, canola and margarine, are prized."
    surfaces = ["fats", "olive oil", "canola", "margarine"]
    assert _edges(text, surfaces) == {
        ("olive oil", "fats"),
        ("canola", "fats"),
        ("margarine", "fats"),
    }


def test_intervening_text_kills_the_match():
    assert _edges(
        "fats are found in stores such as delis, olive oil.", ["fats", "olive oil"]
    ) == set()


def test_cross_sentence_connector_fails():
    assert _edges("we saw fats. Such as olive oil.", ["fats", "olive oil"]) == set()


def test_same_concept_pair_is_skipped():
    # both surfaces resolve to the same lowercased concept
    assert _edges("Fat is a fat.", ["Fat", "fat"]) == set()


def test_cross_unit_pairs_never_match():
    document = make_document(id="d1")
    units = [
        make_unit(id="d1:u0", document_id="d1", text="fats such as"),
        make_unit(id="d1:u1", document_id="d1", text="olive oil", order=1),
    ]
    resolutions = [
        _resolution("fats", "d1:u0", 0),
        _resolution("olive oil", "d1:u1", 0),
    ]
    assert HearstInducer().induce(resolutions, units, document) == []


def test_relation_shape():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="fats such as olive oil")]
    resolutions = [_resolution("fats", "d1:u0", 0), _resolution("olive oil", "d1:u0", 13)]
    [relation] = HearstInducer().induce(resolutions, units, document)
    assert relation.type == "IS_A"
    assert relation.confidence == 1.0
    assert relation.provenance == "d1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_hearst_inducer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.relation_inducer.hearst'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/relation_inducer/hearst.py`:

```python
import re
from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import register

# Separators accepted between coordinated hyponyms ("x1, x2 and x3").
_COORD = re.compile(r",\s+|,?\s+(?:and|or)\s+", re.IGNORECASE)


class _Pattern:
    """One lexico-syntactic pattern: `connector` is fullmatched against the
    text between two adjacent anchor spans; `hyper` says which anchor is the
    hypernym; `coordination` enables forward hyponym-list walking; `prefix`
    (if set) must match immediately before the left anchor."""

    __slots__ = ("name", "connector", "hyper", "coordination", "prefix")

    def __init__(
        self,
        name: str,
        connector: str,
        hyper: str,
        coordination: bool = False,
        prefix: str | None = None,
    ):
        self.name = name
        self.connector = re.compile(connector, re.IGNORECASE)
        self.hyper = hyper  # "left" | "right"
        self.coordination = coordination
        self.prefix = re.compile(prefix, re.IGNORECASE) if prefix else None


_BUILTIN = [
    _Pattern("such-as", r",?\s+such\s+as\s+", hyper="left", coordination=True),
    _Pattern(
        "such-np-as", r"\s+as\s+", hyper="left", coordination=True,
        prefix=r"\bsuch\s+\Z",
    ),
    _Pattern("including", r",?\s+including\s+", hyper="left", coordination=True),
    _Pattern("especially", r",?\s+especially\s+", hyper="left", coordination=True),
    _Pattern("and-other", r",?\s+and\s+other\s+", hyper="right"),
    _Pattern("or-other", r",?\s+or\s+other\s+", hyper="right"),
    _Pattern(
        "copula", r"\s+(?:is|are)\s+an?\s+(?:(?:kind|type)\s+of\s+)?",
        hyper="right",
    ),
]


@register(RelationInducer, "hearst")
class HearstInducer(RelationInducer):
    """Anchored Hearst-pattern IS_A induction (M4 spec §4.2). Resolved
    mention spans are the NPs — no chunking; only adjacent anchors in a unit
    are paired, and the connector between them must match a pattern exactly,
    so locality needs no sentence segmentation (a cross-sentence connector
    like ". Such as " never fullmatches). `patterns` selects a subset by
    name; when it is None the full set is used, minus `copula` if
    copula=False (definitional corpora make the copula the productive
    pattern, hence on by default)."""

    def __init__(self, patterns: list[str] | None = None, copula: bool = True):
        available = {p.name: p for p in _BUILTIN}
        if patterns is None:
            selected = [
                p for p in _BUILTIN if copula or p.name != "copula"
            ]
        else:
            unknown = sorted(set(patterns) - set(available))
            if unknown:
                raise ValueError(f"unknown hearst pattern(s): {unknown}")
            selected = [available[name] for name in patterns]
        self._patterns = selected

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        text_of = {u.id: u.text for u in units}
        by_unit: dict[str, list[Resolution]] = {}
        for resolution in resolutions:
            unit_id = resolution.mention.mention.unit_id
            by_unit.setdefault(unit_id, []).append(resolution)
        relations: list[Relation] = []
        seen: set[tuple[str, str]] = set()
        for unit_id in sorted(by_unit):
            text = text_of.get(unit_id)
            if text is None:
                continue
            anchors = sorted(by_unit[unit_id], key=lambda r: r.mention.mention.span)
            for i in range(len(anchors) - 1):
                left, right = anchors[i], anchors[i + 1]
                left_end = left.mention.mention.span[1]
                right_start = right.mention.mention.span[0]
                if right_start < left_end:
                    continue  # overlapping anchors: no clean connector
                between = text[left_end:right_start]
                for pattern in self._patterns:
                    if not pattern.connector.fullmatch(between):
                        continue
                    left_start = left.mention.mention.span[0]
                    if pattern.prefix and not pattern.prefix.search(
                        text[:left_start]
                    ):
                        continue
                    if pattern.hyper == "left":
                        pairs = [(right, left)]
                        if pattern.coordination:
                            pairs += _walk_coordination(anchors, i + 1, left, text)
                    else:
                        pairs = [(left, right)]
                    for hypo, hyper in pairs:
                        edge = (hypo.concept.id, hyper.concept.id)
                        if edge[0] == edge[1] or edge in seen:
                            continue
                        seen.add(edge)
                        relations.append(
                            Relation(
                                type="IS_A",
                                source_id=edge[0],
                                target_id=edge[1],
                                confidence=1.0,
                                provenance=document.id,
                            )
                        )
        return relations


def _walk_coordination(
    anchors: list[Resolution], first_hypo: int, hyper: Resolution, text: str
) -> list[tuple[Resolution, Resolution]]:
    """After `hyper CONN anchors[first_hypo]`, keep consuming following
    anchors while they are separated only by coordination glue."""
    pairs: list[tuple[Resolution, Resolution]] = []
    for j in range(first_hypo, len(anchors) - 1):
        previous_end = anchors[j].mention.mention.span[1]
        next_start = anchors[j + 1].mention.mention.span[0]
        if next_start < previous_end:
            break
        if not _COORD.fullmatch(text[previous_end:next_start]):
            break
        pairs.append((anchors[j + 1], hyper))
    return pairs
```

In `src/lattice/adapters/__init__.py`, extend the relation_inducer import line to:

```python
from lattice.adapters.relation_inducer import co_occurrence, compound, hearst  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_hearst_inducer.py -q`
Expected: PASS (21 tests: 3 contract + 18 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/relation_inducer/hearst.py src/lattice/adapters/__init__.py tests/adapters/test_hearst_inducer.py
git commit -m "feat: add anchored Hearst-pattern IS_A inducer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `union` RelationInducer combinator

**Files:**
- Create: `src/lattice/adapters/relation_inducer/union.py`
- Modify: `src/lattice/adapters/__init__.py` (extend import)
- Test: `tests/adapters/test_union_inducer.py`

**Interfaces:**
- Consumes: `registry.lookup(RelationInducer, name)` and `RegistryError` from `lattice.registry.registry`; Tasks 1–2's registered `compound`/`hearst`.
- Produces: registered adapter `("union", RelationInducer)`, constructor `UnionInducer(members: list[dict])` where each member dict is `{"name": str}` or `{"name": str, "params": dict}`. Task 8's configs reference `name = "union"` with `params.members`.

Note: spec §7 says an unknown member name surfaces "the registry's normal KeyError" — in this codebase that error is `RegistryError` (the registry's lookup exception); same fail-fast intent, no spec amendment needed.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_union_inducer.py`:

```python
import pytest

from lattice.adapters.relation_inducer.union import UnionInducer
from lattice.registry.registry import RegistryError
from tests.contracts.relation_inducer_contract import RelationInducerContract
from tests.helpers import make_document, make_resolution, make_unit


class TestUnionContract(RelationInducerContract):
    def make_inducer(self):
        return UnionInducer(
            members=[{"name": "hearst"}, {"name": "compound"}]
        )


def test_union_concatenates_member_outputs_in_member_order():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="olive oil oil")]
    resolutions = [
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
    ]
    # compound alone finds the edge; hearst finds nothing (fake spans overlap)
    inducer = UnionInducer(members=[{"name": "hearst"}, {"name": "compound"}])
    relations = inducer.induce(resolutions, units, document)
    assert len(relations) == 1
    assert relations[0].type == "IS_A"


def test_member_params_are_forwarded():
    inducer = UnionInducer(
        members=[{"name": "compound", "params": {"longest_only": False}}]
    )
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="x")]
    resolutions = [
        make_resolution(surface="extra virgin olive oil", unit_id="d1:u0"),
        make_resolution(surface="olive oil", unit_id="d1:u0"),
        make_resolution(surface="oil", unit_id="d1:u0"),
    ]
    # longest_only=False emits every matching suffix: 3 edges, not 2
    assert len(inducer.induce(resolutions, units, document)) == 3


def test_unknown_member_name_fails_at_construction():
    with pytest.raises(RegistryError, match="no adapter 'nope'"):
        UnionInducer(members=[{"name": "nope"}])


def test_empty_members_yield_no_relations():
    document = make_document(id="d1")
    units = [make_unit(id="d1:u0", document_id="d1", text="x")]
    inducer = UnionInducer(members=[])
    assert inducer.induce([], units, document) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_union_inducer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.relation_inducer.union'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/relation_inducer/union.py`:

```python
from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit
from lattice.ports import RelationInducer
from lattice.registry.registry import lookup, register


@register(RelationInducer, "union")
class UnionInducer(RelationInducer):
    """Combinator (M4 spec §4.3): runs member inducers in order and
    concatenates their relations; the graph integrator dedupes by
    (type, source, target). Members are instantiated from the registry at
    construction time — an unknown name fails fast with RegistryError.
    Members needing shared-dep injection (embedder/concept_store) are out of
    scope: params are the only constructor arguments forwarded."""

    def __init__(self, members: list[dict]):
        self._members: list[RelationInducer] = []
        for spec in members:
            adapter_cls = lookup(RelationInducer, spec["name"])
            self._members.append(adapter_cls(**spec.get("params", {})))

    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]:
        relations: list[Relation] = []
        for member in self._members:
            relations.extend(member.induce(resolutions, units, document))
        return relations
```

In `src/lattice/adapters/__init__.py`, extend the relation_inducer import line to:

```python
from lattice.adapters.relation_inducer import (  # noqa: F401
    co_occurrence,
    compound,
    hearst,
    union,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_union_inducer.py -q`
Expected: PASS (7 tests: 3 contract + 4 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/relation_inducer/union.py src/lattice/adapters/__init__.py tests/adapters/test_union_inducer.py
git commit -m "feat: add union relation-inducer combinator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `gazetteer` Extractor

**Files:**
- Create: `src/lattice/adapters/extractor/gazetteer.py`
- Create: `tests/fixtures/mini_texeval/contract/terms.txt`
- Modify: `src/lattice/adapters/__init__.py` (extend import)
- Test: `tests/adapters/test_gazetteer_extractor.py`

**Interfaces:**
- Consumes: `Extractor` port (`extract(units) -> list[Mention]`), `Mention`, `ExtractorContract`, `make_unit`.
- Produces: registered adapter `("gazetteer", Extractor)`, constructor `GazetteerExtractor(root: str, gold: str)` reading `{root}/{gold}/terms.txt` (one lowercase term per line — the format Task 7's fetch script and Task 5's toy fixture emit). Task 8's configs reference `name = "gazetteer"`.

- [ ] **Step 1: Create the contract fixture**

Create `tests/fixtures/mini_texeval/contract/terms.txt` (exactly these three lines — the ExtractorContract's fixed sentences must produce mentions):

```
vector stores
embeddings
encoders
```

- [ ] **Step 2: Write the failing test**

Create `tests/adapters/test_gazetteer_extractor.py`:

```python
import pytest

from lattice.adapters.extractor.gazetteer import GazetteerExtractor
from tests.contracts.extractor_contract import ExtractorContract
from tests.helpers import make_unit

ROOT = "tests/fixtures/mini_texeval"


class TestGazetteerContract(ExtractorContract):
    def make_extractor(self):
        return GazetteerExtractor(root=ROOT, gold="contract")


def _extract(text: str, terms: list[str], tmp_path):
    gold_dir = tmp_path / "g"
    gold_dir.mkdir()
    (gold_dir / "terms.txt").write_text("\n".join(terms) + "\n")
    extractor = GazetteerExtractor(root=str(tmp_path), gold="g")
    return extractor.extract([make_unit(id="d:u0", text=text)])


def test_longest_match_wins(tmp_path):
    mentions = _extract("olive oil is nice", ["oil", "olive oil"], tmp_path)
    assert [m.surface for m in mentions] == ["olive oil"]
    assert mentions[0].span == (0, 9)


def test_case_insensitive_and_surface_keeps_original_case(tmp_path):
    [mention] = _extract("Olive Oil!", ["olive oil"], tmp_path)
    assert mention.surface == "Olive Oil"
    assert mention.span == (0, 9)


def test_whole_word_boundaries(tmp_path):
    assert _extract("pineapples and oils", ["apple", "oil"], tmp_path) == []


def test_hyphen_neighbors_do_not_match(tmp_path):
    # "(?<!\\w)/(?!\\w)" treats "-" as a boundary; matching inside a
    # hyphenated compound is allowed, matching inside a word is not.
    [mention] = _extract("olive-oil blend", ["oil"], tmp_path)
    assert mention.span == (6, 9)


def test_matches_after_punctuation(tmp_path):
    [mention] = _extract("we love oil.", ["oil"], tmp_path)
    assert mention.span == (8, 11)


def test_multiple_occurrences_all_reported(tmp_path):
    mentions = _extract("oil, oil and oil", ["oil"], tmp_path)
    assert [m.span for m in mentions] == [(0, 3), (5, 8), (13, 16)]


def test_context_windows_the_match(tmp_path):
    text = "x" * 100 + " oil " + "y" * 100
    [mention] = _extract(text, ["oil"], tmp_path)
    start, end = mention.span
    assert mention.context == text[start - 40 : end + 40]


def test_missing_terms_file_names_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_texeval"):
        GazetteerExtractor(root=str(tmp_path), gold="absent")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_gazetteer_extractor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.extractor.gazetteer'`

- [ ] **Step 4: Write the implementation**

Create `src/lattice/adapters/extractor/gazetteer.py`:

```python
import re
from collections.abc import Sequence
from pathlib import Path

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register


@register(Extractor, "gazetteer")
class GazetteerExtractor(Extractor):
    """Dictionary extractor (M4 spec §4.5): case-insensitive, whole-word,
    longest-match scan of a fixed term list against unit text. Boundaries
    are non-word-char lookarounds rather than \\b because terms may contain
    hyphens and punctuation. Must be configured with the same root/gold as
    the taxonomy dataset."""

    def __init__(self, root: str, gold: str):
        path = Path(root) / gold / "terms.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run --no-sync python "
                "scripts/fetch_texeval.py` first"
            )
        terms = [line.strip() for line in path.read_text().splitlines()]
        terms = [t for t in terms if t]
        alternation = "|".join(
            re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t))
        )
        self._regex = re.compile(
            rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE
        )

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            for match in self._regex.finditer(unit.text):
                start, end = match.span()
                mentions.append(
                    Mention(
                        surface=match.group(0),
                        unit_id=unit.id,
                        span=(start, end),
                        context=unit.text[max(0, start - 40) : end + 40],
                    )
                )
        return mentions
```

In `src/lattice/adapters/__init__.py`, extend the extractor import line to:

```python
from lattice.adapters.extractor import (  # noqa: F401
    gazetteer,
    gold_mentions,
    noun_chunk,
    token,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_gazetteer_extractor.py -q`
Expected: PASS (11 tests: 3 contract + 8 unit)

- [ ] **Step 6: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/extractor/gazetteer.py src/lattice/adapters/__init__.py tests/adapters/test_gazetteer_extractor.py tests/fixtures/mini_texeval/contract/terms.txt
git commit -m "feat: add gazetteer dictionary extractor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `taxonomy` Dataset + toy fixture

**Files:**
- Create: `src/lattice/adapters/dataset/taxonomy.py`
- Create: `tests/fixtures/mini_texeval/toy/terms.txt`
- Create: `tests/fixtures/mini_texeval/toy/gold_edges.jsonl`
- Create: `tests/fixtures/mini_texeval/toy/documents.jsonl`
- Modify: `src/lattice/adapters/__init__.py` (extend import)
- Test: `tests/adapters/test_taxonomy_dataset.py`

**Interfaces:**
- Consumes: `Dataset` port (`documents() -> Iterator[Document]`, `ground_truth() -> dict`), `DatasetContract`.
- Produces: registered adapter `("taxonomy", Dataset)`, constructor `TaxonomyDataset(root: str, gold: str, limit: int | None = None)`; ground truth `{"is_a_edges": list[list[str]], "terms": list[str]}` — the shape Task 6's edge-f1 consumes. The toy fixture is shared by Task 8's e2e; its texts are hand-verified: compound alone scores F1 0.75, hearst alone 0.75, union exactly 1.0.

- [ ] **Step 1: Create the toy fixture**

Create `tests/fixtures/mini_texeval/toy/terms.txt`:

```
oil
olive oil
vegetable oil
sunflower oil
fat
```

Create `tests/fixtures/mini_texeval/toy/gold_edges.jsonl`:

```
["olive oil", "oil"]
["vegetable oil", "oil"]
["sunflower oil", "oil"]
["olive oil", "fat"]
["sunflower oil", "fat"]
```

Create `tests/fixtures/mini_texeval/toy/documents.jsonl` (exactly 3 lines):

```
{"id": "toy:glossary", "kind": "terminology", "text": "oil\nolive oil\nvegetable oil\nsunflower oil\nfat"}
{"id": "toy:olive-oil", "kind": "article", "term": "olive oil", "text": "Olive oil is a fat. Many kinds of fat, such as olive oil and sunflower oil, appear in cooking."}
{"id": "toy:vegetable-oil", "kind": "article", "term": "vegetable oil", "text": "Vegetable oil is an oil extracted from plants."}
```

(Why these texts: the glossary gives compound its closure — olive/vegetable/sunflower oil → oil. Doc 2 gives hearst `olive oil IS_A fat` via copula and `olive oil, sunflower oil IS_A fat` via such-as coordination. Doc 3 gives hearst `vegetable oil IS_A oil` via copula — an edge compound also finds, exercising integrator dedupe. Union = exactly the 5 gold edges.)

- [ ] **Step 2: Write the failing test**

Create `tests/adapters/test_taxonomy_dataset.py`:

```python
import pytest

from lattice.adapters.dataset.taxonomy import TaxonomyDataset
from tests.contracts.dataset_contract import DatasetContract

ROOT = "tests/fixtures/mini_texeval"


class TestTaxonomyContract(DatasetContract):
    def make_dataset(self):
        return TaxonomyDataset(root=ROOT, gold="toy")


def test_glossary_document_comes_first():
    docs = list(TaxonomyDataset(root=ROOT, gold="toy").documents())
    assert docs[0].id == "toy:glossary"
    assert docs[0].kind == "terminology"
    assert docs[0].text.splitlines() == [
        "oil", "olive oil", "vegetable oil", "sunflower oil", "fat",
    ]
    assert [d.timestamp for d in docs] == [0.0, 1.0, 2.0]


def test_limit_truncates_articles_but_never_the_glossary():
    docs = list(TaxonomyDataset(root=ROOT, gold="toy", limit=1).documents())
    assert [d.id for d in docs] == ["toy:glossary", "toy:olive-oil"]
    assert [d.id for d in TaxonomyDataset(root=ROOT, gold="toy", limit=0).documents()] == [
        "toy:glossary"
    ]


def test_ground_truth_shape():
    truth = TaxonomyDataset(root=ROOT, gold="toy").ground_truth()
    assert truth["terms"] == ["oil", "olive oil", "vegetable oil", "sunflower oil", "fat"]
    assert ["olive oil", "oil"] in truth["is_a_edges"]
    assert len(truth["is_a_edges"]) == 5


def test_missing_data_names_the_fetch_script():
    with pytest.raises(FileNotFoundError, match="fetch_texeval"):
        list(TaxonomyDataset(root=ROOT, gold="absent").documents())
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_taxonomy_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.dataset.taxonomy'`

- [ ] **Step 4: Write the implementation**

Create `src/lattice/adapters/dataset/taxonomy.py`:

```python
import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "taxonomy")
class TaxonomyDataset(Dataset):
    """TExEval-2 taxonomy benchmark reader (M4 spec §4.4), emitted by
    scripts/fetch_texeval.py: the glossary document first (every gold term
    one-per-line, so the whole term list resolves to concepts in one
    document), then one Wikipedia-summary document per term. `limit`
    truncates the per-term article documents, never the glossary. Ground
    truth is the gold edge list plus the term universe."""

    def __init__(self, root: str, gold: str, limit: int | None = None):
        self._dir = Path(root) / gold
        self._limit = limit

    def _lines(self, name: str) -> list[str]:
        path = self._dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run --no-sync python "
                "scripts/fetch_texeval.py` first"
            )
        return path.read_text().splitlines()

    def documents(self) -> Iterator[Document]:
        for i, line in enumerate(self._lines("documents.jsonl")):
            if self._limit is not None and i > self._limit:
                return
            record = json.loads(line)
            yield Document(
                id=record["id"], kind=record["kind"], text=record["text"],
                timestamp=float(i),
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "is_a_edges": [
                json.loads(line) for line in self._lines("gold_edges.jsonl")
            ],
            "terms": [line for line in self._lines("terms.txt") if line],
        }
```

In `src/lattice/adapters/__init__.py`, extend the dataset import line to:

```python
from lattice.adapters.dataset import (  # noqa: F401
    inspec,
    mention_clusters,
    taxonomy,
    toy,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_taxonomy_dataset.py -q`
Expected: PASS (8 tests: 4 contract + 4 unit)

- [ ] **Step 6: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/dataset/taxonomy.py src/lattice/adapters/__init__.py tests/adapters/test_taxonomy_dataset.py tests/fixtures/mini_texeval/toy
git commit -m "feat: add taxonomy benchmark dataset adapter and toy fixture

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `edge-f1` Metric

**Files:**
- Create: `src/lattice/adapters/metric/edge_f1.py`
- Modify: `src/lattice/adapters/__init__.py` (extend import)
- Test: `tests/adapters/test_edge_f1_metric.py`

**Interfaces:**
- Consumes: snapshot-level `Metric` port (`evaluate(snapshot, ground_truth) -> dict[str, float]`), `GraphSnapshot`, `MetricContract`, `make_concept`, `Relation`.
- Produces: registered adapter `("edge-f1", Metric)` returning keys `precision`, `recall`, `f1`, `predicted_edges`, `gold_edges`. Consumes Task 5's ground-truth shape. Task 8's configs reference `name = "edge-f1"` under `[[base.metrics]]` (snapshot metrics, not `document_metrics`).

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_edge_f1_metric.py`:

```python
import pytest

from lattice.adapters.metric.edge_f1 import EdgeF1
from lattice.core.types import GraphSnapshot, Relation
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


class TestEdgeF1Contract(MetricContract):
    def make_metric(self):
        return EdgeF1()

    def make_ground_truth(self):
        return {"is_a_edges": [["a", "b"]], "terms": ["a", "b"]}


def _snapshot(edges: list[tuple[str, str]], extra_types: bool = False):
    labels = sorted({label for edge in edges for label in edge})
    concepts = tuple(make_concept(id=f"c:{label}", label=label) for label in labels)
    relations = tuple(
        Relation(type="IS_A", source_id=f"c:{a}", target_id=f"c:{b}",
                 confidence=1.0, provenance="d")
        for a, b in edges
    )
    if extra_types:
        relations += (
            Relation(type="CO_OCCURS", source_id=concepts[0].id,
                     target_id=concepts[-1].id, confidence=1.0, provenance="d"),
        )
    return GraphSnapshot(concepts=concepts, relations=relations)


GOLD = {"is_a_edges": [["olive oil", "oil"], ["canola", "oil"]], "terms": []}


def test_perfect_prediction():
    snapshot = _snapshot([("olive oil", "oil"), ("canola", "oil")])
    result = EdgeF1().evaluate(snapshot, GOLD)
    assert result == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "predicted_edges": 2.0, "gold_edges": 2.0,
    }


def test_partial_overlap():
    snapshot = _snapshot([("olive oil", "oil"), ("oil", "olive oil")])
    result = EdgeF1().evaluate(snapshot, GOLD)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_direction_matters():
    snapshot = _snapshot([("oil", "olive oil")])
    assert EdgeF1().evaluate(snapshot, GOLD)["f1"] == 0.0


def test_non_is_a_relations_are_ignored():
    snapshot = _snapshot([("olive oil", "oil")], extra_types=True)
    assert EdgeF1().evaluate(snapshot, GOLD)["predicted_edges"] == 1.0


def test_empty_everything_is_all_zeros_without_crashing():
    result = EdgeF1().evaluate(
        GraphSnapshot(concepts=(), relations=()), {"is_a_edges": [], "terms": []}
    )
    assert result == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "predicted_edges": 0.0, "gold_edges": 0.0,
    }


def test_all_values_are_floats():
    snapshot = _snapshot([("olive oil", "oil")])
    assert all(isinstance(v, float) for v in EdgeF1().evaluate(snapshot, GOLD).values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/adapters/test_edge_f1_metric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.metric.edge_f1'`

- [ ] **Step 3: Write the implementation**

Create `src/lattice/adapters/metric/edge_f1.py`:

```python
from lattice.core.types import GraphSnapshot
from lattice.ports import Metric
from lattice.registry.registry import register


@register(Metric, "edge-f1")
class EdgeF1(Metric):
    """Edge precision/recall/F1 of the snapshot's deduped IS_A edges —
    expressed as (hyponym label, hypernym label) pairs — against
    ground_truth["is_a_edges"] (M4 spec §4.6; TExEval-2 task paper §4.3).
    Direction matters. predicted_edges/gold_edges counts are returned for
    diagnosis (floats, like every metric value)."""

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        label_of = {concept.id: concept.label.lower() for concept in snapshot.concepts}
        predicted = {
            (label_of[relation.source_id], label_of[relation.target_id])
            for relation in snapshot.relations
            if relation.type == "IS_A"
        }
        gold = {
            (str(hypo).lower(), str(hyper).lower())
            for hypo, hyper in ground_truth.get("is_a_edges", [])
        }
        true_positives = len(predicted & gold)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_edges": float(len(predicted)),
            "gold_edges": float(len(gold)),
        }
```

In `src/lattice/adapters/__init__.py`, extend the metric import line to:

```python
from lattice.adapters.metric import edge_f1, label_f1  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/adapters/test_edge_f1_metric.py -q`
Expected: PASS (8 tests: 2 contract + 6 unit)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add src/lattice/adapters/metric/edge_f1.py src/lattice/adapters/__init__.py tests/adapters/test_edge_f1_metric.py
git commit -m "feat: add edge-f1 snapshot metric for IS_A taxonomy evaluation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `scripts/fetch_texeval.py`

**Files:**
- Create: `scripts/fetch_texeval.py`
- Test: `tests/scripts/test_fetch_texeval.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure functions + stdlib network I/O).
- Produces: on-disk layout consumed by Tasks 4–5's adapters: `data/texeval/{key}/terms.txt`, `gold_edges.jsonl`, `documents.jsonl`, `CHECKSUMS`, shared `data/texeval/wiki-cache/`. Gold keys: `env-eurovoc`, `food`, `food-wordnet`, `science`, `science-eurovoc`, `science-wordnet`. Pure functions under test: `normalize_terms`, `parse_taxo`, `usable_extract`, `slugify`, `build_documents`, `recall_ceiling`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_fetch_texeval.py`:

```python
from scripts.fetch_texeval import (
    GOLDS,
    build_documents,
    normalize_terms,
    parse_taxo,
    recall_ceiling,
    slugify,
    usable_extract,
)


def test_normalize_terms_lowercases_and_dedupes_preserving_order():
    lines = ["0\tAdriatic Sea", "1\tolive oil", "2\tADRIATIC SEA", "", "3\tfat"]
    assert normalize_terms(lines) == ["adriatic sea", "olive oil", "fat"]


def test_parse_taxo_lowercases_and_dedupes():
    lines = [
        "0\tChocos\tbreakfast cereal",
        "1\tchocos\tBREAKFAST CEREAL",
        "",
        "2\twaffle crisp\tbreakfast cereal",
    ]
    assert parse_taxo(lines) == [
        ["chocos", "breakfast cereal"],
        ["waffle crisp", "breakfast cereal"],
    ]


def test_usable_extract_requires_standard_type_and_text():
    assert usable_extract({"type": "standard", "extract": " Olive oil is… "}) == (
        "Olive oil is…"
    )
    assert usable_extract({"type": "disambiguation", "extract": "x"}) is None
    assert usable_extract({"type": "not-found"}) is None
    assert usable_extract({"type": "standard", "extract": "  "}) is None
    assert usable_extract({"type": "standard"}) is None


def test_slugify():
    assert slugify("olive oil") == "olive-oil"
    assert slugify("fisherman's soup") == "fishermans-soup"
    assert slugify("pulp/paper technology") == "pulppaper-technology"
    assert slugify("st. louis-style pizza") == "st-louis-style-pizza"


def test_build_documents_glossary_first_then_articles_in_term_order():
    docs = build_documents(
        "toy",
        ["oil", "olive oil", "fat"],
        {"olive oil": "Olive oil is a fat.", "oil": "Oil is a liquid."},
    )
    assert [d["id"] for d in docs] == ["toy:glossary", "toy:oil", "toy:olive-oil"]
    assert docs[0] == {
        "id": "toy:glossary",
        "kind": "terminology",
        "text": "oil\nolive oil\nfat",
    }
    assert docs[2] == {
        "id": "toy:olive-oil",
        "kind": "article",
        "term": "olive oil",
        "text": "Olive oil is a fat.",
    }


def test_build_documents_slug_collisions_get_numeric_suffixes():
    docs = build_documents(
        "toy",
        ["a b", "a-b"],
        {"a b": "one", "a-b": "two"},
    )
    assert [d["id"] for d in docs] == ["toy:glossary", "toy:a-b", "toy:a-b-2"]


def test_recall_ceiling_counts_edges_with_both_endpoints_in_terms():
    terms = ["a", "b", "c"]
    edges = [["a", "b"], ["a", "z"], ["z", "b"], ["c", "a"]]
    assert recall_ceiling(terms, edges) == (2, 4)


def test_gold_keys_are_the_six_configured_english_golds():
    assert list(GOLDS) == [
        "env-eurovoc",
        "food",
        "food-wordnet",
        "science",
        "science-eurovoc",
        "science-wordnet",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/scripts/test_fetch_texeval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_texeval'`

- [ ] **Step 3: Write the implementation**

Create `scripts/fetch_texeval.py`:

```python
"""Fetch TExEval-2 (SemEval-2016 Task 13) English golds and build lattice's
taxonomy benchmark corpus (M4 spec §5). Stdlib only:
    uv run --no-sync python scripts/fetch_texeval.py
macOS SSL quirk: first run
    export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())")

Per gold: terms.txt (lowercased, deduped — the concept universe and the
gazetteer dictionary), gold_edges.jsonl (one [hyponym, hypernym] pair per
line, lowercased, deduped; endpoints outside the term list are KEPT because
the official gold is the full .taxo file — the printed recall ceiling says
how many edges are reachable), and documents.jsonl (glossary document first,
then one document per term with a usable Wikipedia summary). Wikipedia
responses are cached in wiki-cache/ keyed by sha256(term), so reruns are
cheap and resumable; 404s and non-standard pages (disambiguation) are cached
too and yield no article document — the term still becomes a concept via the
glossary."""

import argparse
import hashlib
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ARCHIVE_URL = (
    "http://alt.qcri.org/semeval2016/task13/data/uploads/"
    "texeval-2_testdata_1.2.tar.gz"
)
ARCHIVE_PREFIX = "TExEval-2_testdata_1.2"
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "lattice-m4-benchmark-fetch/1.0 (research use)"
GOLDS = {
    "env-eurovoc": "environment_eurovoc_en",
    "food": "food_en",
    "food-wordnet": "food_wordnet_en",
    "science": "science_en",
    "science-eurovoc": "science_eurovoc_en",
    "science-wordnet": "science_wordnet_en",
}


def normalize_terms(lines: list[str]) -> list[str]:
    """`id⇥term` lines -> lowercased terms, deduped, first occurrence wins
    (food_en.terms has 1555 lines but 1549 unique lowercased terms)."""
    seen: set[str] = set()
    terms: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        _, term = line.split("\t", 1)
        term = term.strip().lower()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def parse_taxo(lines: list[str]) -> list[list[str]]:
    """`id⇥term⇥hypernym` lines -> [hypo, hyper] pairs, lowercased, deduped
    (food_wordnet has 43 duplicate edges, science_wordnet 11)."""
    seen: set[tuple[str, str]] = set()
    edges: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        _, hypo, hyper = line.split("\t", 2)
        pair = (hypo.strip().lower(), hyper.strip().lower())
        if pair not in seen:
            seen.add(pair)
            edges.append([pair[0], pair[1]])
    return edges


def usable_extract(response: dict) -> str | None:
    """The summary text, or None for disambiguation/missing/empty pages."""
    if response.get("type") != "standard":
        return None
    extract = (response.get("extract") or "").strip()
    return extract or None


def slugify(term: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", term.lower().replace(" ", "-"))


def build_documents(
    key: str, terms: list[str], extracts: dict[str, str]
) -> list[dict]:
    """Glossary record first (all terms, one per line), then one article
    record per term with a usable extract, in term order."""
    documents = [
        {"id": f"{key}:glossary", "kind": "terminology", "text": "\n".join(terms)}
    ]
    seen_slugs: dict[str, int] = {}
    for term in terms:
        extract = extracts.get(term)
        if not extract:
            continue
        slug = slugify(term) or "term"
        count = seen_slugs.get(slug, 0) + 1
        seen_slugs[slug] = count
        doc_id = f"{key}:{slug}" if count == 1 else f"{key}:{slug}-{count}"
        documents.append(
            {"id": doc_id, "kind": "article", "term": term, "text": extract}
        )
    return documents


def recall_ceiling(terms: list[str], edges: list[list[str]]) -> tuple[int, int]:
    universe = set(terms)
    reachable = sum(
        1 for hypo, hyper in edges if hypo in universe and hyper in universe
    )
    return reachable, len(edges)


def _fetch_summary(term: str, cache_dir: Path) -> dict:
    cache = cache_dir / f"{hashlib.sha256(term.encode()).hexdigest()}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = WIKI_URL + urllib.parse.quote(term.replace(" ", "_"), safe="")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        payload = {"type": "not-found"}
    cache.write_text(json.dumps(payload))
    time.sleep(0.05)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    out_root = Path(args.root) / "texeval"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "wiki-cache"
    cache_dir.mkdir(exist_ok=True)

    archive = out_root / "texeval-2_testdata_1.2.tar.gz"
    if not archive.exists():
        print(f"downloading {ARCHIVE_URL} …")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)

    with tarfile.open(archive) as tar:

        def read(member: str) -> list[str]:
            return tar.extractfile(member).read().decode("utf-8").splitlines()

        for key, stem in GOLDS.items():
            terms = normalize_terms(
                read(f"{ARCHIVE_PREFIX}/gs_terms/EN/{stem}.terms")
            )
            edges = parse_taxo(read(f"{ARCHIVE_PREFIX}/gs_taxo/EN/{stem}.taxo"))
            extracts: dict[str, str] = {}
            for term in terms:
                extract = usable_extract(_fetch_summary(term, cache_dir))
                if extract:
                    extracts[term] = extract
            documents = build_documents(key, terms, extracts)

            gold_dir = out_root / key
            gold_dir.mkdir(exist_ok=True)
            (gold_dir / "terms.txt").write_text("\n".join(terms) + "\n")
            with (gold_dir / "gold_edges.jsonl").open("w") as f:
                for edge in edges:
                    f.write(json.dumps(edge) + "\n")
            with (gold_dir / "documents.jsonl").open("w") as f:
                for doc in documents:
                    f.write(json.dumps(doc, sort_keys=True) + "\n")
            checksums = []
            for name in ("terms.txt", "gold_edges.jsonl", "documents.jsonl"):
                digest = hashlib.sha256((gold_dir / name).read_bytes()).hexdigest()
                checksums.append(f"{digest}  {name}")
            (gold_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")

            reachable, total = recall_ceiling(terms, edges)
            print(
                f"{key}: {len(terms)} terms, {total} gold edges "
                f"(recall ceiling {reachable}/{total} = {reachable / total:.3f}), "
                f"{len(documents) - 1} summary documents"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/scripts/test_fetch_texeval.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add scripts/fetch_texeval.py tests/scripts/test_fetch_texeval.py
git commit -m "feat: add TExEval-2 fetch/convert script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Sweep configs + M4 e2e tests

**Files:**
- Create: `configs/m4-env-eurovoc-sweep.toml`, `configs/m4-food-sweep.toml`, `configs/m4-food-wordnet-sweep.toml`, `configs/m4-science-sweep.toml`, `configs/m4-science-eurovoc-sweep.toml`, `configs/m4-science-wordnet-sweep.toml`
- Test: `tests/harness/test_m4_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6 by registered name; the toy fixture from Task 5; `ExperimentConfig`/`run_experiment`/`SweepConfig`/`expand`/`load_config` exactly as `tests/harness/test_m3_e2e.py` uses them.
- Produces: the six configs Task 9 sweeps.

- [ ] **Step 1: Write the failing test**

Create `tests/harness/test_m4_e2e.py`:

```python
import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ROOT = "tests/fixtures/mini_texeval"
GOLD_KEYS = [
    "env-eurovoc",
    "food",
    "food-wordnet",
    "science",
    "science-eurovoc",
    "science-wordnet",
]
METRIC_KEYS = {"precision", "recall", "f1", "predicted_edges", "gold_edges"}
UNION = {
    "name": "union",
    "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
}


def _config(inducer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gazetteer", "params": {"root": ROOT, "gold": "toy"}},
            "scorer": {"name": "passthrough"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": inducer,
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {"name": "taxonomy", "params": {"root": ROOT, "gold": "toy"}},
            "metrics": [{"name": "edge-f1"}],
        }
    )


@pytest.mark.parametrize(
    "inducer", [{"name": "compound"}, {"name": "hearst"}, UNION]
)
def test_m4_rows_run_clean(inducer):
    report = run_experiment(_config(inducer))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["edge-f1"]) == METRIC_KEYS


def test_union_dominates_members_on_the_fixture():
    """The M4 thesis in miniature: string structure and corpus evidence each
    find edges the other cannot; their union is exactly the toy gold."""
    f1 = {
        name: run_experiment(_config(spec)).metrics["edge-f1"]["f1"]
        for name, spec in [
            ("compound", {"name": "compound"}),
            ("hearst", {"name": "hearst"}),
            ("union", UNION),
        ]
    }
    # 2*1.0*0.6/1.6 is 0.7499999999999999 in floats — approx, not ==
    assert f1["compound"] == pytest.approx(0.75)
    assert f1["hearst"] == pytest.approx(0.75)
    assert f1["union"] == 1.0  # 2*1*1/2 is exact
    assert f1["union"] >= max(f1["compound"], f1["hearst"])


def test_m4_run_is_reproducible():
    assert run_experiment(_config(UNION)) == run_experiment(_config(UNION))


@pytest.mark.parametrize("key", GOLD_KEYS)
def test_m4_sweep_configs_expand_to_three_rows(key):
    sweep = load_config(f"configs/m4-{key}-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 3
    assert [c.relation_inducer.name for c in configs] == [
        "compound",
        "hearst",
        "union",
    ]
    for config in configs:
        assert config.dataset.params["gold"] == key
        assert config.extractor.params["gold"] == key
        assert config.metrics[0].name == "edge-f1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_m4_e2e.py -q`
Expected: FAIL — the sweep-config tests error with missing `configs/m4-*.toml` files. (The fixture-backed tests should already PASS if Tasks 1–6 are correct — if any of those fail, STOP and report; do not patch other tasks' code.)

- [ ] **Step 3: Generate the six sweep configs**

Run this loop from the repo root (then inspect one file):

```bash
for key in env-eurovoc food food-wordnet science science-eurovoc science-wordnet; do
cat > "configs/m4-$key-sweep.toml" <<EOF
# M4 hierarchy sweep (spec §6): compound vs hearst vs union IS_A induction
# on TExEval-2 $key. Glossary-first taxonomy dataset + gazetteer extractor;
# hashing embedder because embeddings are irrelevant to this protocol.

[base.segmenter]
name = "block"

[base.extractor]
name = "gazetteer"
[base.extractor.params]
root = "data/texeval"
gold = "$key"

[base.scorer]
name = "passthrough"

[base.resolver]
name = "exact-label"

[base.relation_inducer]
name = "compound"

[base.graph_integrator]
name = "in-memory"

[base.embedder]
name = "hashing"

[base.concept_store]
name = "in-memory"

[base.run]
on_error = "fail"
seed = 0

[base.dataset]
name = "taxonomy"
[base.dataset.params]
root = "data/texeval"
gold = "$key"

[[base.metrics]]
name = "edge-f1"

[axes]
relation_inducer = [
  { name = "compound" },
  { name = "hearst" },
  { name = "union", params = { members = [
      { name = "hearst" }, { name = "compound" } ] } },
]
EOF
done
cat configs/m4-env-eurovoc-sweep.toml
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/harness/test_m4_e2e.py -q`
Expected: PASS (11 tests)

Then the whole suite: `uv run --no-sync pytest -q`
Expected: all tests pass, no new failures anywhere.

- [ ] **Step 5: Lint and commit**

```bash
uv run --no-sync ruff check .
git add configs/m4-*.toml tests/harness/test_m4_e2e.py
git commit -m "feat: add M4 sweep configs and end-to-end tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Exit criteria — real fetch + six sweeps + adjudication

Operational task (orchestrator/observer — no committed code; datasets and reports are gitignored). Spec §9 is the contract.

- [ ] **Step 1: Fetch the corpus**

```bash
export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())")
uv run --no-sync python scripts/fetch_texeval.py
```

Expected: ~3,900 unique Wikipedia summary fetches on first run (50ms politeness sleep ⇒ allow 10–30 min; reruns are cache-hits and fast). Printed recall ceilings MUST match spec §5.4 exactly: env-eurovoc 260/261 (0.996), food 1355/1587 (0.854), food-wordnet 1533/1533 (1.0), science 464/465 (0.998), science-eurovoc 124/124 (1.0), science-wordnet 382/441 (0.866). Any mismatch = STOP, report, adjudicate before sweeping.

- [ ] **Step 2: Run the six sweeps**

```bash
for key in env-eurovoc food food-wordnet science science-eurovoc science-wordnet; do
  uv run --no-sync python -m lattice.harness --sweep "configs/m4-$key-sweep.toml" "reports/m4-$key"
done
```

Expected: 18 rows total, `errors` column 0 everywhere.

- [ ] **Step 3: Adjudicate against spec §9**

1. All 18 rows zero errors.
2. Union F1 ≥ each member's F1 on ≥ 4 of 6 golds.
3. Best lattice row per gold lands within/above the published band (spec §9.3). Read recall against the §5.4 ceilings.
4. Record all six sweep tables + adjudication in the ledger.

- [ ] **Step 4: Full verification**

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
git status
```

Expected: full suite green, lint clean, working tree clean (data/ and reports/ are gitignored and must stay uncommitted).

---

## Execution Amendments

- Task 2, `test_explicit_patterns_select_a_subset`: fixture text reordered
  (defect found during execution, 2026-07-12). The plan's `_edges` helper
  anchors surfaces with naive `text.index`, which found "fat" inside "fats"
  at position 0 — overlapping anchors the (correct) overlap guard skips, so
  the intended copula anchor never formed. The pre-commit machine
  verification missed this because it located anchors with the gazetteer
  (whole-word matching) instead of the test helper's `text.index` — when
  simulating a plan's tests, simulate the tests' own anchoring, not an
  equivalent-looking one. Implementation unchanged (plan-verbatim).

## Self-Review Notes (already applied)

- Spec coverage: §4.1→T1, §4.2→T2, §4.3→T3, §4.4→T5, §4.5→T4, §4.6→T6, §5→T7, §6→T8, §7 (FileNotFoundError hints: T4/T5; RegistryError fail-fast: T3; on_error=fail: configs), §8→each task + T8, §9→T9.
- The toy fixture's expected scores are machine-verified (the plan's hearst/gazetteer/compound logic was executed against every unit-test case and the fixture before commit): compound predicts {olive oil→oil, vegetable oil→oil, sunflower oil→oil} (P 1.0, R 0.6, F ≈0.75); hearst predicts {olive oil→fat, sunflower oil→fat, vegetable oil→oil} (P 1.0, R 0.6, F ≈0.75); union predicts all five gold edges exactly (F 1.0 exact). The F≈0.75 values are 0.7499999999999999 in floats, hence pytest.approx in the e2e test.
- Type consistency: `edge-f1` consumes exactly the `{"is_a_edges": list[list[str]], "terms": list[str]}` shape `taxonomy.ground_truth()` returns; `union` member dicts match the TOML inline-table shape; every registered name in configs appears in a task.
- Spec §7 "KeyError" ≙ `RegistryError` (noted in T3) — intent identical.
