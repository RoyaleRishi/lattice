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
