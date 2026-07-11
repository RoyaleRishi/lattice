"""Fetch ECB+ (Cybulska & Vossen 2014) and convert to lattice's unified
mention-cluster JSONL (M3 spec §4.4/§5). Stdlib only:
    uv run --no-sync python scripts/fetch_ecbplus.py
Entity chains only (tags starting HUMAN_PART/NON_HUMAN_PART/LOC/TIME);
validated-sentences filter applied; cluster ids: CROSS_DOC_COREF note,
INTRA_DOC_COREF {doc}:r{r_id}, else singleton {doc}:m{m_id}. Non-contiguous
mentions (3 corpus-wide) and duplicate spans are skipped, keeping the lowest
m_id. Splits by topic: train 1-35, test 36-45."""

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://github.com/cltl/ecbPlus/archive/refs/heads/master.zip"
INNER_ZIP = "ecbPlus-master/ECB+_LREC2014/ECB+.zip"
SENTENCES_CSV = "ecbPlus-master/ECB+_LREC2014/ECBplus_coreference_sentences.csv"
ENTITY_PREFIXES = ("HUMAN_PART", "NON_HUMAN_PART", "LOC", "TIME")
TEST_TOPICS = set(range(36, 46))


def is_entity_tag(tag: str) -> bool:
    return tag.startswith(ENTITY_PREFIXES)


def convert_document(
    doc_name: str, xml_text: str, validated_sentences: set[str]
) -> dict | None:
    root = ET.fromstring(xml_text)
    sentences: dict[str, list[tuple[int, str]]] = {}
    for t in root.iter("token"):
        if t.get("sentence") in validated_sentences:
            sentences.setdefault(t.get("sentence"), []).append(
                (int(t.get("t_id")), t.text or "")
            )
    if not sentences:
        return None
    spans: dict[int, tuple[int, int]] = {}
    lines: list[str] = []
    cursor = 0
    for s in sorted(sentences, key=int):
        col = 0
        pieces: list[str] = []
        for t_id, word in sorted(sentences[s]):
            if pieces:
                col += 1  # joining space
            spans[t_id] = (cursor + col, cursor + col + len(word))
            pieces.append(word)
            col += len(word)
        lines.append(" ".join(pieces))
        cursor += col + 1  # the joining "\n"
    text = "\n".join(lines)

    cluster_of: dict[str, str] = {}
    for rel in root.find("Relations") or []:
        if rel.tag == "CROSS_DOC_COREF":
            cluster_id = rel.get("note")
        elif rel.tag == "INTRA_DOC_COREF":
            cluster_id = f"{doc_name}:r{rel.get('r_id')}"
        else:
            continue
        for source in rel.findall("source"):
            cluster_of[source.get("m_id")] = cluster_id

    candidates: list[tuple[int, int, int, str, str]] = []
    for m in root.find("Markables") or []:
        anchors = sorted(int(a.get("t_id")) for a in m.findall("token_anchor"))
        if not anchors or not is_entity_tag(m.tag):
            continue
        if anchors != list(range(anchors[0], anchors[-1] + 1)):
            continue  # 3 non-contiguous mentions corpus-wide — skipped (spec §5)
        if any(t_id not in spans for t_id in anchors):
            continue  # anchored outside validated sentences
        start = spans[anchors[0]][0]
        end = spans[anchors[-1]][1]
        m_id = m.get("m_id")
        candidates.append((start, end, int(m_id), m_id, text[start:end]))

    mentions: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, end, _, m_id, surface in sorted(candidates):
        if (start, end) in seen_spans:
            continue  # duplicate span: keep lowest m_id (spec §5)
        seen_spans.add((start, end))
        mentions.append(
            {
                "start": start,
                "end": end,
                "surface": surface,
                "cluster": cluster_of.get(m_id, f"{doc_name}:m{m_id}"),
            }
        )
    return {"id": doc_name, "kind": "article", "text": text, "mentions": mentions}


def _doc_sort_key(doc_name: str) -> tuple[int, int, str]:
    topic, rest = doc_name.split("_", 1)
    number = int(re.match(r"\d+", rest).group())
    return int(topic), number, rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    out_dir = Path(args.root) / "ecbplus"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = out_dir / "ecbplus-master.zip"
    if not archive_path.exists():
        print(f"downloading {ARCHIVE_URL} …")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    with zipfile.ZipFile(archive_path) as outer:
        validated: dict[str, set[str]] = {}
        with outer.open(SENTENCES_CSV) as f:
            for row in csv.DictReader(io.TextIOWrapper(f)):
                key = f"{row['Topic']}_{row['File']}"
                validated.setdefault(key, set()).add(row["Sentence Number"])
        inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
        rows: dict[str, list[dict]] = {"train": [], "test": []}
        for name in sorted(inner.namelist()):
            if not name.endswith(".xml") or "__MACOSX" in name:
                continue
            doc_name = Path(name).stem
            if doc_name not in validated:
                continue
            row = convert_document(
                doc_name, inner.read(name).decode("utf-8"), validated[doc_name]
            )
            if row is None:
                continue
            topic = int(doc_name.split("_", 1)[0])
            rows["test" if topic in TEST_TOPICS else "train"].append(row)

    checksums: list[str] = []
    for split, split_rows in rows.items():
        split_rows.sort(key=lambda r: _doc_sort_key(r["id"]))
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        n_mentions = sum(len(r["mentions"]) for r in split_rows)
        print(f"wrote {out_path} ({len(split_rows)} docs, {n_mentions} mentions, {digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
