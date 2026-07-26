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
