"""Fetch ConEL-2 (Joko & Hasibi 2022) and convert to lattice's unified
mention-cluster JSONL (M3 spec §4.4/§5). Stdlib only:
    uv run --no-sync python scripts/fetch_conel2.py
Splits: Train/Val/Test JSON -> train/validation/test.jsonl. Personal-entity
annotations are excluded (speaker-relative references, not shared concepts).
Cluster id = the gold Wikipedia entity. One known corpus defect (a span
including a trailing period) is fixed by the prefix-trim rule; anything else
raises.

Utterances are rstripped and whitespace-only turns dropped so the emitted
text survives the block segmenter unchanged (37 of 290 raw conversations end
with trailing whitespace, which BlockSegmenter's strip() would remove and
break the gold-mentions single-unit invariant; no gold span extends into a
stripped tail — verified corpus-wide)."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/informagi/"
    "conversational-entity-linking-2022/main/dataset/"
    "Conversational_Entity_Linking_Annotations"
)
SPLITS = {"Train": "train", "Val": "validation", "Test": "test"}


def convert_dialogue(dialogue: dict) -> dict:
    texts: list[str] = []
    mentions: list[dict] = []
    offset = 0
    for turn in dialogue["turns"]:
        utterance = turn["utterance"].rstrip()
        if not utterance:
            continue  # a whitespace-only turn would emit a blank line ("\n\n")
        for ann in turn.get("el_annotations", []):
            start, end = ann["span"]
            surface = ann["mention"]
            if utterance[start:end] != surface:
                if utterance[start:start + len(surface)] == surface:
                    end = start + len(surface)
                else:
                    raise ValueError(
                        f"unfixable span in dialogue {dialogue['dialogue_id']}: "
                        f"{ann!r}"
                    )
            mentions.append(
                {
                    "start": offset + start,
                    "end": offset + end,
                    "surface": surface,
                    "cluster": ann["entity"],
                }
            )
        texts.append(utterance)
        offset += len(utterance) + 1  # the joining "\n"
    return {
        "id": f"conel-{dialogue['dialogue_id']}",
        "kind": "transcript",
        "text": "\n".join(texts),
        "mentions": mentions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    out_dir = Path(args.root) / "conel2"
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    for raw_split, split in SPLITS.items():
        url = f"{BASE}/ConEL22_EL_{raw_split}.json"
        with urllib.request.urlopen(url) as response:
            dialogues = json.load(response)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for dialogue in dialogues:
                f.write(json.dumps(convert_dialogue(dialogue), sort_keys=True) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        print(f"wrote {out_path} ({len(dialogues)} conversations, {digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
