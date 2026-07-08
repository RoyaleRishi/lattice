"""Fetch benchmark datasets and convert once to lattice's plain JSONL format
(M2 spec §6.4). Requires the ml dependency group:
    uv run --group ml python scripts/fetch_datasets.py inspec
The emitted JSONL is the only format adapters read — stdlib only at runtime."""

import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def record_to_line(record: dict, fallback_id: str = "") -> str:
    return json.dumps(
        {
            "id": str(record.get("id") or fallback_id),
            "text": " ".join(record["document"]),
            "keyphrases": list(record["extractive_keyphrases"])
            + list(record["abstractive_keyphrases"]),
        },
        sort_keys=True,
    )


def fetch_inspec(root: Path) -> None:
    from datasets import load_dataset  # ml group; imported lazily on purpose

    out_dir = root / "inspec"
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    for split in SPLITS:
        dataset = load_dataset("midas/inspec", "extraction", split=split)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for i, record in enumerate(dataset):
                f.write(record_to_line(record, fallback_id=str(i)) + "\n")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {out_path.name}")
        print(f"wrote {out_path} ({digest[:12]}…)")
    (out_dir / "CHECKSUMS").write_text("\n".join(checksums) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["inspec"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    fetch_inspec(Path(args.root))


if __name__ == "__main__":
    main()
