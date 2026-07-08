"""Download the ML models M2 needs (the only sanctioned model-download path).
    uv run --group ml python scripts/fetch_models.py"""

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"sentence-transformer ready: dim={model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()
