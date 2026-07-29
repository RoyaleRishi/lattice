"""Download the ML models M2 needs (the only sanctioned model-download path).
    uv run --group ml python scripts/fetch_models.py"""

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    get_dim = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    print(f"sentence-transformer ready: dim={get_dim()}")

    # PromptRank baseline (credibility Track 3): cache t5-base via the fast
    # tokenizer (no sentencepiece dependency; pyproject frozen).
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    AutoTokenizer.from_pretrained("t5-base")
    T5ForConditionalGeneration.from_pretrained("t5-base")
    print("t5-base ready (PromptRank)")


if __name__ == "__main__":
    main()
