import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        load_dotenv(ROOT / ".env")
        model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
        model_device = os.getenv("EMBED_DEVICE", "cpu")
        _MODEL = SentenceTransformer(model_name, device=model_device)
    return _MODEL


def main() -> None:
    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("No input text provided on stdin")

    model = get_model()
    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(json.dumps(vector.tolist()))


if __name__ == "__main__":
    main()
