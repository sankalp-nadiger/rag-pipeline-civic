import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


def write_msg(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    load_dotenv(ROOT / ".env")
    model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
    model_device = os.getenv("EMBED_DEVICE", "cpu")
    model = SentenceTransformer(model_name, device=model_device)

    write_msg({"type": "ready", "model": model_name, "device": model_device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            text = str(req.get("text", "")).strip()
            if not text:
                write_msg({"id": req_id, "error": "empty_text"})
                continue
            vec = model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            write_msg({"id": req_id, "embedding": vec.tolist()})
        except Exception as exc:  # noqa: BLE001
            write_msg({"id": req.get("id") if "req" in locals() else None, "error": str(exc)})


if __name__ == "__main__":
    main()
