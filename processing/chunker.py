import json
import logging
import re
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT = ROOT / "data" / "raw" / "raw_documents.json"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_OUTPUT = CHUNKS_DIR / "chunks.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "chunker.log"

# Approximate token sizing by characters.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chunker")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


def create_chunks(documents: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    chunk_rows: List[Dict] = []
    for doc in documents:
        text = clean_text(doc.get("text", ""))
        if not text:
            continue
        segments = split_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
        for idx, segment in enumerate(segments):
            chunk_rows.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx:04d}",
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "domain": doc.get("domain", ""),
                    "category": doc.get("category", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "chunk_index": idx,
                    "text": segment,
                    "char_length": len(segment),
                }
            )
    return chunk_rows


def main() -> None:
    logger = setup_logger()
    logger.info("Starting chunking...")
    if not RAW_INPUT.exists():
        raise FileNotFoundError(f"Raw input file not found: {RAW_INPUT}")

    with RAW_INPUT.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    chunks = create_chunks(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Loaded %s docs and generated %s chunks", len(documents), len(chunks))
    logger.info("Saved chunks to %s", CHUNKS_OUTPUT)


if __name__ == "__main__":
    main()
