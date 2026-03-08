import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_INPUT = ROOT / "data" / "chunks" / "chunks.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "embedder.log"

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_VECTOR_INDEX_NAME = "policy_chunks_vector_index"
DEFAULT_COLLECTION_NAME = "policy_chunks"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("embedder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def batched(items: List[Dict], size: int) -> List[List[Dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def ensure_vector_index(
    collection, vector_index_name: str, embedding_dimensions: int, logger: logging.Logger
) -> None:
    # Atlas Search vector index for field "embedding".
    index_definition = {
        "name": vector_index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": embedding_dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "domain"},
                {"type": "filter", "path": "category"},
                {"type": "filter", "path": "doc_type"},
            ]
        },
    }

    try:
        collection.database.command(
            {
                "createSearchIndexes": collection.name,
                "indexes": [index_definition],
            }
        )
        logger.info("Created vector index '%s'", vector_index_name)
    except Exception as exc:
        if "already exists" in str(exc).lower() or "index already exists" in str(exc).lower():
            logger.info("Vector index '%s' already exists", vector_index_name)
        else:
            logger.warning("Could not create vector index automatically: %s", exc)


def main() -> None:
    load_dotenv(ROOT / ".env")
    logger = setup_logger()

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB_NAME", "rag_policy_db")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    vector_index_name = os.getenv("MONGO_VECTOR_INDEX_NAME", DEFAULT_VECTOR_INDEX_NAME)
    embed_model = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
    model_device = os.getenv("EMBED_DEVICE", "cpu")
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    if not mongo_uri:
        raise EnvironmentError("MONGO_URI is not set")
    if not CHUNKS_INPUT.exists():
        raise FileNotFoundError(f"Chunks input file not found: {CHUNKS_INPUT}")

    with CHUNKS_INPUT.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Loaded %s chunks", len(chunks))
    logger.info("Loading embedding model: %s (device=%s)", embed_model, model_device)
    embedder = SentenceTransformer(embed_model, device=model_device)
    embedding_dimensions = int(embedder.get_sentence_embedding_dimension())
    logger.info("Embedding dimension resolved to %s", embedding_dimensions)

    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[mongo_db][collection_name]

    ensure_vector_index(collection, vector_index_name, embedding_dimensions, logger)

    upsert_ops: List[UpdateOne] = []
    processed = 0

    for batch in batched(chunks, batch_size):
        inputs = [item["text"] for item in batch]
        vectors = embedder.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = vectors.tolist()

        for chunk, vector in zip(batch, vectors):
            doc = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "domain": chunk.get("domain", ""),
                "category": chunk.get("category", ""),
                "doc_type": chunk.get("doc_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "char_length": chunk.get("char_length", len(chunk["text"])),
                "embedding_model": embed_model,
                "embedding_dimensions": embedding_dimensions,
                "embedding": vector,
            }
            upsert_ops.append(
                UpdateOne({"chunk_id": chunk["chunk_id"]}, {"$set": doc}, upsert=True)
            )
            processed += 1

        if len(upsert_ops) >= 500:
            collection.bulk_write(upsert_ops, ordered=False)
            upsert_ops = []
            logger.info("Embedded and upserted %s chunks", processed)

    if upsert_ops:
        collection.bulk_write(upsert_ops, ordered=False)

    logger.info("Finished embedding pipeline. Total chunks upserted: %s", processed)


if __name__ == "__main__":
    main()
