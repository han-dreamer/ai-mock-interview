"""One-click script to load question-bank chunks into ChromaDB.

Usage:
    python -m scripts.init_vector_store          # normal run
    python -m scripts.init_vector_store --reset   # drop existing data first
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunking import build_question_chunks  # noqa: E402
from app.rag.embeddings import get_embedding_service  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUESTION_BANK_DIR = PROJECT_ROOT / "data" / "question_bank"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
EMBEDDING_BATCH_SIZE = 10


def load_questions_from_json(file_path: Path) -> list[dict]:
    """Parse a question bank JSON file into parent-child retrieval chunks."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    category = data.get("category", file_path.stem)
    questions = data.get("questions", [])

    docs: list[dict] = []
    for question in questions:
        chunks = build_question_chunks(
            question,
            category=category,
            source_file=file_path.name,
        )
        for chunk in chunks:
            docs.append({
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
    return docs


async def main(reset: bool = False) -> None:
    store = get_vector_store()
    embedding_service = get_embedding_service()

    json_files = sorted(QUESTION_BANK_DIR.glob("*.json"))
    knowledge_files = sorted(KNOWLEDGE_DIR.glob("*.json")) if KNOWLEDGE_DIR.exists() else []
    all_json = json_files + knowledge_files

    if not all_json:
        logger.error("No JSON files found in %s or %s", QUESTION_BANK_DIR, KNOWLEDGE_DIR)
        return

    all_docs: list[dict] = []
    for file_path in all_json:
        docs = load_questions_from_json(file_path)
        logger.info("Loaded %d retrieval chunks from %s", len(docs), file_path.name)
        all_docs.extend(docs)

    logger.info("Total retrieval chunks to index: %d", len(all_docs))

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict] = []
    all_embeddings: list[list[float]] = []

    for i in range(0, len(all_docs), EMBEDDING_BATCH_SIZE):
        batch = all_docs[i : i + EMBEDDING_BATCH_SIZE]
        texts = [d["text"] for d in batch]
        logger.info(
            "Embedding batch %d/%d (%d chunks)...",
            i // EMBEDDING_BATCH_SIZE + 1,
            (len(all_docs) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE,
            len(texts),
        )
        embeddings = await embedding_service.embed_texts(texts)

        all_ids.extend(d["id"] for d in batch)
        all_texts.extend(texts)
        all_metadatas.extend(d["metadata"] for d in batch)
        all_embeddings.extend(embeddings)

    if reset:
        logger.info("Resetting vector store collection after embeddings are ready...")
        store.delete_collection()
        from app.rag.vector_store import VectorStore

        store = VectorStore()

    store.add_documents(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metadatas,
        embeddings=all_embeddings,
    )

    logger.info(
        "Done! %d retrieval chunks indexed in vector store (total in collection: %d)",
        len(all_docs),
        store.count,
    )


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    asyncio.run(main(reset=reset_flag))
