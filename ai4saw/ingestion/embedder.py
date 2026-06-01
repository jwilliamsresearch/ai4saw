"""Embedder — wraps get_embedder() and writes chunks to ChromaDB."""

from __future__ import annotations

import hashlib
import json
from datetime import date

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.providers import get_embedder


def _chunk_id(doc: Document) -> str:
    """Deterministic ID from filename + chunk index so re-runs are idempotent."""
    key = f"{doc.metadata.get('source_filename', '')}_{doc.metadata.get('chunk_index', 0)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _serialise_meta(meta: dict) -> dict:
    """ChromaDB requires all metadata values to be str/int/float/bool."""
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, date):
            out[k] = v.isoformat()
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v)
    return out


def get_vector_store() -> Chroma:
    """Return the shared Chroma vector store instance."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embedder(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def embed_and_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and upsert them into ChromaDB.

    Uses deterministic IDs so re-running ingestion is idempotent —
    duplicate chunks are overwritten rather than accumulated.
    """
    if not chunks:
        logger.warning("embed_and_store called with empty chunk list — nothing to do.")
        return get_vector_store()

    ids = [_chunk_id(chunk) for chunk in chunks]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [_serialise_meta(chunk.metadata) for chunk in chunks]

    store = get_vector_store()
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    logger.info(
        f"Upserted {len(chunks)} chunk(s) into ChromaDB "
        f"(collection={settings.chroma_collection!r}, "
        f"dir={settings.chroma_persist_dir})"
    )
    return store
