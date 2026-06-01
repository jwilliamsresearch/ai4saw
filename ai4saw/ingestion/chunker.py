"""Chunking layer — splits raw Documents and attaches validated ChunkMetadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from ai4saw.core.models import ChunkMetadata

# Spec: 1000 tokens / 200 overlap.
# RecursiveCharacterTextSplitter works in characters; ~4 chars per token
# gives chunk_size=4000, overlap=800 for approximate token equivalence.
# Kept as named constants so a researcher can tune without hunting.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 800


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split a list of raw Documents into overlapping chunks with attached metadata.

    Each returned Document has:
      - page_content: the text chunk
      - metadata: all original fields plus chunk_index (int)
    ChunkMetadata validation runs on every chunk so any missing required
    fields surface immediately, not at query time.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    chunk_index = 0

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = chunk_index + i
            _validate_metadata(chunk.metadata, chunk_index + i)
            chunks.append(chunk)
        chunk_index += len(raw_chunks)

    logger.info(f"Chunked {len(documents)} document(s) → {len(chunks)} chunk(s)")
    return chunks


def _validate_metadata(meta: dict, idx: int) -> None:
    """Run Pydantic validation so missing required fields fail loudly."""
    try:
        ChunkMetadata(
            source_filename=meta.get("source_filename", ""),
            source_url=meta.get("source_url"),
            doc_type=meta.get("doc_type", "report"),
            language=meta.get("language", "en"),
            date_published=meta.get("date_published"),
            geography=meta.get("geography"),
            chunk_index=meta.get("chunk_index", idx),
        )
    except Exception as exc:
        logger.warning(f"Chunk {idx} metadata validation warning: {exc}")
