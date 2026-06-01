"""Document loaders for PDF, HTML, plaintext, and DOCX sources.

Returns a list of LangChain Document objects with pre-populated metadata
so the chunker can attach ChunkMetadata without re-reading source files.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from loguru import logger


def load_document(
    source: str | Path,
    doc_type: str = "report",
    language: str = "en",
    date_published: Optional[date] = None,
    geography: Optional[str] = None,
    source_url: Optional[str] = None,
) -> list[Document]:
    """Load a document from a local path or URL and return raw Document objects.

    Metadata is attached here so every Document carries full provenance
    before it reaches the chunker.
    """
    source = str(source)
    is_url = source.startswith("http://") or source.startswith("https://")

    if is_url:
        docs = _load_url(source)
        filename = urlparse(source).path.split("/")[-1] or source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source document not found: {path}")
        docs = _load_local(path)
        filename = path.name

    base_meta = {
        "source_filename": filename,
        "source_url": source_url or (source if is_url else None),
        "doc_type": doc_type,
        "language": language,
        "date_published": date_published.isoformat() if date_published else None,
        "geography": geography,
    }

    for doc in docs:
        doc.metadata.update(base_meta)

    logger.info(f"Loaded {len(docs)} page(s) from {filename!r}")
    return docs


def _load_local(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    match suffix:
        case ".pdf":
            return PyPDFLoader(str(path)).load()
        case ".html" | ".htm":
            return BSHTMLLoader(str(path)).load()
        case ".docx":
            return Docx2txtLoader(str(path)).load()
        case ".txt" | ".md" | "":
            return TextLoader(str(path), encoding="utf-8").load()
        case _:
            logger.warning(
                f"Unknown extension {suffix!r} for {path.name} — attempting plaintext load."
            )
            return TextLoader(str(path), encoding="utf-8").load()


def _load_url(url: str) -> list[Document]:
    return WebBaseLoader(url).load()


def load_corpus(
    corpus_dir: str | Path,
    doc_type: str = "report",
    language: str = "en",
    geography: Optional[str] = None,
) -> list[Document]:
    """Recursively load all supported documents from a directory."""
    corpus_dir = Path(corpus_dir)
    supported = {".pdf", ".html", ".htm", ".docx", ".txt", ".md"}
    all_docs: list[Document] = []

    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() in supported and path.is_file():
            try:
                docs = load_document(
                    path,
                    doc_type=doc_type,
                    language=language,
                    geography=geography,
                )
                all_docs.extend(docs)
            except Exception as exc:
                logger.error(f"Failed to load {path}: {exc}")

    logger.info(f"Corpus load complete — {len(all_docs)} total pages from {corpus_dir}")
    return all_docs
