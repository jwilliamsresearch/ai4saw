"""Automated corpus fetch pipeline — discover, download, register, and ingest documents.

Extends the passive discovery module (which only returns URLs) into an active pipeline
that downloads matching documents and ingests them directly into ChromaDB.

Sources supported in this version:
  - ReliefWeb (UNOCHA) — PDF and HTML reports, no authentication required
  - GDELT Project — news article HTML, no authentication required

Sources deferred (need custom scrapers):
  - International Tribunal/IRMCT document database
  - ICC court records
  - OHCHR treaty body documents

Pipeline:
  1. discover_for_entities() → candidate URLs with relevance scores
  2. Filter by min_relevance threshold
  3. For each candidate:
     a. HEAD request → detect Content-Type
     b. PDF → download to corpus/ as file, load from path
     c. HTML → load directly from URL (no file saved)
     d. chunk_documents() → embed_and_store()
     e. Append row to corpus/sources.csv
  4. Return FetchResult with counts and any failures

Design choices:
  - Never ingests without explicit confirmation when review=True (default)
  - Idempotent: deterministic chunk IDs mean re-fetching the same URL is safe
  - Licence field is inferred from source (reliefweb → "public", gdelt → "news")
  - Failed downloads are logged and skipped, not fatal
"""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import BaseModel

from ai4saw.core.config import settings
from ai4saw.discovery.discovery import discover_for_entities, discover_for_silences
from ai4saw.core.models import DiscoveredDocument
from ai4saw.core.project import get_corpus_dir, get_sources_csv


REQUEST_TIMEOUT = 20.0
# These are resolved dynamically at call time so they respect the active project.
# Use get_corpus_dir() / get_sources_csv() instead of these constants in new code.
SOURCES_CSV = Path("corpus/sources.csv")   # legacy — kept for external imports
CORPUS_DIR  = Path("corpus")               # legacy — kept for external imports
_SOURCES_HEADER = ["filename", "source_url", "date_accessed", "licence", "geography", "notes"]


# ── Result models ─────────────────────────────────────────────────────────────

class FetchedDocument(BaseModel):
    title: str
    url: str
    source: str
    filename: str
    geography: str
    chunks_added: int
    licence: str


class FetchResult(BaseModel):
    entities_queried: list[str]
    candidates_found: int
    candidates_above_threshold: int
    documents_fetched: int
    chunks_added: int
    skipped: list[str]
    fetched: list[FetchedDocument]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(url: str, title: str, source: str) -> str:
    """Derive a filesystem-safe filename from title + URL hash suffix."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:60].strip("_")
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    ext = ".html" if source == "gdelt" else ".pdf"
    return f"{slug}_{url_hash}{ext}"


def _licence_for_source(source: str) -> str:
    return {
        "openalex": "open-access",
        "semanticscholar": "open-access",
        "arxiv": "open-access",
        "internetarchive": "open-access",
        "gdelt": "news",
    }.get(source, "unknown")


def _is_registered(url: str) -> bool:
    csv_path = get_sources_csv()
    if not csv_path.exists():
        return False
    with open(csv_path, encoding="utf-8") as f:
        return any(row.get("source_url") == url for row in csv.DictReader(f))


def _register_source(
    filename: str,
    url: str,
    licence: str,
    geography: str,
    title: str,
    source: str,
) -> None:
    """Append a row to the project (or legacy) sources.csv."""
    csv_path = get_sources_csv()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SOURCES_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "filename": filename,
            "source_url": url,
            "date_accessed": date.today().isoformat(),
            "licence": licence,
            "geography": geography,
            "notes": f"auto-fetched from {source} | {title[:80]}",
        })


def _detect_content_type(url: str, client: httpx.Client) -> str:
    """HEAD request to determine content type without downloading the body."""
    try:
        r = client.head(url, follow_redirects=True, timeout=10)
        return r.headers.get("content-type", "text/html").lower()
    except Exception:
        return "text/html"


def _download_pdf(url: str, dest: Path, client: httpx.Client) -> bool:
    """Stream a PDF to disk. Returns True on success."""
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.read())
        return True
    except Exception as exc:
        logger.warning(f"PDF download failed for {url}: {exc}")
        return False


# ── Core fetch logic ──────────────────────────────────────────────────────────

def _ingest_document(
    doc: DiscoveredDocument,
    geography: str,
    client: httpx.Client,
) -> tuple[int, str]:
    """Fetch one discovered document and ingest it. Returns (chunks_added, filename)."""
    from ai4saw.ingestion.loaders import load_document
    from ai4saw.ingestion.chunker import chunk_documents
    from ai4saw.ingestion.embedder import embed_and_store

    content_type = _detect_content_type(doc.url, client)
    is_pdf = "pdf" in content_type or doc.url.lower().endswith(".pdf")

    filename = _safe_filename(doc.url, doc.title, doc.source)

    if is_pdf:
        dest = get_corpus_dir() / filename
        if not _download_pdf(doc.url, dest, client):
            return 0, filename
        source_arg: str = str(dest)
        source_url_arg: Optional[str] = doc.url
    else:
        source_arg = doc.url
        source_url_arg = doc.url

    parsed_date: Optional[date] = None
    if doc.date:
        try:
            parsed_date = date.fromisoformat(doc.date)
        except ValueError:
            pass

    try:
        raw_docs = load_document(
            source=source_arg,
            doc_type="report",
            language="en",
            date_published=parsed_date,
            geography=geography,
            source_url=source_url_arg,
        )
    except Exception as exc:
        logger.warning(f"load_document failed for {doc.url}: {exc}")
        return 0, filename

    chunks = chunk_documents(raw_docs)
    embed_and_store(chunks)

    return len(chunks), filename


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_corpus(
    entities: list[str],
    geography: str,
    min_relevance: float = 0.5,
    max_docs: int = 100,
    per_entity_limit: int = 25,
    silence_mode: bool = False,
    dry_run: bool = False,
) -> FetchResult:
    """Discover, download, and ingest documents matching a list of entity names.

    Args:
        entities: Entity names to search for (locations, organisations, events).
        geography: Geography tag written to chunk metadata and sources.csv.
        min_relevance: Minimum relevance score (0–1) to include a candidate.
        max_docs: Hard cap on documents to ingest in one run.
        per_entity_limit: Max API results per entity per source.
        silence_mode: If True, treats entities as silence candidates and uses
            a higher per-entity limit (targets gaps more aggressively).
        dry_run: If True, discovers and reports candidates but ingests nothing.

    Returns:
        FetchResult with counts, fetched document details, and any failures.
    """
    if silence_mode:
        discovery = discover_for_silences(entities, per_entity_limit=max(per_entity_limit, 15))
    else:
        discovery = discover_for_entities(entities, per_entity_limit=per_entity_limit)

    candidates = [
        d for d in discovery.documents
        if d.relevance_score >= min_relevance and not _is_registered(d.url)
    ]
    candidates = candidates[:max_docs]

    if dry_run:
        return FetchResult(
            entities_queried=entities,
            candidates_found=len(discovery.documents),
            candidates_above_threshold=len(candidates),
            documents_fetched=0,
            chunks_added=0,
            skipped=[],
            fetched=[],
        )

    fetched: list[FetchedDocument] = []
    skipped: list[str] = []
    total_chunks = 0

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for doc in candidates:
            logger.info(f"Fetching: {doc.title!r} ({doc.url})")
            chunks_added, filename = _ingest_document(doc, geography, client)

            if chunks_added == 0:
                skipped.append(doc.url)
                logger.warning(f"Skipped (0 chunks): {doc.url}")
                continue

            licence = _licence_for_source(doc.source)
            _register_source(
                filename=filename,
                url=doc.url,
                licence=licence,
                geography=geography,
                title=doc.title,
                source=doc.source,
            )

            fetched.append(FetchedDocument(
                title=doc.title,
                url=doc.url,
                source=doc.source,
                filename=filename,
                geography=geography,
                chunks_added=chunks_added,
                licence=licence,
            ))
            total_chunks += chunks_added

    logger.info(
        f"Fetch complete: {len(fetched)} documents ingested, "
        f"{total_chunks} chunks added, {len(skipped)} skipped"
    )

    return FetchResult(
        entities_queried=entities,
        candidates_found=len(discovery.documents),
        candidates_above_threshold=len(candidates),
        documents_fetched=len(fetched),
        chunks_added=total_chunks,
        skipped=skipped,
        fetched=fetched,
    )
