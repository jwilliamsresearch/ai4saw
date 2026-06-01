"""Active Corpus Discovery — query external APIs to find documents not yet in the corpus.

The silence detector identifies what's missing. This module tries to fill those gaps.

Given a list of high-frequency entities from the corpus (locations, organisations,
groups), it queries two free, no-auth-required APIs:

  ReliefWeb (https://reliefweb.int/):
    UNOCHA's humanitarian information platform. Covers NGO reports, UN situation
    reports, and assessments from conflict and humanitarian crises worldwide.
    Free API, no authentication required.

  GDELT Project (https://www.gdeltproject.org/):
    Global news event database monitoring 65+ languages in real time. The DOC API
    searches full news article text. Free, no authentication required.

Discovered documents are scored by relevance and deduplicated against URLs already
in corpus/sources.csv. The output is a DiscoveryResult that a researcher can review
and selectively ingest — the pipeline never ingests automatically without human review.

Rate limiting: both APIs are generous but not unlimited. The default delay between
calls (1.0 second) keeps requests well within acceptable limits.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import httpx
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import (
    DiscoveredDocument,
    DiscoveryResult,
    EntityResolutionResult,
)

RELIEFWEB_BASE = "https://api.reliefweb.int/v1/reports"
GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_TIMEOUT = 15.0
INTER_REQUEST_DELAY = 1.0


# ── Existing URL deduplication ────────────────────────────────────────────────

def _known_urls(sources_csv: str = "corpus/sources.csv") -> set[str]:
    """Load URLs already registered in corpus/sources.csv to avoid re-discovering them."""
    path = Path(sources_csv)
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("source_url", "") for row in reader if row.get("source_url")}


# ── ReliefWeb ─────────────────────────────────────────────────────────────────

def _query_reliefweb(
    entity: str,
    limit: int = 10,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Query the ReliefWeb API for reports mentioning an entity.

    Searches title + body text. Returns reports sorted by date descending.
    """
    params = {
        "appname": "ai4saw",
        "query[value]": entity,
        "query[operator]": "AND",
        "fields[include][]": ["title", "url", "date.created", "body-html"],
        "sort[]": "date:desc",
        "limit": limit,
    }

    close = False
    if client is None:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
        close = True

    try:
        response = client.get(RELIEFWEB_BASE, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning(f"ReliefWeb query failed for '{entity}': {exc}")
        return []
    finally:
        if close:
            client.close()

    docs = []
    for item in data.get("data", []):
        fields = item.get("fields", {})
        title = fields.get("title", "Untitled")
        url = fields.get("url", "")
        date_obj = fields.get("date", {})
        date_str = date_obj.get("created", "")[:10] if isinstance(date_obj, dict) else ""
        body = fields.get("body-html", "")
        snippet = body[:300].strip() if body else None

        # Simple relevance: count of entity tokens in title (0–1 normalised)
        token_hits = sum(1 for tok in entity.lower().split() if tok in title.lower())
        relevance = min(1.0, 0.4 + 0.2 * token_hits)

        if url:
            docs.append(DiscoveredDocument(
                title=title,
                url=url,
                source="reliefweb",
                date=date_str or None,
                relevance_score=round(relevance, 3),
                trigger_entity=entity,
                snippet=snippet,
            ))

    logger.debug(f"ReliefWeb: {len(docs)} results for '{entity}'")
    return docs


# ── GDELT ─────────────────────────────────────────────────────────────────────

def _query_gdelt(
    entity: str,
    limit: int = 10,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Query the GDELT DOC 2.0 API for news articles mentioning an entity.

    The GDELT API searches article text across 65+ languages. Returns ArtList
    (article list) mode — article metadata without requiring full-text download.
    """
    params = {
        "query": entity,
        "mode": "ArtList",
        "maxrecords": limit,
        "format": "json",
        "sort": "DateDesc",
    }

    close = False
    if client is None:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
        close = True

    try:
        response = client.get(GDELT_DOC_BASE, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning(f"GDELT query failed for '{entity}': {exc}")
        return []
    finally:
        if close:
            client.close()

    docs = []
    for art in data.get("articles", []):
        title = art.get("title", "Untitled")
        url = art.get("url", "")
        date_str = art.get("seendate", "")[:8]
        # Convert GDELT date format YYYYMMDD to ISO YYYY-MM-DD
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        token_hits = sum(1 for tok in entity.lower().split() if tok in title.lower())
        relevance = min(1.0, 0.3 + 0.2 * token_hits)

        if url:
            docs.append(DiscoveredDocument(
                title=title,
                url=url,
                source="gdelt",
                date=date_str or None,
                relevance_score=round(relevance, 3),
                trigger_entity=entity,
                snippet=None,
            ))

    logger.debug(f"GDELT: {len(docs)} results for '{entity}'")
    return docs


# ── Deduplication and ranking ─────────────────────────────────────────────────

def _dedup_and_rank(
    docs: list[DiscoveredDocument],
    known_urls: set[str],
) -> list[DiscoveredDocument]:
    """Remove documents already in the corpus and deduplicate by URL."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        if doc.url in seen or doc.url in known_urls:
            continue
        seen.add(doc.url)
        unique.append(doc)
    unique.sort(key=lambda d: d.relevance_score, reverse=True)
    return unique


# ── Public API ────────────────────────────────────────────────────────────────

def discover_for_entities(
    entities: list[str],
    per_entity_limit: int = 10,
    delay: float = INTER_REQUEST_DELAY,
    sources_csv: str = "corpus/sources.csv",
) -> DiscoveryResult:
    """Run corpus discovery for a list of entity strings.

    Args:
        entities: Entity names to search for (typically top entities by frequency
            from the resolved entity registry, or silence detection candidates).
        per_entity_limit: Maximum results per entity per API source.
        delay: Seconds to wait between API calls. 1.0 is safe for both APIs.
        sources_csv: Path to the corpus provenance register for deduplication.

    Returns:
        DiscoveryResult with discovered documents, deduplicated and ranked.
    """
    known = _known_urls(sources_csv)
    all_docs: list[DiscoveredDocument] = []
    query_count = 0

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for entity in entities:
            logger.info(f"Discovering for entity: {entity!r}")

            rw_docs = _query_reliefweb(entity, limit=per_entity_limit, client=client)
            all_docs.extend(rw_docs)
            query_count += 1
            time.sleep(delay)

            gdelt_docs = _query_gdelt(entity, limit=per_entity_limit, client=client)
            all_docs.extend(gdelt_docs)
            query_count += 1
            time.sleep(delay)

    deduped = _dedup_and_rank(all_docs, known)
    new_count = len(deduped)

    logger.info(
        f"Discovery complete: {query_count} API queries, "
        f"{len(all_docs)} raw results, {new_count} new documents found"
    )

    return DiscoveryResult(
        trigger_entities=entities,
        documents=deduped,
        query_count=query_count,
        new_documents=new_count,
    )


def discover_from_registry(
    registry: EntityResolutionResult,
    top_n: int = 10,
    entity_labels: list[str] | None = None,
    per_entity_limit: int = 10,
) -> DiscoveryResult:
    """Convenience wrapper: discover documents for the top-N entities by frequency.

    Args:
        registry: Resolved entity registry (output of entity_resolution).
        top_n: How many top entities to query. Start small (5–10); each entity
            costs 2 API calls.
        entity_labels: Filter to specific label types, e.g. ["LOCATION", "ORG"].
            None means all labels.
        per_entity_limit: Results per entity per source.

    Returns:
        DiscoveryResult.
    """
    entities = registry.entities
    if entity_labels:
        entities = [e for e in entities if e.label in entity_labels]

    top_entities = [e.canonical_text for e in entities[:top_n]]
    return discover_for_entities(top_entities, per_entity_limit=per_entity_limit)


def discover_for_silences(
    silence_locations: list[str],
    per_entity_limit: int = 15,
) -> DiscoveryResult:
    """Run targeted discovery for locations identified as informational silences.

    Silence candidates (from silence.py) represent high-conflict areas with low
    corpus coverage. Querying these locations directly is the highest-yield
    discovery strategy — it directly addresses the gap the silence detector found.
    """
    return discover_for_entities(
        silence_locations,
        per_entity_limit=per_entity_limit,
        delay=INTER_REQUEST_DELAY,
    )
