"""Active Corpus Discovery — query external APIs to find documents not yet in the corpus.

Six free, no-registration sources:

  OpenAlex (https://openalex.org/):
    Open scholarly index — academic papers, working papers, reports.
    Per-entity queries. Polite pool via CONTACT_EMAIL env var.

  Semantic Scholar (https://semanticscholar.org/):
    AI-focused academic index. Distinct corpus from OpenAlex; good complementary
    coverage of political science, international law, peace studies. Per-entity.

  arXiv (https://arxiv.org/):
    Preprints in political science, peace studies, economics.
    Catches working papers before journal publication. Per-entity, XML API.

  Internet Archive (https://archive.org/):
    Archived NGO reports, historical news, HRW/Amnesty PDFs, government documents.
    The highest-value source for pre-2010 grey literature. Per-entity.

  UN Digital Library (https://digitallibrary.un.org/):
    UN Secretariat documents — SG reports, Security Council resolutions, OHCHR
    submissions. Directly relevant for conflict-region/Sudan corpora. Per-entity.

  GDELT Project (https://gdeltproject.org/):
    Global news in 65+ languages. Strict 1 req/5s rate limit — batched across
    all entities in a single OR query to avoid IP bans.

Discovered documents are scored by relevance and deduplicated against URLs already
in corpus/sources.csv. Output is for researcher review; fetch_agent.py handles
automated download + ingestion.
"""

from __future__ import annotations

import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import (
    DiscoveredDocument,
    DiscoveryResult,
    EntityResolutionResult,
)

OPENALEX_BASE        = "https://api.openalex.org/works"
S2_BASE              = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_BASE           = "https://export.arxiv.org/api/query"
ARCHIVE_BASE         = "https://archive.org/advancedsearch.php"
GDELT_DOC_BASE       = "https://api.gdeltproject.org/api/v2/doc/doc"
# UN Digital Library deferred: /search ignores ?of=json, /api/v1 requires auth.

REQUEST_TIMEOUT      = 20.0
INTER_REQUEST_DELAY  = 1.5   # between per-entity calls; generous APIs (IA, OpenAlex) tolerate this
OPENALEX_DELAY       = 3.0   # OpenAlex polite pool
# S2 and arXiv are batched (one OR query for all entities) — no per-entity delay needed
GDELT_RETRY_WAIT     = 60.0  # GDELT: 1 req/5s; on 429 wait 60s then retry once


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _known_urls(sources_csv: str = "corpus/sources.csv") -> set[str]:
    path = Path(sources_csv)
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("source_url", "") for row in reader if row.get("source_url")}


def _relevance(entity: str, title: str, base: float = 0.4) -> float:
    token_hits = sum(1 for tok in entity.lower().split() if tok in title.lower())
    return round(min(1.0, base + 0.2 * token_hits), 3)


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def _query_openalex(
    entity: str,
    limit: int = 25,
    client: httpx.Client | None = None,
    contact_email: str = "",
) -> list[DiscoveredDocument]:
    """Open scholarly index — title-matched, open-access only."""
    # Fetch 3x the requested limit — many OA records lack oa_url so we filter after.
    # Cap at OpenAlex's max of 200 per page.
    fetch_limit = min(limit * 3, 200)
    params: dict = {
        "filter": f"title.search:{entity},open_access.is_oa:true",
        "sort": "cited_by_count:desc",
        "per-page": fetch_limit,
        "select": "id,title,doi,open_access,publication_date,abstract_inverted_index",
    }
    if contact_email:
        params["mailto"] = contact_email

    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)

    try:
        r = client.get(OPENALEX_BASE, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning(f"OpenAlex failed for '{entity}': {exc}")
        return []
    finally:
        if close:
            client.close()

    raw = data.get("results", [])
    logger.debug(f"OpenAlex: {len(raw)} raw results for '{entity}' (sent per-page={fetch_limit}, want={limit})")
    docs = []
    for work in raw:
        title = work.get("title") or "Untitled"
        oa = work.get("open_access") or {}
        url = oa.get("oa_url") or work.get("doi") or ""
        if not url:
            continue
        date_str = (work.get("publication_date") or "")[:10]

        snippet: Optional[str] = None
        inv = work.get("abstract_inverted_index")
        if inv:
            pos_word = sorted((pos, w) for w, positions in inv.items() for pos in positions)
            snippet = " ".join(w for _, w in pos_word[:40])

        docs.append(DiscoveredDocument(
            title=title, url=url, source="openalex",
            date=date_str or None,
            relevance_score=_relevance(entity, title),
            trigger_entity=entity, snippet=snippet,
        ))

    logger.debug(f"OpenAlex: {len(docs)} usable results for '{entity}' (from {len(raw)} raw)")
    return docs[:limit]


# ── Semantic Scholar (batched) ────────────────────────────────────────────────

def _query_semanticscholar_batch(
    entities: list[str],
    limit: int = 100,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Single OR query for all entities — avoids S2's per-IP rate limit."""
    if not entities:
        return []

    combined = " | ".join(entities)  # S2 uses | for OR
    params = {
        "query": combined,
        "fields": "title,year,openAccessPdf,externalIds",
        "limit": min(limit, 100),
    }

    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)

    try:
        r = client.get(S2_BASE, params=params)
        if not r.is_success:
            logger.warning(
                f"Semantic Scholar batch failed: HTTP {r.status_code} — {r.text[:200]}"
            )
            return []
        data = r.json()
    except Exception as exc:
        logger.warning(f"Semantic Scholar batch failed: {exc}")
        return []
    finally:
        if close:
            client.close()

    docs = []
    for paper in data.get("data", []):
        title = paper.get("title") or "Untitled"
        oa_pdf = paper.get("openAccessPdf") or {}
        url = oa_pdf.get("url") or ""
        if not url:
            continue
        year = paper.get("year")
        trigger = next((e for e in entities if e.lower() in title.lower()), entities[0])
        docs.append(DiscoveredDocument(
            title=title, url=url, source="semanticscholar",
            date=f"{year}-01-01" if year else None,
            relevance_score=_relevance(trigger, title),
            trigger_entity=trigger, snippet=None,
        ))

    logger.debug(f"Semantic Scholar batch: {len(docs)} results for {len(entities)} entities")
    return docs


# ── arXiv (batched) ───────────────────────────────────────────────────────────

_ARXIV_NS = "http://www.w3.org/2005/Atom"

def _query_arxiv_batch(
    entities: list[str],
    limit: int = 100,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Single OR query across all entities — avoids arXiv's per-IP rate limit."""
    if not entities:
        return []

    terms = " OR ".join(f"(ti:{e} OR abs:{e})" for e in entities)
    params = {
        "search_query": terms,
        "max_results": min(limit, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)

    try:
        r = client.get(ARXIV_BASE, params=params)
        if not r.is_success:
            logger.warning(
                f"arXiv batch failed: HTTP {r.status_code} — {r.text[:200]}"
            )
            return []
        root = ET.fromstring(r.text)
    except Exception as exc:
        logger.warning(f"arXiv batch failed: {exc}")
        return []
    finally:
        if close:
            client.close()

    docs = []
    for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
        title_el = entry.find(f"{{{_ARXIV_NS}}}title")
        title = (title_el.text or "Untitled").strip() if title_el is not None else "Untitled"

        url = ""
        for link in entry.findall(f"{{{_ARXIV_NS}}}link"):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                break
        if not url:
            id_el = entry.find(f"{{{_ARXIV_NS}}}id")
            url = id_el.text.strip() if id_el is not None else ""
        if not url:
            continue

        published_el = entry.find(f"{{{_ARXIV_NS}}}published")
        date_str = (published_el.text or "")[:10] if published_el is not None else None
        summary_el = entry.find(f"{{{_ARXIV_NS}}}summary")
        snippet = (summary_el.text or "").strip()[:300] or None

        trigger = next((e for e in entities if e.lower() in title.lower()), entities[0])
        docs.append(DiscoveredDocument(
            title=title, url=url, source="arxiv",
            date=date_str,
            relevance_score=_relevance(trigger, title),
            trigger_entity=trigger, snippet=snippet,
        ))

    logger.debug(f"arXiv batch: {len(docs)} results for {len(entities)} entities")
    return docs


# ── Internet Archive ───────────────────────────────────────────────────────────

def _query_internetarchive(
    entity: str,
    limit: int = 25,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Internet Archive — highest-value source for pre-2010 NGO and grey literature."""
    params = {
        "q": f"{entity} AND mediatype:texts",
        "output": "json",
        "rows": limit,
        "fl": "identifier,title,date,description",
        "sort": "downloads desc",
    }

    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)

    try:
        r = client.get(ARCHIVE_BASE, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning(f"Internet Archive failed for '{entity}': {exc}")
        return []
    finally:
        if close:
            client.close()

    raw_ia = data.get("response", {}).get("docs", [])
    logger.debug(f"Internet Archive: {len(raw_ia)} raw results for '{entity}' (requesting {limit})")
    docs = []
    for doc in raw_ia:
        identifier = doc.get("identifier", "")
        title = doc.get("title") or identifier or "Untitled"
        if isinstance(title, list):
            title = title[0]
        url = f"https://archive.org/details/{identifier}" if identifier else ""
        if not url:
            continue

        raw_date = doc.get("date") or ""
        if isinstance(raw_date, list):
            raw_date = raw_date[0]
        date_str = str(raw_date)[:10] or None

        desc = doc.get("description") or ""
        if isinstance(desc, list):
            desc = desc[0]
        snippet = str(desc)[:300] or None

        docs.append(DiscoveredDocument(
            title=title, url=url, source="internetarchive",
            date=date_str,
            relevance_score=_relevance(entity, title),
            trigger_entity=entity, snippet=snippet,
        ))

    logger.debug(f"Internet Archive: {len(docs)} results for '{entity}'")
    return docs


# ── GDELT ─────────────────────────────────────────────────────────────────────

def _query_gdelt_batch(
    entities: list[str],
    limit: int = 250,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Single batched OR query for all entities — avoids GDELT's 1 req/5s IP ban."""
    if not entities:
        return []

    terms = [f'"{e}"' if " " in e else e for e in entities]
    combined_query = " OR ".join(terms)

    params = {
        "query": combined_query,
        "mode": "ArtList",
        "maxrecords": min(limit, 250),
        "format": "json",
        "sort": "DateDesc",
        "timespan": "FULL",
    }

    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)

    data: dict = {}
    try:
        r = client.get(GDELT_DOC_BASE, params=params)
        if r.status_code == 429:
            logger.warning(
                f"GDELT rate limited (429) — body: {r.text[:300]!r} — "
                f"waiting {GDELT_RETRY_WAIT:.0f}s then retrying once"
            )
            time.sleep(GDELT_RETRY_WAIT)
            r = client.get(GDELT_DOC_BASE, params=params)
        if not r.is_success:
            logger.warning(
                f"GDELT batch failed: HTTP {r.status_code} — {r.text[:300]!r}"
            )
        else:
            data = r.json()
    except Exception as exc:
        logger.warning(f"GDELT batch failed: {exc}")
    finally:
        if close:
            client.close()

    docs = []
    for art in data.get("articles", []):
        title = art.get("title", "Untitled")
        url = art.get("url", "")
        if not url:
            continue
        date_str = art.get("seendate", "")[:8]
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        title_lower = title.lower()
        trigger = next((e for e in entities if e.lower() in title_lower), entities[0])

        docs.append(DiscoveredDocument(
            title=title, url=url, source="gdelt",
            date=date_str or None,
            relevance_score=_relevance(trigger, title, base=0.3),
            trigger_entity=trigger, snippet=None,
        ))

    logger.debug(f"GDELT batch: {len(docs)} results for {len(entities)} entities")
    return docs


# ── Deduplication and ranking ─────────────────────────────────────────────────

def _dedup_and_rank(
    docs: list[DiscoveredDocument],
    known_urls: set[str],
) -> list[DiscoveredDocument]:
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
    per_entity_limit: int = 25,
    delay: float = INTER_REQUEST_DELAY,
    sources_csv: str = "corpus/sources.csv",
) -> DiscoveryResult:
    """Run corpus discovery across all six sources for a list of entity strings.

    Per-entity sources (OpenAlex, Semantic Scholar, arXiv, Internet Archive, UNDL)
    are queried once per entity. GDELT is queried once for all entities combined
    to respect its strict 1-request-per-5-seconds rate limit.

    Args:
        entities: Entity names to search for.
        per_entity_limit: Max results per entity per per-entity source.
        delay: Seconds between per-entity API calls.
        sources_csv: Corpus provenance register for deduplication.
    """
    known = _known_urls(sources_csv)
    all_docs: list[DiscoveredDocument] = []
    query_count = 0
    contact_email = getattr(settings, "contact_email", "")

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        # Per-entity sources (generous rate limits)
        for entity in entities:
            logger.info(f"Discovering: {entity!r}")

            all_docs.extend(_query_openalex(entity, per_entity_limit, client, contact_email))
            query_count += 1
            time.sleep(OPENALEX_DELAY)

            all_docs.extend(_query_internetarchive(entity, per_entity_limit, client))
            query_count += 1
            time.sleep(delay)

        # Batched sources (one OR query for all entities — avoids per-IP rate bans)
        logger.info(f"Semantic Scholar: batch query for {len(entities)} entities")
        all_docs.extend(_query_semanticscholar_batch(entities, limit=100, client=client))
        query_count += 1
        time.sleep(delay)

        logger.info(f"arXiv: batch query for {len(entities)} entities")
        all_docs.extend(_query_arxiv_batch(entities, limit=100, client=client))
        query_count += 1
        time.sleep(delay)

        time.sleep(6)  # GDELT enforces 1 req/5s strictly; guarantee gap from previous calls
        logger.info(f"GDELT: batch query for {len(entities)} entities")
        all_docs.extend(_query_gdelt_batch(entities, limit=250, client=client))
        query_count += 1

    deduped = _dedup_and_rank(all_docs, known)
    logger.info(
        f"Discovery complete: {query_count} API queries, "
        f"{len(all_docs)} raw results, {len(deduped)} new documents"
    )
    return DiscoveryResult(
        trigger_entities=entities,
        documents=deduped,
        query_count=query_count,
        new_documents=len(deduped),
    )


def discover_from_registry(
    registry: EntityResolutionResult,
    top_n: int = 10,
    entity_labels: list[str] | None = None,
    per_entity_limit: int = 25,
) -> DiscoveryResult:
    """Discover for the top-N entities by frequency from the entity registry."""
    entities = registry.entities
    if entity_labels:
        entities = [e for e in entities if e.label in entity_labels]
    top_entities = [e.canonical_text for e in entities[:top_n]]
    return discover_for_entities(top_entities, per_entity_limit=per_entity_limit)


def discover_for_silences(
    silence_locations: list[str],
    per_entity_limit: int = 25,
) -> DiscoveryResult:
    """Targeted discovery for locations identified as informational silences."""
    return discover_for_entities(silence_locations, per_entity_limit=per_entity_limit)
