"""Autonomous web discovery agent with persistent state.

The agent maintains a state file (output/web_agent_state.json) between runs so it can:

  - Remember every URL it has already visited (never re-fetches)
  - Maintain a priority frontier of URLs still to visit
  - Score domains by how often they produce quality documents
  - Score each DuckDuckGo query template by new-document yield
  - Adapt: high-yield templates run first; low-yield domains are deprioritised

Loop behaviour (--loop mode in CLI):

  Each session:
    1. Drain frontier — visit the highest-priority queued URLs
    2. Every N sessions, run fresh DuckDuckGo / Wikipedia / CrossRef queries
    3. All new URLs go back into the frontier for future sessions
    4. State is saved between sessions → agent is crash-safe

This means the agent is genuinely continuous: it works through backlogged URLs
between discovery sessions, not just "polling on a timer".

Free sources (no API keys required):
  DuckDuckGo  — duckduckgo-search library, multiple query strategies
  Wikipedia   — MediaWiki API, external links from article pages
  CrossRef    — open-access academic papers indexed by DOI
  Link follower — PDF + trusted-domain links extracted from visited HTML pages
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel, Field

from ai4saw.core.models import DiscoveredDocument, DiscoveryResult


# ── Constants ──────────────────────────────────────────────────────────────────

STATE_FILE        = Path("output/web_agent_state.json")
CROSSREF_BASE     = "https://api.crossref.org/works"
WP_SEARCH_API     = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT   = 15.0
DDG_DELAY         = 3.5   # between DDG queries — stay polite
CROSSREF_DELAY    = 1.0
WP_DELAY          = 1.0
FRONTIER_MAX      = 5_000  # cap frontier size to avoid unbounded growth

# DuckDuckGo query templates. {entity} substituted per entity.
# Each template has a key used in query_stats for adaptation.
_DDG_TEMPLATES: list[tuple[str, str]] = [
    ("pdf_report",     '"{entity}" filetype:pdf human rights report'),
    ("hrw_amnesty",    '"{entity}" site:hrw.org OR site:amnesty.org OR site:ohchr.org'),
    ("icty_icc",       '"{entity}" site:icty.org OR site:irmct.org OR site:icc-cpi.int'),
    ("un_reliefweb",   '"{entity}" site:un.org OR site:reliefweb.int'),
    ("tribunal_crime", '"{entity}" war crime genocide slavery tribunal court'),
]

# Domains that reliably host high-quality documents
TRUSTED_DOMAINS = frozenset({
    "hrw.org", "amnesty.org", "ohchr.org",
    "icty.org", "irmct.org", "icc-cpi.int",
    "un.org", "reliefweb.int", "unhcr.org", "ocha.org",
    "crisisgroup.org", "prio.org", "ssrn.com",
    "jstor.org", "cambridge.org", "oxfordjournals.org",
    "archive.org", "acleddata.com", "globalr2p.org",
    "justicehub.org", "peacepalacelibrary.nl",
})

_PDF_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)


# ── State models ───────────────────────────────────────────────────────────────

class FrontierItem(BaseModel):
    url: str
    priority: float = Field(..., ge=0.0, le=1.0)
    trigger_entity: str
    source: str
    depth: int = 0
    added: str = Field(default_factory=lambda: _now())


class DomainStats(BaseModel):
    hits: int = 0      # times domain produced ingested chunks
    attempts: int = 0  # total visit attempts

    @property
    def score(self) -> float:
        if self.attempts == 0:
            return 0.5
        return round(self.hits / self.attempts, 3)


class QueryTemplateStats(BaseModel):
    runs: int = 0
    new_docs: int = 0  # docs above threshold from this template

    @property
    def yield_rate(self) -> float:
        """Average new docs per run. Untested templates return 1.0 (optimistic)."""
        if self.runs == 0:
            return 1.0
        return round(self.new_docs / self.runs, 3)


class WebAgentState(BaseModel):
    """Persistent state for the autonomous web discovery agent."""
    visited_urls: dict[str, dict] = {}           # url → {timestamp, chunks, entity}
    frontier: list[FrontierItem] = []             # priority queue of pending URLs
    domain_scores: dict[str, DomainStats] = {}   # domain → hit stats
    query_stats: dict[str, QueryTemplateStats] = {}  # template key → stats
    entity_graph: dict[str, list[str]] = {}      # entity → co-occurring entities
    session_count: int = 0
    total_docs_ingested: int = 0
    total_chunks_added: int = 0
    last_run: Optional[str] = None


# ── State persistence ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> WebAgentState:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return WebAgentState.model_validate_json(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"State file corrupt, starting fresh: {exc}")
    return WebAgentState()


def save_state(state: WebAgentState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(state.model_dump_json(indent=2), encoding="utf-8")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _relevance(entity: str, text: str, base: float = 0.4) -> float:
    tokens = entity.lower().split()
    hits = sum(1 for t in tokens if t in text.lower())
    return round(min(1.0, base + 0.2 * hits), 3)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return ""


def _trusted(url: str) -> bool:
    host = _domain(url)
    return any(host.endswith(d) for d in TRUSTED_DOMAINS)


def _is_pdf(url: str) -> bool:
    return bool(_PDF_RE.search(url))


def _frontier_priority(
    relevance: float,
    url: str,
    state: WebAgentState,
    depth: int = 0,
) -> float:
    """Combine relevance, domain history, and depth into a single priority score."""
    domain = _domain(url)
    domain_boost = state.domain_scores.get(domain, DomainStats()).score
    depth_penalty = 0.1 * depth
    pdf_boost = 0.15 if _is_pdf(url) else 0.0
    trust_boost = 0.10 if _trusted(url) else 0.0
    return round(min(1.0, relevance * 0.6 + domain_boost * 0.3 + pdf_boost + trust_boost - depth_penalty), 3)


def _add_to_frontier(
    state: WebAgentState,
    url: str,
    priority: float,
    entity: str,
    source: str,
    depth: int = 0,
) -> bool:
    """Add a URL to the frontier if not already visited or queued. Returns True if added."""
    if url in state.visited_urls:
        return False
    if any(f.url == url for f in state.frontier):
        return False
    state.frontier.append(FrontierItem(
        url=url, priority=priority,
        trigger_entity=entity, source=source, depth=depth,
    ))
    return True


def _sort_frontier(state: WebAgentState) -> None:
    state.frontier.sort(key=lambda f: f.priority, reverse=True)
    if len(state.frontier) > FRONTIER_MAX:
        state.frontier = state.frontier[:FRONTIER_MAX]


def _record_visit(state: WebAgentState, url: str, entity: str, chunks: int) -> None:
    state.visited_urls[url] = {"timestamp": _now(), "chunks": chunks, "entity": entity}
    domain = _domain(url)
    if domain:
        if domain not in state.domain_scores:
            state.domain_scores[domain] = DomainStats()
        state.domain_scores[domain].attempts += 1
        if chunks > 0:
            state.domain_scores[domain].hits += 1


def _extract_links(html: str, base_url: str) -> list[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith(("mailto:", "javascript:", "#")):
            continue
        links.append(urljoin(base_url, href))
    return links


def _fetch_html(url: str, client: httpx.Client) -> Optional[str]:
    try:
        r = client.get(url, follow_redirects=True, timeout=REQUEST_TIMEOUT)
        if "text/html" in r.headers.get("content-type", ""):
            return r.text
    except Exception as exc:
        logger.debug(f"Fetch failed {url}: {exc}")
    return None


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _run_ddg_template(
    key: str,
    query: str,
    entity: str,
    state: WebAgentState,
    max_results: int = 10,
) -> list[DiscoveredDocument]:
    """Run one DuckDuckGo query template, update query_stats, return docs."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed — run: uv sync")
        return []

    if key not in state.query_stats:
        state.query_stats[key] = QueryTemplateStats()
    state.query_stats[key].runs += 1

    docs: list[DiscoveredDocument] = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            url = r.get("href") or r.get("url") or ""
            if not url or url in state.visited_urls:
                continue
            title = r.get("title") or url
            snippet = r.get("body") or None
            relevance = _relevance(entity, title + " " + (snippet or ""))
            if _trusted(url) or _is_pdf(url):
                relevance = min(1.0, relevance + 0.2)
            docs.append(DiscoveredDocument(
                title=title, url=url, source="duckduckgo",
                date=None, relevance_score=relevance,
                trigger_entity=entity, snippet=snippet,
            ))
        time.sleep(DDG_DELAY)
    except Exception as exc:
        logger.warning(f"DuckDuckGo query failed ({query!r}): {exc}")
        time.sleep(DDG_DELAY * 2)

    return docs


def _search_duckduckgo(
    entity: str,
    state: WebAgentState,
    max_results_per_template: int = 10,
) -> list[DiscoveredDocument]:
    """Run all query templates for one entity, ordered by historical yield."""
    # Sort templates by yield rate so highest-value queries run first
    sorted_templates = sorted(
        _DDG_TEMPLATES,
        key=lambda t: state.query_stats.get(t[0], QueryTemplateStats()).yield_rate,
        reverse=True,
    )
    docs: list[DiscoveredDocument] = []
    for key, template in sorted_templates:
        query = template.format(entity=entity)
        new_docs = _run_ddg_template(key, query, entity, state, max_results_per_template)
        docs.extend(new_docs)
        # Update yield stats with docs above 0.4 relevance
        hits = sum(1 for d in new_docs if d.relevance_score >= 0.4)
        state.query_stats[key].new_docs += hits

    logger.debug(f"DuckDuckGo: {len(docs)} results for '{entity}'")
    return docs


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _search_wikipedia(
    entity: str,
    client: httpx.Client,
    state: WebAgentState,
    max_pages: int = 3,
) -> list[DiscoveredDocument]:
    """Search Wikipedia for entity; harvest external links from article pages."""
    docs: list[DiscoveredDocument] = []
    try:
        r = client.get(
            WP_SEARCH_API,
            params={"action": "query", "list": "search", "srsearch": entity,
                    "srlimit": max_pages, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        page_titles = [h["title"] for h in r.json().get("query", {}).get("search", [])]
    except Exception as exc:
        logger.warning(f"Wikipedia search failed for '{entity}': {exc}")
        return []

    time.sleep(WP_DELAY)

    for title in page_titles:
        try:
            r = client.get(
                WP_SEARCH_API,
                params={"action": "query", "titles": title, "prop": "extlinks",
                        "ellimit": "50", "format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
        except Exception:
            continue

        for page in pages.values():
            for link in page.get("extlinks", []):
                url = link.get("*") or link.get("url") or ""
                if not url or url in state.visited_urls:
                    continue
                if not (_trusted(url) or _is_pdf(url)):
                    continue
                relevance = _relevance(entity, title)
                if _is_pdf(url):
                    relevance = min(1.0, relevance + 0.15)
                docs.append(DiscoveredDocument(
                    title=f"{title} [Wikipedia → {_domain(url)}]",
                    url=url, source="wikipedia", date=None,
                    relevance_score=relevance, trigger_entity=entity,
                    snippet=f"External link from Wikipedia: {title}",
                ))
        time.sleep(WP_DELAY)

    logger.debug(f"Wikipedia: {len(docs)} external links for '{entity}'")
    return docs


# ── CrossRef ──────────────────────────────────────────────────────────────────

def _search_crossref(
    entity: str,
    client: httpx.Client,
    state: WebAgentState,
    limit: int = 20,
    contact_email: str = "",
) -> list[DiscoveredDocument]:
    """CrossRef open-access papers — no API key required."""
    params: dict = {
        "query": entity, "filter": "has-full-text:true",
        "rows": limit, "select": "DOI,title,URL,published,abstract",
        "sort": "relevance", "order": "desc",
    }
    if contact_email:
        params["mailto"] = contact_email
    try:
        r = client.get(CROSSREF_BASE, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception as exc:
        logger.warning(f"CrossRef failed for '{entity}': {exc}")
        return []

    docs = []
    for item in items:
        title = ((item.get("title") or ["Untitled"])[0])
        url = item.get("URL") or item.get("DOI") or ""
        if not url or url in state.visited_urls:
            continue
        if not url.startswith("http"):
            url = f"https://doi.org/{url}"
        pub = item.get("published") or {}
        date_parts = (pub.get("date-parts") or [[]])[0]
        date_str = "-".join(str(p).zfill(2) for p in date_parts[:3]) if date_parts else None
        abstract = re.sub(r"<[^>]+>", "", item.get("abstract") or "")[:300]
        docs.append(DiscoveredDocument(
            title=title, url=url, source="crossref", date=date_str,
            relevance_score=_relevance(entity, title + " " + abstract),
            trigger_entity=entity, snippet=abstract or None,
        ))

    logger.debug(f"CrossRef: {len(docs)} results for '{entity}'")
    return docs


# ── Frontier drain ────────────────────────────────────────────────────────────

def drain_frontier(
    state: WebAgentState,
    geography: str,
    client: httpx.Client,
    batch_size: int = 30,
    min_relevance: float = 0.4,
) -> tuple[int, int]:
    """Visit the top-priority frontier items, ingest them, extract new links.

    Returns (docs_ingested, chunks_added).
    """
    from ai4saw.agents.fetch_agent import _ingest_document, _register_source, _licence_for_source
    from ai4saw.agents.fetch_agent import _is_registered

    _sort_frontier(state)
    batch = state.frontier[:batch_size]
    state.frontier = state.frontier[batch_size:]

    docs_ingested = 0
    chunks_added = 0

    for item in batch:
        if item.url in state.visited_urls or _is_registered(item.url):
            _record_visit(state, item.url, item.trigger_entity, 0)
            continue

        logger.info(f"[Frontier] Visiting: {item.url}")

        if item.priority < min_relevance and not _is_pdf(item.url) and not _trusted(item.url):
            _record_visit(state, item.url, item.trigger_entity, 0)
            continue

        doc = DiscoveredDocument(
            title=f"Frontier item ({_domain(item.url)})",
            url=item.url, source=item.source,
            date=None, relevance_score=item.priority,
            trigger_entity=item.trigger_entity,
        )
        chunks, filename = _ingest_document(doc, geography, client)
        _record_visit(state, item.url, item.trigger_entity, chunks)

        if chunks > 0:
            licence = _licence_for_source(item.source) if item.source in {
                "openalex", "semanticscholar", "arxiv", "internetarchive"
            } else "web"
            _register_source(filename, item.url, licence, geography, doc.title, item.source)
            docs_ingested += 1
            chunks_added += chunks

        # Extract links from HTML pages and add to frontier
        if not _is_pdf(item.url):
            html = _fetch_html(item.url, client)
            if html:
                for link in _extract_links(html, item.url):
                    if link in state.visited_urls:
                        continue
                    if not (_is_pdf(link) or _trusted(link)):
                        continue
                    priority = _frontier_priority(
                        _relevance(item.trigger_entity, link, base=0.3),
                        link, state, depth=item.depth + 1,
                    )
                    _add_to_frontier(state, link, priority, item.trigger_entity, item.source, item.depth + 1)

    return docs_ingested, chunks_added


# ── Public API ────────────────────────────────────────────────────────────────

def web_discover(
    entities: list[str],
    per_entity_limit: int = 15,
    sources_csv: str = "corpus/sources.csv",
    contact_email: str = "",
    state: Optional[WebAgentState] = None,
) -> tuple[DiscoveryResult, WebAgentState]:
    """Run one discovery pass: DDG + Wikipedia + CrossRef, add results to frontier.

    Returns the DiscoveryResult and the updated state (not yet saved — caller saves).
    """
    from ai4saw.discovery.discovery import _known_urls, _dedup_and_rank

    if state is None:
        state = load_state()

    known = _known_urls(sources_csv)
    all_docs: list[DiscoveredDocument] = []
    query_count = 0

    with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        for entity in entities:
            logger.info(f"[WebAgent] Discovering: {entity!r}")

            ddg_docs = _search_duckduckgo(entity, state, per_entity_limit)
            all_docs.extend(ddg_docs)
            query_count += len(_DDG_TEMPLATES)

            wp_docs = _search_wikipedia(entity, client, state)
            all_docs.extend(wp_docs)
            query_count += 1

            cr_docs = _search_crossref(entity, client, state, limit=per_entity_limit, contact_email=contact_email)
            all_docs.extend(cr_docs)
            query_count += 1
            time.sleep(CROSSREF_DELAY)

    deduped = _dedup_and_rank(all_docs, known | set(state.visited_urls.keys()))

    # Add all new docs to frontier
    added_to_frontier = 0
    for doc in deduped:
        priority = _frontier_priority(doc.relevance_score, doc.url, state)
        if _add_to_frontier(state, doc.url, priority, doc.trigger_entity, doc.source):
            added_to_frontier += 1

    _sort_frontier(state)
    state.session_count += 1
    state.last_run = _now()

    logger.info(
        f"[WebAgent] Discovery: {query_count} queries, {len(deduped)} new docs, "
        f"{added_to_frontier} added to frontier (frontier size: {len(state.frontier)})"
    )

    return DiscoveryResult(
        trigger_entities=entities,
        documents=deduped,
        query_count=query_count,
        new_documents=len(deduped),
    ), state


def get_state_summary(state: WebAgentState) -> dict:
    """Return a concise summary of agent state for display."""
    top_domains = sorted(
        ((d, s.score, s.hits, s.attempts) for d, s in state.domain_scores.items() if s.attempts > 0),
        key=lambda x: x[1], reverse=True,
    )[:5]
    top_queries = sorted(
        ((k, s.yield_rate, s.runs) for k, s in state.query_stats.items()),
        key=lambda x: x[1], reverse=True,
    )[:3]
    return {
        "sessions": state.session_count,
        "urls_visited": len(state.visited_urls),
        "frontier_size": len(state.frontier),
        "docs_ingested": state.total_docs_ingested,
        "chunks_added": state.total_chunks_added,
        "top_domains": top_domains,
        "top_queries": top_queries,
    }
