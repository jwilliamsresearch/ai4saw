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
from ai4saw.core.project import get_sources_csv
from ai4saw.core.search_graph import (
    g_record_seed,
    g_record_source_query,
    g_record_url,
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

# ── RSS feed registry (all confirmed working, no API key required) ────────────
# 83 feeds tested and verified returning HTTP 200 with valid content
_RSS_FEEDS: list[tuple[str, str]] = [
    # ── North Korea / nuclear / proliferation ────────────────────────────────
    ("rfa_korea",       "https://www.rfa.org/english/news/korea/rss2.xml"),
    ("nknews",          "https://www.nknews.org/feed/"),
    ("armscontrol",     "https://www.armscontrol.org/rss.xml"),
    ("nuclear_threat",  "https://www.nti.org/feed/"),
    ("ploughshares",    "https://www.ploughshares.org/feed"),
    ("fas_security",    "https://fas.org/feed/"),
    ("inkstick",        "https://inkstickmedia.com/feed/"),
    ("just_security",   "https://www.justsecurity.org/feed/"),
    ("war_on_rocks",    "https://warontherocks.com/feed/"),
    ("defense_news",    "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("spaceflight",     "https://api.spaceflightnewsapi.net/v4/articles/?limit=20&search=missile+nuclear+korea"),
    # ── International / global ────────────────────────────────────────────────
    ("bbc_world",       "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("bbc_asia",        "https://feeds.bbci.co.uk/news/world/asia/rss.xml"),
    ("aljazeera",       "https://www.aljazeera.com/xml/rss/all.xml"),
    ("guardian_world",  "https://www.theguardian.com/world/rss"),
    ("france24",        "https://www.france24.com/en/rss"),
    ("dw_english",      "https://rss.dw.com/rdf/rss-en-all"),
    ("le_monde_en",     "https://www.lemonde.fr/en/rss/une.xml"),
    ("politico_eu",     "https://www.politico.eu/feed/"),
    ("foreign_policy",  "https://foreignpolicy.com/feed/"),
    ("wapo_world",      "http://feeds.washingtonpost.com/rss/world"),
    ("bellingcat",      "https://www.bellingcat.com/feed/"),
    ("intercept",       "https://theintercept.com/feed/?rss"),
    ("newsweek",        "https://www.newsweek.com/rss"),
    ("time_world",      "https://time.com/feed/"),
    ("atlantic",        "https://feeds.feedburner.com/TheAtlantic"),
    ("npr_world",       "https://feeds.npr.org/1004/rss.xml"),
    ("sky_news",        "https://feeds.skynews.com/feeds/rss/world.xml"),
    ("independent_uk",  "https://www.independent.co.uk/news/world/rss"),
    ("telegraph",       "https://www.telegraph.co.uk/rss"),
    ("rfi_english",     "https://www.rfi.fr/en/rss"),
    ("euronews",        "https://feeds.feedburner.com/euronews/en/news/"),
    ("cbc_world",       "https://www.cbc.ca/cmlink/rss-world"),
    ("globe_mail",      "https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/world/?outputType=xml"),
    ("tass",            "https://tass.com/rss/v2.xml"),
    ("moscow_times",    "https://www.themoscowtimes.com/rss/news"),
    ("brookings",       "https://www.brookings.edu/feed/"),
    ("the_conversation","https://theconversation.com/articles.atom"),
    ("un_news",         "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
    ("reliefweb",       "https://reliefweb.int/updates/rss.xml"),
    # ── Asia / Pacific ────────────────────────────────────────────────────────
    ("nhk_world",       "https://www3.nhk.or.jp/rss/news/cat6.xml"),
    ("korea_herald",    "https://www.koreaherald.com/rss/02.xml"),
    ("yonhap",          "https://en.yna.co.kr/RSS/news.xml"),
    ("asia_times",      "https://asiatimes.com/feed/"),
    ("the_diplomat",    "https://thediplomat.com/feed/"),
    ("nikkei_asia",     "https://asia.nikkei.com/rss/feed/nar"),
    ("abc_australia",   "https://www.abc.net.au/news/feed/51120/rss.xml"),
    ("philstar",        "https://www.philstar.com/rss/headlines"),
    ("japan_times",     "https://www.japantimes.co.jp/feed/topstories/"),
    ("scmp",            "https://www.scmp.com/rss/91/feed"),
    ("hkfp",            "https://www.hongkongfp.com/feed/"),
    ("bangkok_post",    "https://www.bangkokpost.com/rss/data/topstories.xml"),
    ("straits_times",   "https://www.straitstimes.com/news/world/rss.xml"),
    ("the_hindu",       "https://www.thehindu.com/news/international/?service=rss"),
    ("times_india",     "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    ("the_wire_in",     "https://thewire.in/rss"),
    ("scroll_in",       "https://scroll.in/feed"),
    ("nation_thailand", "https://www.nationthailand.com/rss"),
    ("nz_herald",       "https://www.nzherald.co.nz/arc/outboundfeeds/rss/section/world/?outputType=xml"),
    ("smh",             "https://www.smh.com.au/rss/world.xml"),
    ("myanmar_now",     "https://myanmar-now.org/en/feed/"),
    ("global_times",    "https://www.globaltimes.cn/rss/outbrain.xml"),
    ("mongolia_news",   "https://news.mn/en/feed/"),
    ("georgia_today",   "https://georgiatoday.ge/feed/"),
    ("kashmir_obs",     "https://kashmirobserver.net/feed/"),
    # ── Middle East ───────────────────────────────────────────────────────────
    ("jpost",           "https://www.jpost.com/Rss/RssFeedsHeadlines.aspx"),
    ("haaretz",         "https://www.haaretz.com/cmlink/1.628765"),
    ("arab_news",       "https://www.arabnews.com/rss.xml"),
    ("new_arab",        "https://english.alaraby.co.uk/rss"),
    ("middle_east_eye", "https://www.middleeasteye.net/rss"),
    ("kurdistan24",     "https://www.kurdistan24.net/en/rss.xml"),
    # ── Africa ────────────────────────────────────────────────────────────────
    ("allafrica",       "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf"),
    ("mail_guardian",   "https://mg.co.za/feed/"),
    ("premium_times",   "https://www.premiumtimesng.com/feed"),
    ("vanguard_ng",     "https://www.vanguardngr.com/feed/"),
    # ── Latin America ─────────────────────────────────────────────────────────
    ("mercopress",      "https://en.mercopress.com/rss"),
    ("ba_herald",       "https://buenosairesherald.com/feed"),
    ("nacla",           "https://nacla.org/taxonomy/term/6/feed"),
    # ── Human rights / humanitarian / investigative ───────────────────────────
    ("amnesty",         "https://www.amnesty.org/en/latest/feed/"),
    ("new_humanitarian","https://www.thenewhumanitarian.org/rss.xml"),
    ("refugees_intl",   "https://www.refugeesinternational.org/feed/"),
    ("rinj",            "https://rinj.org/feed/"),
]

RSS_REQUEST_TIMEOUT = 12.0


# ── RSS feed discovery ────────────────────────────────────────────────────────

def _query_rss_feeds(
    entities: list[str],
    client: httpx.Client,
    limit_per_feed: int = 10,
    on_event=None,
) -> list[DiscoveredDocument]:
    """Scan all RSS feeds and return articles matching any entity term."""
    import xml.etree.ElementTree as ET

    entity_terms = [e.lower() for e in entities if e]
    docs: list[DiscoveredDocument] = []
    headers = {"User-Agent": "ai4saw/0.1 (research; academic) httpx"}
    if on_event: on_event("info", f"RSS: scanning {len(_RSS_FEEDS)} feeds…")

    for feed_id, feed_url in _RSS_FEEDS:
        try:
            r = client.get(feed_url, headers=headers, timeout=RSS_REQUEST_TIMEOUT,
                           follow_redirects=True)
            if not r.is_success:
                continue

            # Handle JSON feeds (Spaceflight News API)
            if "json" in r.headers.get("content-type", ""):
                import json as _json
                data = r.json()
                items = data.get("results", data.get("articles", []))
                for item in items[:limit_per_feed]:
                    title   = item.get("title", "")
                    url     = item.get("url", "")
                    summary = item.get("summary", "")
                    if not url:
                        continue
                    combined = (title + " " + summary).lower()
                    if not any(t in combined for t in entity_terms):
                        continue
                    relevance = _relevance(entities[0], title + " " + summary)
                    docs.append(DiscoveredDocument(
                        title=title, url=url, source=feed_id,
                        date=item.get("published_at", "")[:10] or None,
                        relevance_score=relevance,
                        trigger_entity=entities[0],
                        snippet=summary[:300],
                    ))
                continue

            # Parse RSS/Atom XML
            try:
                root = ET.fromstring(r.content)
            except ET.ParseError:
                continue

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            # RSS 2.0
            items = root.findall(".//item")
            # Atom fallback
            if not items:
                items = root.findall(".//atom:entry", ns)

            count = 0
            for item in items:
                if count >= limit_per_feed:
                    break
                title = (
                    getattr(item.find("title"), "text", "") or
                    getattr(item.find("atom:title", ns), "text", "") or ""
                )
                link = (
                    getattr(item.find("link"), "text", "") or
                    (item.find("link") is not None and item.find("link").get("href", "")) or
                    getattr(item.find("atom:link", ns), "text", "") or
                    (item.find("atom:link", ns) is not None and
                     item.find("atom:link", ns).get("href", "")) or ""
                )
                desc = (
                    getattr(item.find("description"), "text", "") or
                    getattr(item.find("summary"), "text", "") or
                    getattr(item.find("atom:summary", ns), "text", "") or ""
                )
                pub = (
                    getattr(item.find("pubDate"), "text", "") or
                    getattr(item.find("atom:published", ns), "text", "") or ""
                )

                if not link or not title:
                    continue

                combined = (title + " " + (desc or "")).lower()
                if not any(t in combined for t in entity_terms):
                    continue

                relevance = _relevance(entities[0], title + " " + (desc or ""))
                docs.append(DiscoveredDocument(
                    title=title.strip(), url=link.strip(), source=feed_id,
                    date=pub[:10] if pub else None,
                    relevance_score=relevance,
                    trigger_entity=entities[0],
                    snippet=(desc or "")[:300],
                ))
                count += 1

        except Exception as exc:
            logger.debug(f"RSS feed {feed_id} failed: {exc}")
            if on_event: on_event("error", f"RSS {feed_id}: {str(exc)[:50]}")
            continue

        # Report per-feed results only when matches found
        feed_docs = [d for d in docs if d.source == feed_id]
        if feed_docs and on_event:
            on_event("info", f"RSS {feed_id}: {len(feed_docs)} matches")

    total = len(docs)
    if on_event:
        on_event("info", f"RSS complete: {total} articles from {len(_RSS_FEEDS)} feeds")
    logger.info(f"RSS feeds: {total} matching articles across {len(_RSS_FEEDS)} feeds")
    return docs


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _known_urls(sources_csv: Optional[str] = None) -> set[str]:
    path = Path(sources_csv) if sources_csv else get_sources_csv()
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


# ── DOAJ — Directory of Open Access Journals ─────────────────────────────────

def _query_doaj(
    entities: list[str],
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """DOAJ full-text search — open access journal articles, no API key required."""
    if not entities:
        return []
    query = " OR ".join(f'"{e}"' for e in entities[:5])
    params = {"pageSize": limit}
    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
    docs: list[DiscoveredDocument] = []
    try:
        r = client.get(
            f"https://doaj.org/api/v2/search/articles/{httpx.URL(query)}",
            params=params,
        )
        if not r.is_success:
            return []
        for item in r.json().get("results", []):
            bib = item.get("bibjson", {})
            title = bib.get("title", "")
            links = bib.get("link", [])
            url = next((lk.get("url", "") for lk in links if lk.get("type") == "fulltext"), "")
            if not url:
                url = next((lk.get("url", "") for lk in links), "")
            if not url or not title:
                continue
            abstract = bib.get("abstract", "")
            year = str(bib.get("year", "")) or None
            trigger = entities[0]
            docs.append(DiscoveredDocument(
                title=title, url=url, source="doaj",
                date=year, relevance_score=_relevance(trigger, title + " " + abstract),
                trigger_entity=trigger, snippet=abstract[:300],
            ))
    except Exception as exc:
        logger.warning(f"DOAJ failed: {exc}")
    finally:
        if close:
            client.close()
    logger.debug(f"DOAJ: {len(docs)} results")
    return docs


# ── Europe PMC ────────────────────────────────────────────────────────────────

def _query_europepmc(
    entities: list[str],
    limit: int = 50,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """Europe PMC — open-access biomedical and life sciences literature."""
    if not entities:
        return []
    query = " OR ".join(f'"{e}"' for e in entities[:5])
    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
    docs: list[DiscoveredDocument] = []
    try:
        r = client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": limit,
                    "resultType": "core", "sort": "RELEVANCE"},
        )
        if not r.is_success:
            return []
        trigger = entities[0]
        for item in r.json().get("resultList", {}).get("result", []):
            title = item.get("title", "")
            pmid  = item.get("pmid", "")
            doi   = item.get("doi", "")
            url   = (f"https://doi.org/{doi}" if doi else
                     f"https://europepmc.org/article/MED/{pmid}" if pmid else "")
            if not url or not title:
                continue
            abstract = item.get("abstractText", "")
            docs.append(DiscoveredDocument(
                title=title, url=url, source="europepmc",
                date=str(item.get("pubYear", "")) or None,
                relevance_score=_relevance(trigger, title + " " + abstract),
                trigger_entity=trigger, snippet=abstract[:300],
            ))
    except Exception as exc:
        logger.warning(f"Europe PMC failed: {exc}")
    finally:
        if close:
            client.close()
    logger.debug(f"Europe PMC: {len(docs)} results")
    return docs


# ── PubMed E-utilities ────────────────────────────────────────────────────────

def _query_pubmed(
    entities: list[str],
    limit: int = 30,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """PubMed via NCBI E-utilities — no API key required (3 req/s limit)."""
    if not entities:
        return []
    query = " OR ".join(f'"{e}"' for e in entities[:5])
    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
    docs: list[DiscoveredDocument] = []
    try:
        # Step 1: search for IDs
        r = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": limit,
                    "retmode": "json", "sort": "relevance"},
        )
        if not r.is_success:
            return []
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        # Step 2: fetch summaries
        r2 = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        )
        if not r2.is_success:
            return []
        result = r2.json().get("result", {})
        trigger = entities[0]
        for pmid in ids:
            item = result.get(pmid, {})
            title = item.get("title", "")
            if not title:
                continue
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            year = item.get("pubdate", "")[:4] or None
            docs.append(DiscoveredDocument(
                title=title, url=url, source="pubmed",
                date=year, relevance_score=_relevance(trigger, title),
                trigger_entity=trigger, snippet=None,
            ))
    except Exception as exc:
        logger.warning(f"PubMed failed: {exc}")
    finally:
        if close:
            client.close()
    logger.debug(f"PubMed: {len(docs)} results")
    return docs


# ── World Bank Open Documents ─────────────────────────────────────────────────

def _query_worldbank(
    entities: list[str],
    limit: int = 30,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """World Bank Documents & Reports — policy papers, country assessments, free."""
    if not entities:
        return []
    query = " ".join(entities[:5])
    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
    docs: list[DiscoveredDocument] = []
    try:
        r = client.get(
            "https://search.worldbank.org/api/v2/wds",
            params={"format": "json", "q": query, "rows": limit,
                    "fl": "docdt,display_title,url,docty,pdfurl"},
        )
        if not r.is_success:
            return []
        trigger = entities[0]
        for item in r.json().get("documents", {}).values():
            title = item.get("display_title", "")
            url   = item.get("pdfurl", "") or item.get("url", "")
            if not url or not title:
                continue
            date = str(item.get("docdt", ""))[:10] or None
            docs.append(DiscoveredDocument(
                title=title, url=url, source="worldbank",
                date=date, relevance_score=_relevance(trigger, title),
                trigger_entity=trigger, snippet=item.get("docty", ""),
            ))
    except Exception as exc:
        logger.warning(f"World Bank failed: {exc}")
    finally:
        if close:
            client.close()
    logger.debug(f"World Bank: {len(docs)} results")
    return docs


# ── HDX — Humanitarian Data Exchange ─────────────────────────────────────────

def _query_hdx(
    entities: list[str],
    limit: int = 20,
    client: httpx.Client | None = None,
) -> list[DiscoveredDocument]:
    """OCHA Humanitarian Data Exchange — datasets and reports, no API key."""
    if not entities:
        return []
    query = " ".join(entities[:5])
    close = client is None
    if close:
        client = httpx.Client(timeout=REQUEST_TIMEOUT)
    docs: list[DiscoveredDocument] = []
    try:
        r = client.get(
            "https://data.humdata.org/api/3/action/package_search",
            params={"q": query, "rows": limit},
        )
        if not r.is_success:
            return []
        trigger = entities[0]
        for pkg in r.json().get("result", {}).get("results", []):
            title = pkg.get("title", "")
            name  = pkg.get("name", "")
            url   = f"https://data.humdata.org/dataset/{name}" if name else ""
            if not url or not title:
                continue
            notes = pkg.get("notes", "")[:300]
            date  = pkg.get("metadata_modified", "")[:10] or None
            docs.append(DiscoveredDocument(
                title=title, url=url, source="hdx",
                date=date, relevance_score=_relevance(trigger, title + " " + notes),
                trigger_entity=trigger, snippet=notes,
            ))
    except Exception as exc:
        logger.warning(f"HDX failed: {exc}")
    finally:
        if close:
            client.close()
    logger.debug(f"HDX: {len(docs)} results")
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
    on_event=None,
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

    # Record seed queries in the provenance graph
    for entity in entities:
        g_record_seed(entity)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        # Per-entity sources (generous rate limits)
        for entity in entities:
            logger.info(f"Discovering: {entity!r}")

            g_record_source_query(entity, "openalex")
            new = _query_openalex(entity, per_entity_limit, client, contact_email)
            all_docs.extend(new)
            query_count += 1
            time.sleep(OPENALEX_DELAY)

            g_record_source_query(entity, "internetarchive")
            new = _query_internetarchive(entity, per_entity_limit, client)
            all_docs.extend(new)
            query_count += 1
            time.sleep(delay)

        # Batched sources (one OR query for all entities — avoids per-IP rate bans)
        batch_label = " | ".join(entities[:5])

        logger.info(f"Semantic Scholar: batch query for {len(entities)} entities")
        g_record_source_query(batch_label, "semanticscholar")
        new = _query_semanticscholar_batch(entities, limit=100, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        logger.info(f"arXiv: batch query for {len(entities)} entities")
        g_record_source_query(batch_label, "arxiv")
        new = _query_arxiv_batch(entities, limit=100, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        time.sleep(6)  # GDELT enforces 1 req/5s strictly; guarantee gap from previous calls
        logger.info(f"GDELT: batch query for {len(entities)} entities")
        g_record_source_query(batch_label, "gdelt")
        new = _query_gdelt_batch(entities, limit=250, client=client)
        all_docs.extend(new)
        query_count += 1

        logger.info(f"RSS feeds: scanning {len(_RSS_FEEDS)} feeds for {len(entities)} entities")
        new = _query_rss_feeds(entities, client, on_event=on_event)
        for doc in new:
            g_record_source_query(doc.trigger_entity or batch_label, doc.source)
        all_docs.extend(new)
        query_count += 1

        logger.info(f"DOAJ: open access journals for {len(entities)} entities")
        g_record_source_query(batch_label, "doaj")
        new = _query_doaj(entities, limit=50, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        logger.info(f"Europe PMC: biomedical/life sciences for {len(entities)} entities")
        g_record_source_query(batch_label, "europepmc")
        new = _query_europepmc(entities, limit=50, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        logger.info(f"PubMed: NCBI literature for {len(entities)} entities")
        g_record_source_query(batch_label, "pubmed")
        new = _query_pubmed(entities, limit=30, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        logger.info(f"World Bank: policy documents for {len(entities)} entities")
        g_record_source_query(batch_label, "worldbank")
        new = _query_worldbank(entities, limit=30, client=client)
        all_docs.extend(new)
        query_count += 1
        time.sleep(delay)

        logger.info(f"HDX: humanitarian data for {len(entities)} entities")
        g_record_source_query(batch_label, "hdx")
        new = _query_hdx(entities, limit=20, client=client)
        all_docs.extend(new)
        query_count += 1

    deduped = _dedup_and_rank(all_docs, known)

    # Record all discovered URLs in the provenance graph
    for doc in deduped:
        g_record_url(
            doc.url,
            doc.title,
            doc.source,
            trigger_query=doc.trigger_entity,
            query_type="seed_query",
        )
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
