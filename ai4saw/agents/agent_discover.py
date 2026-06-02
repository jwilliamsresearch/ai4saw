"""Genuinely agentic corpus discovery.

Unlike the stateful web crawler (web_agent.py), this agent has an LLM in the
reasoning loop.  After each document is ingested, the LLM reads an excerpt and
decides what to search for next — generating novel, specific queries based on
what it found rather than substituting entities into fixed templates.

ReAct loop per session
───────────────────────
  Observe  → drain frontier → ingest documents → capture text excerpts
  Think    → LLM reads each excerpt → identifies novel entities + generates queries
  Act      → execute LLM-generated queries → add results to frontier
  Update   → persist state + reasoning log → sleep → repeat

Example snowball
─────────────────
  Initial entity: "Location Alpha"
  Ingests: International Tribunal judgment on Commander Beta
  LLM extracts: Witness Alpha, Unit Alpha, Site Alpha
  LLM generates:
    "Witness Alpha plea agreement International Tribunal 1996"
    "Unit Alpha Armed-Group-Alpha executions a specific date"
    "Site Alpha massacre witnesses"
  → 3 novel, specific queries that no template would have produced

State:   output/agent_discover_state.json
Log:     output/agent_discover_log.jsonl  (one entry per document reasoned on)
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from ai4saw.core.models import DiscoveredDocument
from ai4saw.agents.web_agent import (
    FrontierItem, DomainStats, QueryTemplateStats,
    _now, _relevance, _domain, _trusted, _is_pdf,
    _frontier_priority, _add_to_frontier, _sort_frontier,
    _record_visit, _extract_links, _fetch_html,
    TRUSTED_DOMAINS, DDG_DELAY,
)


# ── Paths ──────────────────────────────────────────────────────────────────────

AGENT_STATE_FILE = Path("output/agent_discover_state.json")
AGENT_LOG_FILE   = Path("output/agent_discover_log.jsonl")
FRONTIER_MAX     = 5_000

# Max documents to run LLM reasoning on per session.
# Each reasoning call = one LLM invocation (~5-15s on Ollama).
MAX_REASONING_PER_SESSION = 8


# ── LLM prompts ───────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = (
    "You are a research intelligence analyst. You are building an evidence "
    "corpus for academic research on: {topic}. Your job is to read document "
    "excerpts and identify high-value leads for further investigation."
)


def _make_system(initial_entities: list[str], geography: str) -> str:
    topic = ", ".join(initial_entities[:5]) + (f" ({geography})" if geography else "")
    return _SYSTEM_TEMPLATE.format(topic=topic)

_PRESCREEN_PROMPT = """\
Research project: {geography}
Seed entities: {entities}

Candidate document
  URL:    {url}
  Title:  {title}
  Source: {source}
  Text sample (may be navigation/headers — use URL and title as primary signals):
---
{text}
---

Decide whether this document is worth ingesting into the corpus.

Ingest if the document contains ANY substantive content related to the research topic,
including: analysis, news reporting, policy documents, technical reports, academic papers,
government statements, NGO reports, or expert commentary.

Skip ONLY if:
  - The text sample is clearly an error page (403, 404, blank, redirect)
  - The text sample contains zero words related to the research topic
  - It is obviously a completely different subject (e.g. cooking recipes when researching nuclear weapons)

When in doubt, INGEST. It is better to include a borderline document than to miss evidence.

Output ONLY valid JSON:
{{
  "ingest": true,
  "reason": "one sentence explaining your decision"
}}"""

_PROMPT = """\
Research project: {geography} conflict corpus
Initial entities: {entities}

Document excerpt
  Source: {source} | Trigger: {trigger_entity}
  Title:  {title}
---
{text}
---

Already tracked entities (do not repeat): {known_entities}

What has worked so far:
  High-yield domains: {good_domains}
  Successful recent queries: {good_queries}

What has NOT worked:
  Low-yield domains (avoid): {bad_domains}
  Already executed queries (do not repeat): {executed_queries}

Based on the excerpt above:

1. NEW named entities — people, military units, locations, events, legal instruments
   that appear in this text but are NOT in the already-tracked list.

2. Specific search queries (2–3) that would find corroborating or extending evidence.
   RULES — every query MUST:
     a) Include at least one anchor from the initial entities or geography
        (e.g. the conflict location, country, or topic from the research project)
        so results stay on-topic.
     b) Include the specific new entity or event name you discovered.
     c) Be precise — add dates, unit numbers, court case IDs, locations where known.
   Prefer sources from the high-yield domains listed above.
   Do NOT use site: operators — let the search engine find sources naturally.
   Good: "[Place Name] [topic] [specific event or entity] [year]"
   Good: "[Person/unit name] [location] [role or action] [year] report"
   Good: "[Organisation] [event] [location] [year] findings"
   Bad:  "[entity alone with no topic anchor]"   ← too generic, off-topic results
   Bad:  "site:hrw.org North Korea"              ← site: operators restrict too much

Output ONLY valid JSON (no text outside the braces):
{{
  "novel_entities": ["Entity A", "Entity B"],
  "queries": ["specific query 1", "specific query 2", "specific query 3"],
  "reasoning": "one sentence explaining why these queries are high-value"
}}"""


# ── State models ───────────────────────────────────────────────────────────────

class DiscoveryReasoning(BaseModel):
    """What the LLM produced from one document."""
    source_url: str
    source_title: str
    trigger_entity: str
    novel_entities: list[str]
    generated_queries: list[str]
    reasoning: str
    timestamp: str = Field(default_factory=_now)


class AgentDiscoverState(BaseModel):
    """Full persistent state for the agentic discovery loop."""
    # Search frontier (shared pattern with web_agent)
    visited_urls: dict[str, dict] = {}
    frontier: list[FrontierItem] = []
    domain_scores: dict[str, DomainStats] = {}
    query_stats: dict[str, QueryTemplateStats] = {}

    # Agent-specific
    initial_entities: list[str] = []
    # entity -> {from_url, timestamp, mention_count}
    # mention_count increments each session a doc references this entity
    discovered_entities: dict[str, dict] = {}
    query_queue: list[str] = []       # LLM-generated queries pending execution
    executed_queries: set[str] = Field(default_factory=set)  # O(1) lookup

    # Counters
    session_count: int = 0
    docs_reasoned: int = 0
    total_docs_ingested: int = 0
    total_docs_skipped: int = 0
    total_chunks_added: int = 0
    last_run: Optional[str] = None


# ── Persistence ────────────────────────────────────────────────────────────────

def load_agent_state() -> AgentDiscoverState:
    AGENT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if AGENT_STATE_FILE.exists():
        try:
            return AgentDiscoverState.model_validate_json(
                AGENT_STATE_FILE.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning(f"Agent state corrupt, starting fresh: {exc}")
    return AgentDiscoverState()


def save_agent_state(state: AgentDiscoverState) -> None:
    AGENT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGENT_STATE_FILE.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _log_reasoning(reasoning: DiscoveryReasoning) -> None:
    AGENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(reasoning.model_dump_json() + "\n")


# ── LLM reasoning ─────────────────────────────────────────────────────────────

def _llm_reason(
    text: str,
    doc: DiscoveredDocument,
    initial_entities: list[str],
    geography: str,
    known_entities: list[str],
    state: Optional["AgentDiscoverState"] = None,
) -> Optional[DiscoveryReasoning]:
    """Call the LLM to reason about what to search for next.

    Returns None on parse failure — callers should handle gracefully.
    """
    if not text.strip():
        return None

    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    known_sample = ", ".join(known_entities[:30]) if known_entities else "none yet"

    # Build context from prior session performance
    good_domains = bad_domains = "unknown"
    good_queries = executed_sample = "none yet"
    if state:
        scored = sorted(state.domain_scores.items(),
                        key=lambda kv: kv[1].hits, reverse=True)
        good_domains = ", ".join(d for d, s in scored[:5] if s.hits > 0) or "none yet"
        bad_domains  = ", ".join(d for d, s in scored if s.hits == 0)[:120] or "none"
        # Sample of queries that produced novel entities (in query_queue or recently executed)
        good_queries = "; ".join(list(state.executed_queries)[-5:]) or "none yet"
        executed_sample = "; ".join(list(state.executed_queries)[-10:]) or "none"

    prompt = _PROMPT.format(
        geography=geography,
        entities=", ".join(initial_entities),
        source=doc.source,
        trigger_entity=doc.trigger_entity,
        title=doc.title[:120],
        text=text[:2_000],
        known_entities=known_sample,
        good_domains=good_domains,
        bad_domains=bad_domains,
        good_queries=good_queries,
        executed_queries=executed_sample,
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_make_system(initial_entities, geography)),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()

        # Strip markdown code fences if the model wraps output
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        reasoning = DiscoveryReasoning(
            source_url=doc.url,
            source_title=doc.title[:120],
            trigger_entity=doc.trigger_entity,
            novel_entities=[str(e) for e in data.get("novel_entities", [])[:10]],
            generated_queries=[str(q) for q in data.get("queries", [])[:3]],
            reasoning=str(data.get("reasoning", ""))[:300],
        )
        logger.info(
            f"[AgentReason] {doc.url[:60]}\n"
            f"  Novel entities: {reasoning.novel_entities}\n"
            f"  Queries: {reasoning.generated_queries}\n"
            f"  Why: {reasoning.reasoning}"
        )
        return reasoning

    except Exception as exc:
        logger.warning(f"LLM reasoning failed for {doc.url}: {exc}")
        return None


# ── Text sampling ─────────────────────────────────────────────────────────────

def _sample_text(
    raw_docs: list,
    total_chars: int = 2_000,
    slices: int = 5,
) -> str:
    """Sample evenly-spaced slices across all raw document pages.

    Takes `slices` windows of `total_chars // slices` chars each, drawn from
    evenly-distributed positions in the full document. This ensures the LLM
    sees substantive content (witness names, unit numbers, event dates) rather
    than only the cover page and table of contents.
    """
    if not raw_docs:
        return ""
    full_text = " ".join(d.page_content for d in raw_docs)
    if len(full_text) <= total_chars:
        return full_text

    window = total_chars // slices
    step = max(1, (len(full_text) - window) // (slices - 1)) if slices > 1 else len(full_text)
    parts = []
    for i in range(slices):
        start = min(i * step, len(full_text) - window)
        parts.append(full_text[start : start + window])
    return " … ".join(parts)


# ── Ingest and capture text ────────────────────────────────────────────────────

def _ingest_and_capture(
    doc: DiscoveredDocument,
    geography: str,
    client: httpx.Client,
    initial_entities: Optional[list[str]] = None,
    on_event: Optional[Callable[[str, str], None]] = None,
) -> tuple[int, str, str]:
    """Ingest one document; return (chunks_added, filename, text_excerpt).

    text_excerpt is the first 2 000 chars of raw document text — used for LLM
    reasoning.  Empty string on failure or if LLM prescreens as irrelevant.
    """
    from ai4saw.ingestion.chunker import chunk_documents
    from ai4saw.ingestion.embedder import embed_and_store
    from ai4saw.ingestion.loaders import load_document
    from ai4saw.agents.fetch_agent import (
        _detect_content_type, _download_pdf, _safe_filename, CORPUS_DIR,
    )
    from datetime import date as _date

    content_type = _detect_content_type(doc.url, client)
    is_pdf = "pdf" in content_type or doc.url.lower().endswith(".pdf")
    filename = _safe_filename(doc.url, doc.title, doc.source)

    if is_pdf:
        dest = CORPUS_DIR / filename
        if not _download_pdf(doc.url, dest, client):
            return 0, filename, ""
        source_arg: str = str(dest)
        source_url_arg: Optional[str] = doc.url
    else:
        source_arg = doc.url
        source_url_arg = doc.url

    parsed_date: Optional[_date] = None
    if doc.date:
        try:
            parsed_date = _date.fromisoformat(doc.date)
        except ValueError:
            pass

    try:
        raw_docs = load_document(
            source=source_arg, doc_type="report", language="en",
            date_published=parsed_date, geography=geography,
            source_url=source_url_arg,
        )
    except Exception as exc:
        logger.warning(f"load_document failed for {doc.url}: {exc}")
        return 0, filename, ""

    text_excerpt = _sample_text(raw_docs, total_chars=2_000, slices=5)
    domain = _domain(doc.url)

    should_ingest, reason = _llm_prescreen(
        text_excerpt, doc, initial_entities or [], geography
    )
    if not should_ingest:
        if on_event: on_event("skip", f"{domain} — {reason[:70]}")
        if is_pdf:
            dest = CORPUS_DIR / filename
            if dest.exists():
                dest.unlink(missing_ok=True)
        return 0, filename, ""

    chunks = chunk_documents(raw_docs)
    embed_and_store(chunks)
    return len(chunks), filename, text_excerpt


# ── Pre-ingest LLM filter ─────────────────────────────────────────────────────

def _llm_prescreen(
    text: str,
    doc: DiscoveredDocument,
    initial_entities: list[str],
    geography: str,
) -> bool:
    """Ask the LLM whether a document is worth ingesting. Returns True = ingest, False = skip.

    Falls back to True on any error so a broken LLM call never silently drops documents.
    """
    if not text.strip():
        return False

    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    prompt = _PRESCREEN_PROMPT.format(
        geography=geography,
        entities=", ".join(initial_entities[:10]),
        url=doc.url,
        title=doc.title[:120],
        source=doc.source,
        text=text[:1_500],
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_make_system(initial_entities, geography)),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        ingest = bool(data.get("ingest", True))
        reason = str(data.get("reason", ""))
        verdict = "INGEST" if ingest else "SKIP"
        logger.info(f"[Prescreen] {verdict} — {doc.url[:70]}\n  Reason: {reason}")
        return ingest, reason
    except Exception as exc:
        logger.warning(f"[Prescreen] LLM failed for {doc.url}, defaulting to ingest: {exc}")
        return True, ""


# ── DuckDuckGo for novel queries ───────────────────────────────────────────────

def _execute_novel_query(
    query: str,
    entities: list[str],
    state: AgentDiscoverState,
    max_results: int = 10,
) -> list[DiscoveredDocument]:
    """Run a single LLM-generated query via DuckDuckGo."""
    if query in state.executed_queries:
        return []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        logger.warning("ddgs not installed — run: pip install ddgs")
        return []

    docs = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        trigger = entities[0] if entities else "unknown"
        for r in results:
            url = r.get("href") or r.get("url") or ""
            if not url or url in state.visited_urls:
                continue
            title = r.get("title") or url
            snippet = r.get("body") or None
            relevance = _relevance(trigger, title + " " + (snippet or ""), base=0.5)
            if _trusted(url) or _is_pdf(url):
                relevance = min(1.0, relevance + 0.2)
            docs.append(DiscoveredDocument(
                title=title, url=url, source="duckduckgo",
                date=None, relevance_score=relevance,
                trigger_entity=trigger, snippet=snippet,
            ))
        state.executed_queries.add(query)
        time.sleep(DDG_DELAY)
    except Exception as exc:
        logger.warning(f"Novel query DDG failed ({query!r}): {exc}")
        time.sleep(DDG_DELAY * 2)
    return docs


# ── Session seed (first session or re-seed) ────────────────────────────────────

def _seed_frontier(
    entities: list[str],
    state: AgentDiscoverState,
    per_entity_limit: int = 10,
    contact_email: str = "",
    use_api_sources: bool = True,
    on_event: Optional[Callable[[str, str], None]] = None,
    use_ddg: bool = True,
) -> int:
    """Seed the frontier from DDG/Wikipedia/CrossRef and (optionally) academic APIs.

    use_api_sources=True adds OpenAlex, Semantic Scholar, arXiv, Internet Archive,
    and GDELT results directly into the frontier.
    """
    from ai4saw.agents.web_agent import web_discover

    label = "DDG · Wikipedia · CrossRef…" if use_ddg else "Wikipedia · CrossRef…"
    if on_event: on_event("info", f"Seeding: {label}")
    _, state_out = web_discover(
        entities=entities,
        per_entity_limit=per_entity_limit,
        contact_email=contact_email,
        on_event=on_event,
        use_ddg=use_ddg,
        state=AgentDiscoverState(  # type: ignore[arg-type]  — duck-typed
            visited_urls=state.visited_urls,
            frontier=state.frontier,
            domain_scores=state.domain_scores,
            query_stats=state.query_stats,
        ),
    )
    state.visited_urls = state_out.visited_urls  # type: ignore[assignment]
    state.frontier = state_out.frontier  # type: ignore[assignment]
    state.domain_scores = state_out.domain_scores  # type: ignore[assignment]
    state.query_stats = state_out.query_stats  # type: ignore[assignment]

    if use_api_sources:
        try:
            if on_event: on_event("info", "Seeding: OpenAlex · arXiv · Internet Archive · GDELT…")
            from ai4saw.discovery.discovery import discover_for_entities
            # Cap to 5 entities — API sources batch them so more = proportionally slower
            api_entities = entities[:5]
            result = discover_for_entities(api_entities, per_entity_limit=per_entity_limit)
            added = 0
            for doc in result.documents:
                if doc.url not in state.visited_urls:
                    priority = _frontier_priority(doc.relevance_score, doc.url, state)
                    _add_to_frontier(state, doc.url, priority, doc.trigger_entity, doc.source)
                    added += 1
            logger.info(f"[Seed] API sources added {added} URLs to frontier")
        except Exception as exc:
            logger.warning(f"[Seed] API source discovery failed: {exc}")

    return len(state.frontier)


# ── Core agent session ────────────────────────────────────────────────────────

def run_agent_session(
    state: AgentDiscoverState,
    geography: str,
    frontier_batch: int = 20,
    min_relevance: float = 0.4,
    max_reasoning: int = MAX_REASONING_PER_SESSION,
    on_event: Optional[Callable[[str, str], None]] = None,
    on_reasoning: Optional[Callable[["DiscoveryReasoning"], None]] = None,
) -> tuple[int, int, list[DiscoveryReasoning]]:
    """Run one full agent session.

    Phases:
      1. Drain frontier → ingest → capture text
      2. LLM reasoning on top-N ingested docs → novel entities + queries
      3. Execute novel queries → add to frontier

    Returns (docs_ingested, chunks_added, reasoning_list).
    """
    from ai4saw.agents.fetch_agent import (
        _is_registered, _register_source, _licence_for_source,
    )
    from ai4saw.discovery.discovery import _known_urls

    known_csv = _known_urls()
    _sort_frontier(state)
    batch = state.frontier[:frontier_batch]
    state.frontier = state.frontier[frontier_batch:]

    docs_ingested = 0
    chunks_added = 0
    # Collect (doc, text_excerpt) for reasoning
    ingested_docs: list[tuple[DiscoveredDocument, str]] = []

    # ── Phase 1: drain frontier (HTTP client open only while fetching) ────────
    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "ai4saw/0.1 (research; https://github.com/ai4saw) httpx"},
    ) as client:
        for item in batch:
            if item.url in state.visited_urls or _is_registered(item.url):
                _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore[arg-type]
                continue
            if item.priority < min_relevance and not _is_pdf(item.url) and not _trusted(item.url):
                _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore[arg-type]
                continue

            doc = DiscoveredDocument(
                title=f"Frontier item ({_domain(item.url)})",
                url=item.url, source=item.source,
                date=None, relevance_score=item.priority,
                trigger_entity=item.trigger_entity,
            )
            domain = _domain(item.url)
            logger.info(f"[AgentDiscover] Ingesting: {item.url}")
            if on_event: on_event("info", f"Fetching {domain}…")
            try:
                chunks, filename, text = _ingest_and_capture(
                    doc, geography, client, list(state.initial_entities), on_event
                )
            except Exception as exc:
                logger.warning(f"[AgentDiscover] Skipping {item.url}: {exc}")
                _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore[arg-type]
                if on_event: on_event("error", f"{domain}: {str(exc)[:60]}")
                continue
            _record_visit(state, item.url, item.trigger_entity, chunks)  # type: ignore[arg-type]

            if chunks > 0:
                licence = _licence_for_source(item.source) if item.source in {
                    "openalex", "semanticscholar", "arxiv", "internetarchive"
                } else "web"
                _register_source(filename, item.url, licence, geography, doc.title, item.source)
                docs_ingested += 1
                chunks_added += chunks
                if on_event: on_event("ingest", f"{domain} — {chunks} chunks stored")
                if text:
                    ingested_docs.append((doc, text))
            else:
                state.total_docs_skipped += 1
                if on_event: on_event("skip", f"{domain} — 0 chunks (empty/unreadable)")

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
                            link, state, item.depth + 1,  # type: ignore[arg-type]
                        )
                        _add_to_frontier(state, link, priority, item.trigger_entity, item.source, item.depth + 1)  # type: ignore[arg-type]
    # HTTP client closed — LLM calls happen without holding the connection pool

    # ── Phase 2: LLM reasoning ────────────────────────────────────────────────
    reasoning_results: list[DiscoveryReasoning] = []
    candidates = sorted(ingested_docs, key=lambda x: x[0].relevance_score, reverse=True)
    candidates = candidates[:max_reasoning]
    all_known = list(state.initial_entities) + list(state.discovered_entities.keys())

    for doc, text in candidates:
        if on_event: on_event("reason", f"Reasoning: {doc.title[:55]}")
        reasoning = _llm_reason(
            text=text, doc=doc,
            initial_entities=state.initial_entities,
            geography=geography, known_entities=all_known,
            state=state,
        )
        if not reasoning:
            continue

        reasoning_results.append(reasoning)
        _log_reasoning(reasoning)
        state.docs_reasoned += 1
        if on_reasoning: on_reasoning(reasoning)
        if on_event and reasoning.novel_entities:
            on_event("query", f"Found: {', '.join(reasoning.novel_entities[:3])}")

        # Anchor terms = initial entities + geography + top discovered entities
        # This ensures queries about discovered entities (AIO, Yongbyon, etc.) aren't rejected
        _anchors = (
            state.initial_entities +
            [geography] +
            list(state.discovered_entities.keys())
        )

        for entity in reasoning.novel_entities:
            if not _entity_plausible(entity):
                logger.debug(f"[Guardrail] Rejected entity: {entity!r}")
                continue
            if entity in state.initial_entities:
                continue
            if entity in state.discovered_entities:
                state.discovered_entities[entity]["mention_count"] = (
                    state.discovered_entities[entity].get("mention_count", 1) + 1
                )
            else:
                state.discovered_entities[entity] = {
                    "from_url": doc.url,
                    "timestamp": _now(),
                    "mention_count": 1,
                }
                logger.info(f"[AgentDiscover] Novel entity: {entity!r}")

        for query in reasoning.generated_queries:
            if query in state.executed_queries or query in state.query_queue:
                continue
            if not _query_on_topic(query, _anchors):
                logger.info(f"[Guardrail] Query rejected (off-topic): {query!r}")
                if on_event: on_event("error", f"Guardrail rejected off-topic query: {query[:60]}")
                continue
            state.query_queue.append(query)

    # ── Phase 3: execute novel queries (new HTTP client) ──────────────────────
    queries_this_session = state.query_queue[:5]
    state.query_queue = state.query_queue[5:]
    all_entities = state.initial_entities + list(state.discovered_entities.keys())

    for query in queries_this_session:
        logger.info(f"[AgentDiscover] Executing LLM query: {query!r}")
        if on_event: on_event("query", query[:80])
        new_docs = _execute_novel_query(query, all_entities, state)
        for new_doc in new_docs:
            priority = _frontier_priority(new_doc.relevance_score, new_doc.url, state)  # type: ignore[arg-type]
            _add_to_frontier(state, new_doc.url, priority, new_doc.trigger_entity, new_doc.source)

    _sort_frontier(state)
    if len(state.frontier) > FRONTIER_MAX:
        state.frontier = state.frontier[:FRONTIER_MAX]

    state.session_count += 1
    state.total_docs_ingested += docs_ingested
    state.total_chunks_added += chunks_added
    state.last_run = _now()

    return docs_ingested, chunks_added, reasoning_results


# ── Guardrails ────────────────────────────────────────────────────────────────

def _query_on_topic(query: str, anchor_terms: list[str]) -> bool:
    """Return True if the query contains at least one anchor term (case-insensitive).

    Prevents the LLM from generating queries that drift entirely off the research topic.
    """
    q_lower = query.lower()
    return any(term.lower() in q_lower for term in anchor_terms if len(term) > 2)


def _entity_plausible(entity: str) -> bool:
    """Basic sanity check on a discovered entity name.

    Rejects strings that are clearly not named entities:
    - Too short (single char or empty)
    - Entirely numeric
    - Generic stop-words
    """
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "it", "its", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "not", "no", "nor", "but", "so", "yet",
        "document", "page", "report", "source", "text", "article", "unknown",
    }
    stripped = entity.strip()
    if len(stripped) < 3:
        return False
    if stripped.lower() in _STOPWORDS:
        return False
    if stripped.replace(" ", "").isdigit():
        return False
    return True


# ── LLM-driven seeding ────────────────────────────────────────────────────────

def _llm_generate_seed_queries(
    state: AgentDiscoverState,
    geography: str,
    research_query: str,
    n: int = 8,
) -> list[str]:
    """Ask the LLM to generate targeted DDG search queries for re-seeding.

    Called when drift is detected — replaces generic templates with specific,
    topic-aware queries based on what the agent has discovered so far.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    top_entities = sorted(
        state.discovered_entities.items(),
        key=lambda kv: kv[1].get("mention_count", 1), reverse=True
    )
    entity_list = ", ".join(e for e, _ in top_entities[:10]) or "none yet"
    executed = "; ".join(list(state.executed_queries)[-10:]) or "none"
    scored = sorted(state.domain_scores.items(),
                    key=lambda kv: kv[1].hits, reverse=True)
    good_domains = ", ".join(d for d, s in scored[:5] if s.hits > 0) or "none yet"

    prompt = f"""You are a research intelligence analyst. The system is searching for documents about:

"{research_query}" (geography: {geography})

The last several sessions found nothing relevant — the frontier is full of off-topic content.
You must generate {n} specific, targeted web search queries to find directly relevant documents.

What has been discovered so far:
  Known entities: {entity_list}
  Best domains so far: {good_domains}
  Already executed queries (do not repeat): {executed}

Rules for each query:
  1. MUST contain the conflict location or topic (e.g. "{geography}", "Sarajevo", "Bosnia", "war crimes")
  2. MUST be specific — include entity names, dates, event names, court case IDs where possible
  3. Target primary sources: ICTY judgments, NGO reports, tribunal records, witness testimony
  4. Do NOT use site: operators — let the search engine find the best sources
  5. Do NOT repeat any already-executed query

Output ONLY a JSON array of query strings:
["query 1", "query 2", ...]"""

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="You are a research intelligence analyst generating targeted web search queries."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        queries = json.loads(raw.strip())
        if isinstance(queries, list):
            return [str(q) for q in queries if isinstance(q, str)][:n]
    except Exception as exc:
        logger.warning(f"[LLMSeed] Failed to generate seed queries: {exc}")
    return []


def prune_frontier(state: AgentDiscoverState, anchor_terms: list[str], keep_top: int = 500) -> int:
    """Remove low-relevance frontier items that don't match any anchor term.

    Returns number of items pruned.
    """
    before = len(state.frontier)
    state.frontier = [
        item for item in state.frontier
        if any(t.lower() in (item.url + item.trigger_entity).lower()
               for t in anchor_terms if len(t) > 2)
    ]
    # Also cap to keep_top highest-priority items
    _sort_frontier(state)
    if len(state.frontier) > keep_top:
        state.frontier = state.frontier[:keep_top]
    pruned = before - len(state.frontier)
    if pruned:
        logger.info(f"[Guardrail] Pruned {pruned} off-topic frontier items")
    return pruned


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_agent_summary(state: AgentDiscoverState) -> dict:
    # Sort novel entities by mention_count so the most-corroborated leads surface first
    ranked_entities = sorted(
        state.discovered_entities.items(),
        key=lambda kv: kv[1].get("mention_count", 1),
        reverse=True,
    )
    return {
        "sessions": state.session_count,
        "urls_visited": len(state.visited_urls),
        "frontier_size": len(state.frontier),
        "docs_ingested": state.total_docs_ingested,
        "chunks_added": state.total_chunks_added,
        "docs_reasoned": state.docs_reasoned,
        "novel_entities": len(state.discovered_entities),
        "queries_queued": len(state.query_queue),
        "queries_executed": len(state.executed_queries),
        # Top entities by how often they've been mentioned across documents
        "top_novel_entities": [(e, d.get("mention_count", 1)) for e, d in ranked_entities[:10]],
        "last_run": state.last_run,
    }


def top_novel_entities(state: AgentDiscoverState, n: int = 10) -> list[str]:
    """Return discovered entity names sorted by mention frequency — for re-seeding."""
    ranked = sorted(
        state.discovered_entities.items(),
        key=lambda kv: kv[1].get("mention_count", 1),
        reverse=True,
    )
    return [e for e, _ in ranked[:n]]
