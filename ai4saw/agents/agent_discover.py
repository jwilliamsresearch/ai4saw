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
  Initial entity: "Srebrenica"
  Ingests: ICTY judgment on Krstić
  LLM extracts: Dražen Erdemović, 10th Sabotage Detachment, Branjevo Farm
  LLM generates:
    "Dražen Erdemović plea agreement ICTY 1996"
    "10th Sabotage Detachment VRS executions July 1995"
    "Branjevo Farm massacre witnesses"
  → 3 novel, specific queries that no template would have produced

State:   output/agent_discover_state.json
Log:     output/agent_discover_log.jsonl  (one entry per document reasoned on)
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

_SYSTEM = (
    "You are a research intelligence analyst specialising in conflict-related "
    "slavery, war crimes, and human rights violations. You are building an "
    "evidence corpus for academic research. Your job is to read document "
    "excerpts and identify high-value leads for further investigation."
)

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

Based on the excerpt above:

1. NEW named entities — people, military units, locations, events, legal instruments
   that appear in this text but are NOT in the already-tracked list.

2. Specific search queries (2–3) that would find corroborating or extending evidence.
   Be precise: include names, dates, unit numbers, locations where possible.
   Good: "Dražen Erdemović plea agreement ICTY 1996"
   Bad:  "war crimes Bosnia"

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
    discovered_entities: dict[str, dict] = {}   # entity -> {from_url, timestamp}
    query_queue: list[str] = []                  # LLM-generated queries pending execution
    executed_queries: list[str] = []             # queries already run (dedup)

    # Counters
    session_count: int = 0
    docs_reasoned: int = 0
    total_docs_ingested: int = 0
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
) -> Optional[DiscoveryReasoning]:
    """Call the LLM to reason about what to search for next.

    Returns None on parse failure — callers should handle gracefully.
    """
    if not text.strip():
        return None

    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    known_sample = ", ".join(known_entities[:30]) if known_entities else "none yet"
    prompt = _PROMPT.format(
        geography=geography,
        entities=", ".join(initial_entities),
        source=doc.source,
        trigger_entity=doc.trigger_entity,
        title=doc.title[:120],
        text=text[:2_000],
        known_entities=known_sample,
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
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


# ── Ingest and capture text ────────────────────────────────────────────────────

def _ingest_and_capture(
    doc: DiscoveredDocument,
    geography: str,
    client: httpx.Client,
) -> tuple[int, str, str]:
    """Ingest one document; return (chunks_added, filename, text_excerpt).

    text_excerpt is the first 2 000 chars of raw document text — used for LLM
    reasoning.  Empty string on failure.
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

    # Capture text excerpt before chunking (for LLM reasoning)
    text_excerpt = " ".join(d.page_content for d in raw_docs[:3])[:2_000]

    chunks = chunk_documents(raw_docs)
    embed_and_store(chunks)
    return len(chunks), filename, text_excerpt


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
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed")
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
        state.executed_queries.append(query)
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
) -> int:
    """Run web_discover to seed the frontier. Returns new URL count."""
    from ai4saw.agents.web_agent import web_discover

    _, state_out = web_discover(
        entities=entities,
        per_entity_limit=per_entity_limit,
        contact_email=contact_email,
        state=AgentDiscoverState(  # type: ignore[arg-type]  — duck-typed
            visited_urls=state.visited_urls,
            frontier=state.frontier,
            domain_scores=state.domain_scores,
            query_stats=state.query_stats,
        ),
    )
    # Merge updates back into our state
    state.visited_urls = state_out.visited_urls  # type: ignore[assignment]
    state.frontier = state_out.frontier  # type: ignore[assignment]
    state.domain_scores = state_out.domain_scores  # type: ignore[assignment]
    state.query_stats = state_out.query_stats  # type: ignore[assignment]
    return len(state.frontier)


# ── Core agent session ────────────────────────────────────────────────────────

def run_agent_session(
    state: AgentDiscoverState,
    geography: str,
    frontier_batch: int = 20,
    min_relevance: float = 0.4,
    max_reasoning: int = MAX_REASONING_PER_SESSION,
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

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        # ── Phase 1: drain frontier ───────────────────────────────────────────
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
            logger.info(f"[AgentDiscover] Ingesting: {item.url}")
            chunks, filename, text = _ingest_and_capture(doc, geography, client)
            _record_visit(state, item.url, item.trigger_entity, chunks)  # type: ignore[arg-type]

            if chunks > 0:
                licence = _licence_for_source(item.source) if item.source in {
                    "openalex", "semanticscholar", "arxiv", "internetarchive"
                } else "web"
                _register_source(filename, item.url, licence, geography, doc.title, item.source)
                docs_ingested += 1
                chunks_added += chunks
                if text:
                    ingested_docs.append((doc, text))

            # Follow links from HTML pages
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

        # ── Phase 2: LLM reasoning ────────────────────────────────────────────
        reasoning_results: list[DiscoveryReasoning] = []

        # Pick highest-relevance docs to reason on
        candidates = sorted(ingested_docs, key=lambda x: x[0].relevance_score, reverse=True)
        candidates = candidates[:max_reasoning]

        all_known = list(state.initial_entities) + list(state.discovered_entities.keys())

        for doc, text in candidates:
            reasoning = _llm_reason(
                text=text,
                doc=doc,
                initial_entities=state.initial_entities,
                geography=geography,
                known_entities=all_known,
            )
            if not reasoning:
                continue

            reasoning_results.append(reasoning)
            _log_reasoning(reasoning)
            state.docs_reasoned += 1

            # Register novel entities
            for entity in reasoning.novel_entities:
                if entity not in state.discovered_entities and entity not in state.initial_entities:
                    state.discovered_entities[entity] = {
                        "from_url": doc.url,
                        "timestamp": _now(),
                    }
                    logger.info(f"[AgentDiscover] Novel entity discovered: {entity!r}")

            # Queue novel queries
            for query in reasoning.generated_queries:
                if query not in state.executed_queries and query not in state.query_queue:
                    state.query_queue.append(query)

        # ── Phase 3: execute novel LLM-generated queries ──────────────────────
        queries_this_session = state.query_queue[:5]
        state.query_queue = state.query_queue[5:]

        all_entities = state.initial_entities + list(state.discovered_entities.keys())

        for query in queries_this_session:
            logger.info(f"[AgentDiscover] Executing LLM query: {query!r}")
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


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_agent_summary(state: AgentDiscoverState) -> dict:
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
        "top_novel_entities": list(state.discovered_entities.keys())[:10],
        "last_run": state.last_run,
    }
