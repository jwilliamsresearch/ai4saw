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
from ai4saw.core.search_graph import (
    g_record_entity,
    g_record_llm_query,
    g_record_url,
    g_mark_ingested,
    g_save,
)
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
  - The document is primarily in a non-English language (skip Korean, Chinese, Russian, Arabic etc.)

When in doubt, INGEST. It is better to include a borderline document than to miss evidence.

Output ONLY valid JSON:
{{
  "ingest": true,
  "reason": "one sentence explaining your decision"
}}"""

_PROMPT = """\
Research: {geography} — topic: {entities}

Current research goals:
{goals}

Document
  Title: {title}
  Source: {source}
---
{text}
---

Already tracking (skip these): {known_entities}

Extract from the document above — prioritise anything that advances the research goals:
1. NEW named entities not in the tracking list (people, orgs, places, events, legal cases)
2. 2-3 specific search queries — each MUST include "{geography}" or topic keywords, and ideally address one of the goals above

Output ONLY valid JSON:
{{
  "novel_entities": ["name1", "name2"],
  "queries": ["query 1", "query 2"],
  "reasoning": "one sentence linking findings to goals"
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

    # LLM memory — persists across restarts
    goals: list[str] = []          # current research goals set by the LLM
    goals_updated_at: str = ""     # timestamp of last goal update

    # Counters
    session_count: int = 0
    docs_reasoned: int = 0
    total_docs_ingested: int = 0
    total_docs_skipped: int = 0
    total_chunks_added: int = 0
    last_run: Optional[str] = None


# ── Persistence ────────────────────────────────────────────────────────────────

def load_agent_state(path: Optional[Path] = None) -> AgentDiscoverState:
    p = path or AGENT_STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            return AgentDiscoverState.model_validate_json(
                p.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning(f"Agent state corrupt, starting fresh: {exc}")
    return AgentDiscoverState()


def save_agent_state(state: AgentDiscoverState, path: Optional[Path] = None) -> None:
    p = path or AGENT_STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _log_reasoning(reasoning: DiscoveryReasoning, log_path: Optional[Path] = None) -> None:
    p = log_path or AGENT_LOG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
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

    # Keep known_entities short — long lists confuse the 7B model
    known_sample = ", ".join(known_entities[:15]) if known_entities else "none"

    goals_text = "\n".join(f"  - {g}" for g in (state.goals if state else [])) or "  (none set yet)"
    prompt = _PROMPT.format(
        geography=geography,
        entities=", ".join(initial_entities[:5]),
        goals=goals_text,
        source=doc.source,
        title=doc.title[:80],
        text=text[:1_500],
        known_entities=known_sample,
    )

    def _extract_json(text: str) -> dict:
        """Aggressively extract JSON from LLM output regardless of wrapping."""
        t = text.strip()
        # Strip code fences
        if "```" in t:
            parts = t.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                try:
                    return json.loads(part.strip())
                except Exception:
                    pass
        # Try raw
        try:
            return json.loads(t)
        except Exception:
            pass
        # Find first { ... } block
        start = t.find("{")
        end   = t.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(t[start:end])
        raise ValueError(f"No JSON found in: {t[:100]!r}")

    import pathlib as _pl
    _reason_log = _pl.Path("output/reasoning_debug.log")
    _reason_log.parent.mkdir(parents=True, exist_ok=True)

    last_err = ""
    raw = ""
    data: dict = {}
    for _attempt in range(2):
        try:
            llm = get_llm()
            response = llm.invoke([
                SystemMessage(content=_make_system(initial_entities, geography)),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            # Log every model response so we can see what it's returning
            _reason_log.open("a", encoding="utf-8").write(
                f"\n--- attempt {_attempt+1} for {doc.url[:60]} ---\n{raw}\n"
            )
            data = _extract_json(raw)
            break
        except Exception as exc:
            last_err = str(exc)
            _reason_log.open("a", encoding="utf-8").write(
                f"\n--- PARSE FAIL attempt {_attempt+1}: {exc} ---\nraw={raw[:200]}\n"
            )
            if _attempt == 0:
                time.sleep(1)
    else:
        # Return empty result rather than None so on_reasoning always fires
        data = {"novel_entities": [], "queries": [], "reasoning": f"parse failed: {last_err[:80]}"}

    try:
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
    from ai4saw.core.providers import get_prescreen_llm

    prompt = _PRESCREEN_PROMPT.format(
        geography=geography,
        entities=", ".join(initial_entities[:10]),
        url=doc.url,
        title=doc.title[:120],
        source=doc.source,
        text=text[:1_500],
    )

    try:
        llm = get_prescreen_llm()
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
            doc = DiscoveredDocument(
                title=title, url=url, source="duckduckgo",
                date=None, relevance_score=relevance,
                trigger_entity=trigger, snippet=snippet,
            )
            g_record_url(url, title, "duckduckgo", trigger_query=query, query_type="llm_query")
            docs.append(doc)
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
            result = discover_for_entities(api_entities, per_entity_limit=per_entity_limit, on_event=on_event)
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
            g_record_entity(entity, doc.url)
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
            # Record LLM query in graph — link to entities that triggered it
            for ent in reasoning.novel_entities:
                if _entity_plausible(ent):
                    g_record_llm_query(query, triggered_by_entity=ent, generated_from_url=doc.url)
                    break
            else:
                g_record_llm_query(query, generated_from_url=doc.url)
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


# ── Pipeline session (parallel fetch → prescreen → embed) ─────────────────────

_NUM_FETCHERS   = 4   # concurrent HTTP fetchers
_QUEUE_MAXSIZE  = 40  # max items buffered between stages

# Domains that never contain research-quality documents
_BLOCKED_DOMAINS = frozenset({
    "dictionary.com", "merriam-webster.com", "collinsdictionary.com",
    "thesaurus.com", "oxfordlearnersdictionaries.co.uk", "oxforddictionaries.com",
    "vocabulary.com", "macmillandictionary.com", "cambridge.org/dictionary",
    "en.wiktionary.org", "lexico.com",
    "reddit.com", "twitter.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "pinterest.com",
    "amazon.com", "ebay.com", "etsy.com",
    "tripadvisor.com", "yelp.com", "booking.com",
})


def run_agent_session_pipeline(
    state: AgentDiscoverState,
    geography: str,
    frontier_batch: int = 20,
    min_relevance: float = 0.4,
    max_reasoning: int = MAX_REASONING_PER_SESSION,
    on_event: Optional[Callable[[str, str], None]] = None,
    on_reasoning: Optional[Callable[["DiscoveryReasoning"], None]] = None,
    seeding_done: Optional[threading.Event] = None,
) -> tuple[int, int, list[DiscoveryReasoning]]:
    """Pipeline version of run_agent_session.

    Stages run concurrently:
      4× fetcher threads  — HTTP fetch + load document  → prescreen_q
      1× prescreen thread — LLM prescreen               → embed_q
      1× embed thread     — chunk + ChromaDB write

    The main thread drives Phase 2 (LLM reasoning) and Phase 3 (novel queries)
    after all pipeline workers finish.
    """
    import socket as _socket
    _socket.setdefaulttimeout(25)

    # Patch requests (used by langchain WebBaseLoader) to enforce connect+read timeouts
    try:
        import requests as _req
        _orig_req = _req.Session.request
        def _req_with_timeout(self, method, url, **kw):
            kw.setdefault("timeout", (10, 20))
            return _orig_req(self, method, url, **kw)
        _req.Session.request = _req_with_timeout
    except Exception:
        pass

    # Patch Ollama httpx client to enforce timeouts on LLM/embed calls
    try:
        import httpx as _httpx
        _orig_init = _httpx.Client.__init__
        def _httpx_init_with_timeout(self, *a, **kw):
            kw.setdefault("timeout", _httpx.Timeout(30.0))
            _orig_init(self, *a, **kw)
        _httpx.Client.__init__ = _httpx_init_with_timeout
    except Exception:
        pass

    from ai4saw.agents.fetch_agent import (
        _is_registered, _register_source, _licence_for_source,
        _detect_content_type, _download_pdf, _safe_filename, CORPUS_DIR,
    )
    from ai4saw.ingestion.chunker import chunk_documents
    from ai4saw.ingestion.embedder import embed_and_store
    from ai4saw.ingestion.loaders import load_document
    from ai4saw.discovery.discovery import _known_urls
    from datetime import date as _date

    # ── Shared state / locks ──────────────────────────────────────────────────
    state_lock      = threading.Lock()   # guards state.visited_urls and counters
    chroma_lock     = threading.Lock()   # guards ChromaDB writes
    _active_count   = [0]                # current concurrent fetches (always decremented in finally)
    _host_sems: dict[str, threading.Semaphore] = {}  # max 2 connections per host
    _host_sems_lock = threading.Lock()

    docs_ingested = 0
    chunks_added  = 0
    ingested_docs: list[tuple[DiscoveredDocument, str]] = []
    results_lock  = threading.Lock()

    # ── Pipeline queues ───────────────────────────────────────────────────────
    loaded_q   = queue.Queue(maxsize=_QUEUE_MAXSIZE * 2)
    embed_q    = queue.Queue(maxsize=0)
    _SENTINEL  = object()

    # If there are pending queries, we should be seeding, not just draining.
    if state.query_queue:
        if on_event: on_event("info", "Seeding from LLM-generated queries…")
        queries_to_run = state.query_queue[:5]
        state.query_queue = state.query_queue[5:]
        all_entities = state.initial_entities + list(state.discovered_entities.keys())
        for query in queries_to_run:
            if on_event: on_event("query", query[:80])
            new_docs = _execute_novel_query(query, all_entities, state)
            for new_doc in new_docs:
                priority = _frontier_priority(new_doc.relevance_score, new_doc.url, state)  # type: ignore[arg-type]
                _add_to_frontier(state, new_doc.url, priority, new_doc.trigger_entity, new_doc.source)

    # seeding_done: set by caller (cli.py) when all seeding threads finish
    if seeding_done is None:
        seeding_done = threading.Event()
        seeding_done.set()

    # Session milestone every 100 classified docs (ingested OR skipped)
    _session_milestone = 100
    _classified_count  = [0]
    _session_milestone_lock = threading.Lock()

    # Reason every 3 ingested docs — fires asynchronously, never blocks the pipeline
    _reason_every      = 3
    _since_last_reason = [0]
    _reasoning_lock    = threading.Lock()
    # No shared lock needed — each operation uses its own model in VRAM simultaneously:
    # prescreen → qwen2.5:0.5b (fast), reasoning → qwen2.5:7b (deep), embed → nomic-embed-text

    def _trigger_reasoning_if_due() -> None:
        with _reasoning_lock:
            _since_last_reason[0] += 1
            due = _since_last_reason[0] >= _reason_every
            if due:
                _since_last_reason[0] = 0
        if not due:
            return
        def _do_reason() -> None:
            with results_lock:
                candidates = sorted(ingested_docs[-_reason_every:],
                                    key=lambda x: x[0].relevance_score, reverse=True)
            if not candidates:
                return
            all_known = list(state.initial_entities) + list(state.discovered_entities.keys())
            anchors   = all_known[:]
            for doc, text in candidates:
                if on_event: on_event("reason", f"Reasoning: {doc.title[:55]}")
                try:
                    if on_event: on_event("model_reason", "running")
                    r = _llm_reason(
                            text=text, doc=doc,
                                        initial_entities=state.initial_entities,
                                        geography=geography, known_entities=all_known,
                                        state=state)
                except Exception as exc:
                    if on_event: on_event("model_reason", "idle")
                    if on_event: on_event("error", f"Reasoning exception: {str(exc)[:80]}")
                    continue
                if not r:
                    if on_event: on_event("error", f"Reasoning parse failed — check model JSON output")
                    continue
                # Update state FIRST so on_reasoning sees populated entities
                with state_lock:
                    for ent in r.novel_entities:
                        if _entity_plausible(ent) and ent not in state.initial_entities:
                            g_record_entity(ent, doc.url)
                            state.discovered_entities.setdefault(ent, {
                                "from_url": doc.url, "timestamp": _now(), "mention_count": 0
                            })["mention_count"] = state.discovered_entities.get(ent, {}).get("mention_count", 0) + 1
                    for q in r.generated_queries:
                        if q not in state.executed_queries and q not in state.query_queue:
                            if _query_on_topic(q, anchors):
                                # Link query to first plausible entity that triggered it
                                trigger_ent = next(
                                    (e for e in r.novel_entities if _entity_plausible(e)), None
                                )
                                g_record_llm_query(q, triggered_by_entity=trigger_ent, generated_from_url=doc.url)
                                state.query_queue.append(q)
                                if on_event: on_event("query", q[:80])

                if on_event: on_event("model_reason", "idle")
                if on_event and r.novel_entities:
                    on_event("query", f"Found: {', '.join(r.novel_entities[:3])}")

                # Now call on_reasoning — state.discovered_entities is already updated
                if on_reasoning:
                    try:
                        on_reasoning(r)
                    except Exception as _ore:
                        if on_event: on_event("error", f"on_reasoning callback failed: {_ore}")

                try:
                    reasoning_results.append(r)
                    _log_reasoning(r)
                    state.docs_reasoned += 1
                except Exception:
                    pass
        threading.Thread(target=_do_reason, daemon=True).start()

    # ── Stage 1: Fetcher workers — stream directly from shared frontier ───────
    def _fetch_worker() -> None:
        with httpx.Client(
            timeout=20.0, follow_redirects=True,
            headers={"User-Agent": "ai4saw/0.1 (research; https://github.com/ai4saw) httpx"},
        ) as client:
            while True:
                # Pull next item — prefer a domain not already at its semaphore limit
                item = None
                with state_lock:
                    if state.frontier:
                        _sort_frontier(state)
                        # Prefer domains not saturated AND not in blocklist
                        chosen_idx = 0
                        for _idx, _candidate in enumerate(state.frontier[:20]):
                            _h = _domain(_candidate.url)
                            if _h in _BLOCKED_DOMAINS:
                                continue  # skip blocked domains entirely
                            _sem = _host_sems.get(_h)
                            if _sem is None or _sem._value > 0:  # type: ignore
                                chosen_idx = _idx
                                break
                        item = state.frontier.pop(chosen_idx)
                        # Skip if blocked
                        if _domain(item.url) in _BLOCKED_DOMAINS:
                            _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore
                            item = None

                if item is None:
                    if seeding_done.is_set():
                        break  # seeding finished and frontier empty — done
                    time.sleep(0.1)  # wait for seeding to add more
                    continue

                with state_lock:
                    already = item.url in state.visited_urls or _is_registered(item.url)
                if already:
                    with state_lock:
                        _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore
                    with results_lock:
                        docs_processed[0] -= 1  # didn't actually process
                    continue

                if item.priority < min_relevance and not _is_pdf(item.url) and not _trusted(item.url):
                    with state_lock:
                        _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore
                    continue

                host = _domain(item.url)
                with _host_sems_lock:
                    if host not in _host_sems:
                        _host_sems[host] = threading.Semaphore(3)  # max 3 per host
                host_sem = _host_sems[host]
                host_sem.acquire()

                with state_lock:
                    _active_count[0] += 1
                    n = _active_count[0]
                if on_event: on_event("info", f"[{n} parallel] Fetching {host}…")
                doc = DiscoveredDocument(
                    title=f"Frontier item ({_domain(item.url)})",
                    url=item.url, source=item.source, date=None,
                    relevance_score=item.priority, trigger_entity=item.trigger_entity,
                )

                try:
                    content_type = _detect_content_type(doc.url, client)
                    is_pdf = "pdf" in content_type or doc.url.lower().endswith(".pdf")
                    filename = _safe_filename(doc.url, doc.title, doc.source)
                    if is_pdf:
                        dest = CORPUS_DIR / filename
                        if not _download_pdf(doc.url, dest, client):
                            with state_lock:
                                _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore
                            continue
                        source_arg: str = str(dest)
                        source_url_arg: Optional[str] = doc.url
                    else:
                        source_arg = doc.url
                        source_url_arg = doc.url

                    raw_docs = load_document(
                        source=source_arg, doc_type="report", language="en",
                        date_published=None, geography=geography,
                        source_url=source_url_arg,
                    )
                    text_excerpt = _sample_text(raw_docs, total_chars=2_000, slices=5)

                    # Extract links from HTML while we still have the client
                    if not is_pdf:
                        html = _fetch_html(doc.url, client)
                        if html:
                            with state_lock:
                                for link in _extract_links(html, doc.url):
                                    if link in state.visited_urls:
                                        continue
                                    if not (_is_pdf(link) or _trusted(link)):
                                        continue
                                    priority = _frontier_priority(
                                        _relevance(item.trigger_entity, link, base=0.3),
                                        link, state, item.depth + 1,  # type: ignore
                                    )
                                    _add_to_frontier(state, link, priority, item.trigger_entity,
                                                     item.source, item.depth + 1)  # type: ignore

                    loaded_q.put((item, doc, raw_docs, text_excerpt, filename, is_pdf))
                except Exception as exc:
                    if on_event: on_event("error", f"{host}: {str(exc)[:60]}")
                    with state_lock:
                        _record_visit(state, item.url, item.trigger_entity, 0)  # type: ignore
                finally:
                    # Always release — keeps counter accurate and host semaphore free
                    with state_lock:
                        _active_count[0] -= 1
                    host_sem.release()

    # ── Stage 2: Prescreen worker (single — Ollama serialised) ───────────────
    def _prescreen_worker() -> None:
        while True:
            item_tuple = loaded_q.get()
            if item_tuple is _SENTINEL:
                embed_q.put(_SENTINEL)
                break
            item, doc, raw_docs, text_excerpt, filename, is_pdf = item_tuple
            domain = _domain(item.url)

            try:
                if on_event: on_event("model_prescreen", "running")
                should_ingest, reason = _llm_prescreen(
                    text_excerpt, doc, state.initial_entities, geography
                )
            except Exception:
                should_ingest, reason = True, ""
            finally:
                if on_event: on_event("model_prescreen", "idle")
            with state_lock:
                _record_visit(state, item.url, item.trigger_entity,
                              1 if should_ingest else 0)  # type: ignore

            if not should_ingest:
                if on_event: on_event("skip", f"{domain} — {reason[:70]}")
                if is_pdf:
                    dest = CORPUS_DIR / filename
                    if dest.exists():
                        dest.unlink(missing_ok=True)
                with state_lock:
                    state.total_docs_skipped += 1
            else:
                embed_q.put((item, doc, raw_docs, filename, text_excerpt))

            # Session milestone every 100 classified docs
            with _session_milestone_lock:
                _classified_count[0] += 1
                milestone = _classified_count[0] % _session_milestone == 0
            if milestone:
                state.session_count += 1
                save_agent_state(state)
                if on_event: on_event("info", f"── Session milestone: {state.session_count} ({_classified_count[0]} classified) ──")

    # ── Stage 3: Embed worker (single — ChromaDB lock) ───────────────────────
    def _embed_worker() -> None:
        nonlocal docs_ingested, chunks_added
        while True:
            item_tuple = embed_q.get()
            if item_tuple is _SENTINEL:
                break
            item, doc, raw_docs, filename, text_excerpt = item_tuple
            domain = _domain(item.url)

            try:
                cks = chunk_documents(raw_docs)
                cks = cks[:20]
                if on_event: on_event("model_embed", f"running {len(cks)} chunks")
                embed_and_store(cks)
                n = len(cks)
                licence = (_licence_for_source(item.source)
                           if item.source in {"openalex", "semanticscholar", "arxiv", "internetarchive"}
                           else "web")
                _register_source(filename, item.url, licence, geography, doc.title, item.source)
                g_mark_ingested(item.url)
                with results_lock:
                    docs_ingested += 1
                    chunks_added  += n
                    if text_excerpt:
                        ingested_docs.append((doc, text_excerpt))
                if on_event: on_event("model_embed", "idle")
                if on_event: on_event("ingest", f"{domain} — {n} chunks stored")
                _trigger_reasoning_if_due()
                # Save state and graph every 5 ingested docs so they survive Ctrl-C
                if docs_ingested % 5 == 0:
                    try:
                        save_agent_state(state)
                        g_save()
                    except Exception:
                        pass
            except Exception as exc:
                if on_event: on_event("error", f"{domain} embed failed: {str(exc)[:60]}")

    # ── Launch pipeline ───────────────────────────────────────────────────────
    # Fetchers stream from frontier directly — no pre-loaded batch needed
    fetchers    = [threading.Thread(target=_fetch_worker,    daemon=True)
                   for _ in range(_NUM_FETCHERS)]
    prescreener = threading.Thread(target=_prescreen_worker, daemon=True)
    embedder    = threading.Thread(target=_embed_worker,     daemon=True)

    for t in fetchers:
        t.start()
    prescreener.start()
    embedder.start()

    for t in fetchers:
        t.join()
    loaded_q.put(_SENTINEL)
    prescreener.join()
    embedder.join()

    # Reasoning fired continuously during embedding — just collect results here

    _sort_frontier(state)
    if len(state.frontier) > FRONTIER_MAX:
        state.frontier = state.frontier[:FRONTIER_MAX]

    state.session_count          += 1
    state.total_docs_ingested    += docs_ingested
    state.total_chunks_added     += chunks_added
    state.last_run                = _now()

    return docs_ingested, chunks_added, reasoning_results


# ── Guardrails ────────────────────────────────────────────────────────────────

def _query_on_topic(query: str, anchor_terms: list[str]) -> bool:
    """Return True if the query contains at least one anchor term AND is not obviously
    off-topic (e.g. pure healthcare/dictionary/lifestyle content).

    Prevents the LLM from generating queries that drift off the research topic.
    """
    q_lower = query.lower()
    if not any(term.lower() in q_lower for term in anchor_terms if len(term) > 2):
        return False
    # Reject queries that are clearly off-topic regardless of anchor presence
    _OFF_TOPIC = {
        "patient care", "hospital", "clinical trial", "medication", "drug dosage",
        "dictionary", "definition of", "synonym", "thesaurus", "vocabulary",
        "recipe", "cooking", "restaurant", "travel", "tourism",
        "stock market", "cryptocurrency", "investment portfolio",
        "sports", "football", "basketball", "celebrity",
    }
    return not any(bad in q_lower for bad in _OFF_TOPIC)


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


# ── LLM goal-setting ─────────────────────────────────────────────────────────

def llm_set_goals(
    state: AgentDiscoverState,
    geography: str,
    research_query: str,
    narrator_text: str = "",
) -> list[str]:
    """Ask the LLM to set 3-5 specific research goals based on current findings.

    Goals persist in agent state across restarts and guide future searching.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    top_entities = sorted(
        state.discovered_entities.items(),
        key=lambda kv: kv[1].get("mention_count", 1), reverse=True
    )
    entity_list = ", ".join(e for e, _ in top_entities[:10]) or "none yet"
    docs_in = state.total_docs_ingested
    current_goals = "\n".join(f"  - {g}" for g in state.goals) or "  none yet"

    executed = "; ".join(list(state.executed_queries)[-15:]) or "none"

    prompt = f"""You are a research intelligence analyst. You are building an evidence corpus on:
"{research_query}" (geography: {geography})

Current state:
- Documents ingested: {docs_in}
- Key entities discovered: {entity_list}
- Recent research summary: {narrator_text[:300] or "not yet available"}
- Current goals:
{current_goals}
- Already searched (do not repeat): {executed}

1. Set 3-5 specific, actionable research goals — concrete evidence targets (cases, people, documents, questions).
2. For EACH goal, generate 5-6 targeted search queries that would find evidence for it.
   Every query MUST include "{geography}" or a topic keyword. No site: operators.

Output ONLY valid JSON:
{{
  "goals": ["goal 1", "goal 2", "goal 3"],
  "queries": {{
    "goal 1": ["query a", "query b", "query c", "query d", "query e"],
    "goal 2": ["query a", "query b", "query c", "query d", "query e"]
  }}
}}"""

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=f"You are a research intelligence analyst studying: {research_query}"),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        goals = [str(g) for g in data.get("goals", [])[:5] if isinstance(g, str)]
        # Add all goal queries to the queue
        anchors = state.initial_entities + [geography]
        queries_added = 0
        for goal_queries in data.get("queries", {}).values():
            for q in goal_queries[:6]:
                q = str(q)
                if q not in state.executed_queries and q not in state.query_queue:
                    if _query_on_topic(q, anchors):
                        state.query_queue.append(q)
                        queries_added += 1
        if goals:
            logger.info(f"[Goals] Set {len(goals)} goals, queued {queries_added} queries")
            return goals
    except Exception as exc:
        logger.warning(f"Goal-setting failed: {exc}")
    return state.goals  # keep existing goals on failure


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

    goals_text = "\n".join(f"  - {g}" for g in state.goals) if state.goals else "  (none set yet)"

    prompt = f"""You are a research intelligence analyst. The system is searching for documents about:

"{research_query}" (geography: {geography})

Current research goals:
{goals_text}

What has been discovered so far:
  Known entities: {entity_list}
  Best domains so far: {good_domains}
  Already executed queries (do not repeat): {executed}

Generate {n} specific search queries that directly advance the research goals above.

Rules:
  1. MUST contain the topic or geography keyword
  2. MUST be specific — include names, dates, case IDs where possible
  3. Prioritise queries that address the stated goals
  4. Do NOT use site: operators
  5. Do NOT repeat already-executed queries

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
