"""AI4SAW MCP Server — exposes the pipeline as a Model Context Protocol server.

This makes the entire corpus + knowledge graph available as a first-class tool
inside Claude Desktop, Claude.ai, and any other MCP-compatible AI client.

A researcher can open Claude Desktop and ask:
  "What happened in Location Beta in April 2023?"
and Claude will call search_corpus and query_knowledge_graph directly,
returning a cited answer grounded in the indexed documents.

Tools exposed:
  search_corpus          — semantic search over ChromaDB
  query_knowledge_graph  — temporal-aware k-hop graph traversal
  find_entity            — resolved entity lookup with aliases
  ask_question           — full RAG Q&A with reranking
  get_corpus_stats       — coverage overview (geography, dates, doc types)

Setup (Claude Desktop):
  Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "ai4saw": {
        "command": "uv",
        "args": ["run", "ai4saw-mcp"],
        "cwd": "/absolute/path/to/ai4saw"
      }
    }
  }

Run standalone for testing:
  uv run ai4saw-mcp
  uv run python -m ai4saw.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ai4saw",
    instructions=(
        "You are an expert conflict and human rights research analyst. "
        "You have access to an indexed corpus of conflict documents and a structured "
        "knowledge graph of actors, events, and relations. "
        "Use search_corpus for open-ended questions, query_knowledge_graph for "
        "specific actor/event relations, and find_entity to resolve abbreviations. "
        "Always cite your sources. Be precise about dates, locations, and actor names. "
        "If evidence is insufficient, say so explicitly rather than speculating."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_corpus(query: str, top_k: int = 6, top_n: int = 3) -> str:
    """Search the indexed conflict document corpus for relevant passages.

    Returns the top passages ranked by semantic similarity and re-ranked by
    a cross-encoder. Each result includes source filename, geography, and date.

    Use this for: open-ended thematic questions, specific event lookups,
    finding what documents say about a topic.

    Args:
        query: Natural language search query.
        top_k: How many passages to retrieve before re-ranking.
        top_n: How many passages to return after re-ranking.
    """
    try:
        from ai4saw.ingestion.embedder import get_vector_store
        from ai4saw.retrieval.reranker import rerank

        store = get_vector_store()
        docs = store.similarity_search(query, k=top_k)
        reranked = rerank(query, docs, top_n=top_n)

        if not reranked:
            return "No relevant passages found in the corpus for this query."

        parts = []
        for i, doc in enumerate(reranked, 1):
            m = doc.metadata
            source = m.get("source_filename", "unknown")
            geo = m.get("geography") or ""
            date = m.get("date_published") or ""
            doc_type = m.get("doc_type", "")
            meta = " | ".join(x for x in [geo, date, doc_type] if x)
            parts.append(
                f"[Passage {i}] {source}{' (' + meta + ')' if meta else ''}\n"
                f"{doc.page_content[:700]}"
            )

        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        logger.error(f"MCP search_corpus failed: {exc}")
        return f"Search failed: {exc}. Ensure the corpus has been ingested with `ai4saw ingest`."


@mcp.tool()
def query_knowledge_graph(
    entity_name: str,
    hops: int = 2,
    at_date: str = "",
) -> str:
    """Get structured relations for a named entity from the knowledge graph.

    Returns the entity's neighbourhood: everything it did, everything done to
    it, and the entities it is directly connected to — with evidence and dates.

    Use this for: questions about specific actors, command structures,
    "who ordered X", "what did Y do to Z", temporal command chain queries.

    Args:
        entity_name: Name of the actor, organisation, location, or event to look up.
        hops: Neighbourhood depth (1 = direct relations, 2 = friends-of-friends).
        at_date: ISO 8601 date (e.g. "YYYY-MM-DD"). When set, only relations
            valid on that date are shown — enabling temporal graph queries.
    """
    try:
        from ai4saw.retrieval.graph_rag import graph_context_for_query

        result = graph_context_for_query(
            entity_name,
            hops=hops,
            at_date=at_date if at_date else None,
        )
        if not result:
            return (
                f"No graph entries found for '{entity_name}'. "
                "Either the entity was not extracted or the knowledge graph has not been built. "
                "Run `ai4saw extract pipeline`, `ai4saw extract resolve`, then `ai4saw graph build`."
            )
        return result

    except Exception as exc:
        logger.error(f"MCP query_knowledge_graph failed: {exc}")
        return f"Graph query failed: {exc}"


@mcp.tool()
def find_entity(name: str) -> str:
    """Look up a named entity in the resolved entity registry.

    Returns canonical name, all known aliases, occurrence count across the
    corpus, and which source chunks mention this entity.

    Use this to: resolve abbreviations ("Armed-Group-Beta" → "Rapid Support Forces"),
    check if an entity is in the corpus, find all surface forms of an actor.

    Args:
        name: Full or partial name of the entity to look up.
    """
    registry_path = Path("data/entity_registry.json")
    if not registry_path.exists():
        return (
            "Entity registry not found. "
            "Run `ai4saw extract pipeline` then `ai4saw extract resolve`."
        )

    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        entities = data.get("entities", [])
        name_lower = name.lower().strip()

        matches = []
        for ent in entities:
            canonical = ent.get("canonical_text", "").lower()
            aliases = [a.lower() for a in ent.get("aliases", [])]
            if name_lower in canonical or any(name_lower in a for a in aliases):
                matches.append(ent)

        if not matches:
            return (
                f"No entity matching '{name}' found in the registry. "
                "The entity may not appear in the corpus, or extraction may not have run yet."
            )

        parts = []
        for m in matches[:5]:
            aliases_str = ", ".join(m.get("aliases", [])[:8]) or "none"
            parts.append(
                f"[{m['label']}] {m['canonical_text']}\n"
                f"  Aliases: {aliases_str}\n"
                f"  Occurrences: {m['occurrence_count']}\n"
                f"  Appears in: {len(m.get('source_chunks', []))} chunk(s)\n"
                f"  Mean confidence: {m.get('mean_confidence', 0):.2f}"
            )
        return "\n\n".join(parts)

    except Exception as exc:
        logger.error(f"MCP find_entity failed: {exc}")
        return f"Entity lookup failed: {exc}"


@mcp.tool()
def ask_question(question: str, top_k: int = 8, top_n: int = 3) -> str:
    """Answer a natural language question using full RAG: MMR retrieval + reranking + LLM.

    Returns a cited answer generated by the configured LLM, grounded in the
    corpus. Slower than search_corpus but produces a synthesised answer rather
    than raw passages.

    Use this for: questions requiring synthesis across multiple sources,
    "what does the corpus say about X overall", summary questions.

    Args:
        question: The research question to answer.
        top_k: Chunks to retrieve before reranking.
        top_n: Chunks to pass to the LLM after reranking.
    """
    try:
        from ai4saw.retrieval.qa import answer

        response = answer(question, top_k=top_k, top_n=top_n)

        source_lines = "\n".join(
            f"  [{i}] {s.source_filename}"
            f"{' | ' + (s.geography or '') if s.geography else ''}"
            f"{' | ' + str(s.date_published) if s.date_published else ''}"
            for i, s in enumerate(response.sources, 1)
        )

        return (
            f"{response.answer}\n\n"
            f"Sources:\n{source_lines}\n\n"
            f"[Retrieved {response.retrieved_chunks} chunks, "
            f"reranked to {response.reranked_to}, "
            f"confidence {response.confidence:.2f}]"
        )

    except Exception as exc:
        logger.error(f"MCP ask_question failed: {exc}")
        return f"Q&A failed: {exc}. Ensure the corpus is ingested and the LLM provider is configured."


@mcp.tool()
def get_corpus_stats() -> str:
    """Return an overview of the indexed corpus: document count, coverage by geography and date.

    Use this to understand what the corpus covers before querying it.
    """
    stats_path = Path("output/corpus_stats.json")
    if not stats_path.exists():
        # Fall back to live ChromaDB count
        try:
            from ai4saw.ingestion.embedder import get_vector_store
            store = get_vector_store()
            count = store._collection.count()
            return f"Corpus index: {count} chunks in ChromaDB. Run `ai4saw export all` for full stats."
        except Exception:
            return "Corpus stats not available. Run `ai4saw ingest` and `ai4saw export all`."

    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        geo = data.get("coverage_by_geography", {})
        dates = data.get("coverage_by_date", {})
        doc_types = data.get("doc_types", {})

        geo_str = ", ".join(f"{k}: {v}" for k, v in list(geo.items())[:8])
        date_str = ", ".join(f"{k}: {v}" for k, v in sorted(dates.items())[:10])
        type_str = ", ".join(f"{k}: {v}" for k, v in doc_types.items())

        return (
            f"Corpus overview:\n"
            f"  Documents: {data.get('document_count', '?')}\n"
            f"  Chunks:    {data.get('chunk_count', '?')}\n"
            f"  Geography: {geo_str}\n"
            f"  Date range: {date_str}\n"
            f"  Types: {type_str}"
        )
    except Exception as exc:
        return f"Could not read corpus stats: {exc}"


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the MCP server over stdio (used by Claude Desktop and other MCP clients)."""
    logger.info("Starting AI4SAW MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
