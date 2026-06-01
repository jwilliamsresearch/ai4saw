"""Multi-hop Reasoning Agent — LangGraph ReAct agent for complex corpus queries.

Why multi-hop over single-shot RAG:
  Standard RAG: one retrieval → one answer.
  Multi-hop: decomposes the question → retrieves for each sub-question →
  queries the knowledge graph → synthesises across all context.

This handles questions that require chaining across documents and time:
  "What happened in the six months before the Foča massacres and who were
   the key actors in the buildup?"
  → requires temporal chaining, actor tracking, and event sequencing across
    many documents — single-shot retrieval cannot do this reliably.

Architecture:
  Uses LangGraph's create_react_agent with three tools:
    1. search_corpus        — semantic vector search over ChromaDB
    2. query_knowledge_graph — structured graph traversal (k-hop subgraph)
    3. find_entity          — resolved entity lookup (aliases, frequency, sources)

  The agent decides which tool to call and how many times. A LangGraph
  ReAct agent naturally handles multi-hop: it calls tools iteratively,
  accumulating context, until it has enough information to answer.

  Max iterations is capped (default 8) to bound LLM cost on complex questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import AgentResponse, AgentStep
from ai4saw.core.providers import get_llm

_SYSTEM_PROMPT = """You are an expert conflict and human rights research analyst.
You have access to a corpus of conflict documents and a structured knowledge graph.

Use your tools to answer questions thoroughly and accurately.
- Call search_corpus to find relevant document passages.
- Call query_knowledge_graph to find structured relations between actors and events.
- Call find_entity to resolve an entity name and find all its aliases and appearances.
- Chain multiple tool calls when the question requires cross-referencing information.
- Do not guess or hallucinate. If evidence is insufficient, say so explicitly.
- Cite your sources by referring to the source files returned by your tools.
- Be precise about dates, locations, and actor names."""


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def search_corpus(query: str) -> str:
    """Search the indexed document corpus for passages relevant to a query.

    Returns the top 3 most relevant passages with source metadata.
    Use this for open-ended thematic questions and specific event lookups.
    """
    from ai4saw.ingestion.embedder import get_vector_store
    from ai4saw.retrieval.reranker import rerank

    try:
        store = get_vector_store()
        docs = store.similarity_search(query, k=6)
        reranked = rerank(query, docs, top_n=3)

        if not reranked:
            return "No relevant passages found in the corpus for this query."

        parts = []
        for i, doc in enumerate(reranked, 1):
            source = doc.metadata.get("source_filename", "unknown")
            geo = doc.metadata.get("geography") or ""
            date = doc.metadata.get("date_published") or ""
            meta = " | ".join(x for x in [geo, date] if x)
            parts.append(
                f"[Passage {i}] Source: {source}{' (' + meta + ')' if meta else ''}\n"
                f"{doc.page_content[:600]}"
            )
        return "\n\n".join(parts)

    except Exception as exc:
        return f"Corpus search failed: {exc}"


@tool
def query_knowledge_graph(entity_name: str) -> str:
    """Get all structured relations for a named entity from the knowledge graph.

    Returns the entity's 2-hop neighbourhood: everything it did, everything
    done to it, and the entities it is connected to.
    Use this for questions about specific actors, organisations, or locations.
    """
    from ai4saw.retrieval.graph_rag import graph_context_for_query

    result = graph_context_for_query(entity_name)
    if not result:
        return (
            f"No graph entries found for '{entity_name}'. "
            f"Either the entity was not extracted, or the knowledge graph has not been built yet. "
            f"Try search_corpus instead."
        )
    return result


@tool
def find_entity(name: str) -> str:
    """Look up a named entity in the resolved entity registry.

    Returns the canonical name, all aliases, occurrence count, and source chunks.
    Use this to resolve abbreviations, check if an entity is in the corpus,
    or find all the ways an actor is referred to across sources.
    """
    registry_path = Path("data/entity_registry.json")
    if not registry_path.exists():
        return (
            "Entity registry not found. Run `ai4saw extract resolve` first."
        )

    import json
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
        return f"No entity matching '{name}' found in the registry."

    parts = []
    for m in matches[:5]:
        aliases_str = ", ".join(m.get("aliases", [])[:6]) or "none"
        parts.append(
            f"[{m['label']}] {m['canonical_text']}\n"
            f"  Aliases: {aliases_str}\n"
            f"  Occurrences: {m['occurrence_count']}\n"
            f"  Source chunks: {len(m.get('source_chunks', []))}"
        )
    return "\n\n".join(parts)


# ── Agent construction ─────────────────────────────────────────────────────────

def _build_agent() -> Any:
    """Construct the LangGraph ReAct agent with all tools."""
    from langgraph.prebuilt import create_react_agent

    llm = get_llm()
    tools = [search_corpus, query_knowledge_graph, find_entity]

    return create_react_agent(
        llm,
        tools,
        prompt=_SYSTEM_PROMPT,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def multi_hop_answer(
    question: str,
    max_iterations: int = 8,
) -> AgentResponse:
    """Answer a complex question by decomposing it and chaining tool calls.

    Args:
        question: A research question that may require multiple retrieval steps.
        max_iterations: Maximum tool calls before the agent is forced to answer.
            Higher values allow more thorough research but cost more tokens.

    Returns:
        AgentResponse with the answer, reasoning steps, and consulted sources.
    """
    from langchain_core.messages import HumanMessage

    logger.info(f"Multi-hop agent: {question!r}")
    agent = _build_agent()

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": max_iterations * 3},
        )
    except Exception as exc:
        logger.error(f"Agent invocation failed: {exc}")
        return AgentResponse(
            question=question,
            steps=[],
            answer=f"Agent failed: {exc}",
            sources_consulted=[],
            graph_nodes_consulted=[],
            iterations=0,
        )

    messages = result.get("messages", [])

    # Extract tool calls and results from the message history
    steps: list[AgentStep] = []
    sources: list[str] = []
    graph_nodes: list[str] = []

    for msg in messages:
        # Tool call messages have tool_calls attribute
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                steps.append(AgentStep(
                    sub_question=str(list(tool_args.values())[0]) if tool_args else "",
                    tool_used=tool_name,
                    result_summary="",  # filled below
                ))

        # ToolMessage contains the result
        if hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
            content = str(msg.content)
            if steps:
                steps[-1] = AgentStep(
                    sub_question=steps[-1].sub_question,
                    tool_used=steps[-1].tool_used,
                    result_summary=content[:200],
                )
            # Track sources and graph nodes from tool results
            for line in content.split("\n"):
                if "Source:" in line:
                    src = line.split("Source:")[-1].strip().split("(")[0].strip()
                    if src and src not in sources:
                        sources.append(src)
                if "[PERSON]" in line or "[ORG]" in line or "[GROUP]" in line:
                    node = line.strip().split("]")[-1].strip()
                    if node and node not in graph_nodes:
                        graph_nodes.append(node)

    # Final answer is the last AI message
    final_answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue  # skip tool-call messages
            final_answer = str(msg.content)
            break

    return AgentResponse(
        question=question,
        steps=steps,
        answer=final_answer,
        sources_consulted=sources,
        graph_nodes_consulted=graph_nodes,
        iterations=len(steps),
    )
