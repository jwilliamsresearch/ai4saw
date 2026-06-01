"""GraphRAG — knowledge graph construction and graph-augmented retrieval.

Why GraphRAG over plain vector search:
  Vector search answers "which documents mention X?"
  Graph traversal answers "what did X do to Y, where, when, and who ordered it?"

The latter is the actual question conflict researchers ask. This module implements
both the build phase (relations → NetworkX graph → JSON) and the query phase
(entity lookup → k-hop subgraph → structured context for the LLM).

Build pipeline:
  RelationResults + EntityResolutionResult → KnowledgeGraph → data/knowledge_graph.json

Query pipeline:
  question → entity extraction → subgraph retrieval → text rendering → LLM context

The knowledge graph and vector store are complementary, not alternatives.
GraphRAG combines both: structural graph context for relational questions,
semantic vector search for open-ended or thematic questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
from loguru import logger

from ai4saw.core.models import (
    EntityResolutionResult,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    RelationResult,
    ResolvedEntity,
)

GRAPH_PATH = "data/knowledge_graph.json"


# ── Graph building ─────────────────────────────────────────────────────────────

def _entity_id_for_text(
    text: str,
    registry: EntityResolutionResult,
) -> tuple[str, str]:
    """Map a relation subject/object string to a resolved entity (id, canonical_text).

    Tries exact match first, then alias match, then falls back to the raw text
    as its own node. This ensures no relations are silently dropped.
    """
    text_lower = text.lower().strip()
    for entity in registry.entities:
        if entity.canonical_text.lower() == text_lower:
            return entity.canonical_id, entity.canonical_text
        for alias in entity.aliases:
            if alias.lower() == text_lower:
                return entity.canonical_id, entity.canonical_text
    # Fall back: create an unresolved node for this text
    import hashlib
    fallback_id = hashlib.sha256(text_lower.encode()).hexdigest()[:12]
    return fallback_id, text


def build_knowledge_graph(
    relation_results: list[RelationResult],
    entity_registry: EntityResolutionResult,
    min_confidence: float = 0.5,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from extracted relations and resolved entities.

    Args:
        relation_results: All RelationResults from the extraction pipeline.
        entity_registry: Resolved entity registry (output of entity_resolution.resolve_entities).
        min_confidence: Relations below this confidence are excluded. 0.5 is a
            reasonable floor; lower values add noise, higher values lose coverage.

    Returns:
        A KnowledgeGraph with nodes (entities) and edges (relations).
    """
    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: list[KnowledgeGraphEdge] = []

    # Seed nodes from the entity registry
    for entity in entity_registry.entities:
        nodes[entity.canonical_id] = KnowledgeGraphNode(
            id=entity.canonical_id,
            text=entity.canonical_text,
            entity_type=entity.label,
            aliases=entity.aliases,
            occurrence_count=entity.occurrence_count,
        )

    skipped = 0
    for result in relation_results:
        for rel in result.relations:
            if rel.confidence < min_confidence:
                skipped += 1
                continue

            src_id, src_text = _entity_id_for_text(rel.subject, entity_registry)
            tgt_id, tgt_text = _entity_id_for_text(rel.object, entity_registry)

            # Add unresolved nodes on-the-fly (they may be in a chunk not yet extracted)
            if src_id not in nodes:
                nodes[src_id] = KnowledgeGraphNode(
                    id=src_id, text=src_text, entity_type="UNKNOWN"
                )
            if tgt_id not in nodes:
                nodes[tgt_id] = KnowledgeGraphNode(
                    id=tgt_id, text=tgt_text, entity_type="UNKNOWN"
                )

            edges.append(
                KnowledgeGraphEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    predicate=rel.predicate,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                    date=rel.date,
                    location=rel.location,
                    source_chunk_id=result.source_chunk_id,
                    # Populate temporal window: the relation's date becomes valid_from.
                    # valid_to is left None (open-ended) — we rarely know when a
                    # command relationship ended. Researchers can annotate termination
                    # dates manually in the registry if needed.
                    valid_from=rel.date,
                    valid_to=None,
                )
            )

    graph = KnowledgeGraph(
        nodes=list(nodes.values()),
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
    )

    logger.info(
        f"Knowledge graph built: {len(nodes)} nodes, {len(edges)} edges "
        f"({skipped} relations excluded below confidence {min_confidence})"
    )
    return graph


def save_knowledge_graph(graph: KnowledgeGraph, path: str = GRAPH_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Knowledge graph saved → {out}")
    return out


def load_knowledge_graph(path: str = GRAPH_PATH) -> KnowledgeGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgeGraph(**data)


# ── NetworkX conversion ────────────────────────────────────────────────────────

def to_networkx(graph: KnowledgeGraph) -> nx.DiGraph:
    """Convert a KnowledgeGraph to a NetworkX DiGraph for algorithmic analysis."""
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.id, **node.model_dump())
    for edge in graph.edges:
        G.add_edge(
            edge.source_id,
            edge.target_id,
            **{k: v for k, v in edge.model_dump().items()
               if k not in ("source_id", "target_id")},
        )
    return G


# ── Graph-augmented retrieval ─────────────────────────────────────────────────

def _edge_valid_at(edge_data: dict, at_date: Optional[str]) -> bool:
    """Return True if an edge is temporally valid at the given ISO date string.

    Filtering rules:
    - If at_date is None: all edges pass (no temporal filter requested).
    - If edge has no valid_from: passes (we don't know when it started, assume always valid).
    - If valid_from <= at_date: check valid_to.
    - If valid_to is None: edge is open-ended, passes.
    - If at_date <= valid_to: passes.
    """
    if at_date is None:
        return True
    valid_from = edge_data.get("valid_from")
    valid_to = edge_data.get("valid_to")
    if valid_from and valid_from > at_date:
        return False
    if valid_to and valid_to < at_date:
        return False
    return True


def _render_subgraph_as_text(
    G: nx.DiGraph,
    seed_node_ids: list[str],
    hops: int = 2,
    at_date: Optional[str] = None,
) -> str:
    """Convert a k-hop neighbourhood around seed nodes into readable text.

    The rendered text is injected into the LLM prompt alongside vector search
    results. Structured prose is more faithful than raw JSON for LLM consumption.

    Args:
        at_date: ISO 8601 date string. When set, only edges valid on that date
            are included — enabling "what did the command structure look like
            in a specific date?" queries over the temporal graph.
    """
    visited: set[str] = set()
    frontier = set(seed_node_ids)

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node_id in frontier:
            if node_id not in G:
                continue
            next_frontier.update(G.successors(node_id))
            next_frontier.update(G.predecessors(node_id))
        frontier = next_frontier - visited
        visited.update(frontier)

    all_nodes = set(seed_node_ids) | visited
    header = f"=== Knowledge Graph Context{' (at ' + at_date + ')' if at_date else ''} ===\n"
    lines = [header]

    for node_id in all_nodes:
        if node_id not in G:
            continue
        node_data = G.nodes[node_id]
        label = node_data.get("text", node_id)
        entity_type = node_data.get("entity_type", "UNKNOWN")
        aliases = node_data.get("aliases", [])

        lines.append(f"[{entity_type}] {label}")
        if aliases:
            lines.append(f"  Also known as: {', '.join(aliases[:5])}")

        for _, tgt, edge_data in G.out_edges(node_id, data=True):
            if not _edge_valid_at(edge_data, at_date):
                continue
            tgt_label = G.nodes[tgt].get("text", tgt) if tgt in G else tgt
            conf = edge_data.get("confidence", 0.0)
            date = edge_data.get("date") or ""
            loc = edge_data.get("location") or ""
            evidence = edge_data.get("evidence", "")[:120]
            temporal = (
                f"valid from {edge_data['valid_from']}" if edge_data.get("valid_from") else ""
            )
            meta = " | ".join(x for x in [date, loc, temporal] if x)
            lines.append(
                f"  → {edge_data.get('predicate', '?')} → {tgt_label}"
                f"{' (' + meta + ')' if meta else ''} [conf={conf:.2f}]"
            )
            if evidence:
                lines.append(f'    Evidence: "{evidence}"')
        lines.append("")

    return "\n".join(lines)


def _find_nodes_for_query(G: nx.DiGraph, query: str) -> list[str]:
    """Find graph nodes whose labels appear in the query string."""
    query_lower = query.lower()
    matched: list[str] = []
    for node_id, data in G.nodes(data=True):
        label = data.get("text", "").lower()
        aliases = [a.lower() for a in data.get("aliases", [])]
        if label in query_lower or any(a in query_lower for a in aliases):
            matched.append(node_id)
    return matched


def graph_context_for_query(
    query: str,
    graph: Optional[KnowledgeGraph] = None,
    graph_path: str = GRAPH_PATH,
    hops: int = 2,
    at_date: Optional[str] = None,
) -> str:
    """Return a structured text block of graph context relevant to a query.

    Used by both the standard QA pipeline (as supplementary context) and the
    multi-hop agent (as a dedicated graph_query tool).

    Args:
        at_date: ISO 8601 date (e.g. "YYYY-MM-DD"). When provided, only edges
            valid on that date are included. Enables temporal queries like
            "what was the command structure before Location Alpha?"

    Returns an empty string if the graph file doesn't exist — callers degrade
    gracefully to vector-only retrieval.
    """
    if graph is None:
        path = Path(graph_path)
        if not path.exists():
            logger.debug("Knowledge graph not found — skipping graph context.")
            return ""
        graph = load_knowledge_graph(graph_path)

    G = to_networkx(graph)
    seed_nodes = _find_nodes_for_query(G, query)

    if not seed_nodes:
        logger.debug(f"No graph nodes matched query: {query!r}")
        return ""

    context = _render_subgraph_as_text(G, seed_nodes, hops=hops, at_date=at_date)
    suffix = f" at {at_date}" if at_date else ""
    logger.debug(f"Graph context{suffix}: {len(seed_nodes)} seed nodes, {hops}-hop neighbourhood")
    return context
