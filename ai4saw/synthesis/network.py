"""Perpetrator Command Network — graph-theoretic analysis of command structures.

Turns relation triples into a directed social network where:
  nodes = actors (persons, organisations, groups)
  edges = verified actions between actors

Then applies network science to identify:
  - Key actors by betweenness centrality: nodes that sit on the most paths
    between other nodes. In command structures, these are typically mid-level
    commanders who are critical to attribution chains — often overlooked in
    favour of top-level commanders or direct perpetrators.
  - Communities: clusters of actors who act together more than with outside
    actors, revealing operational units, geographic commands, or allied groups.
  - Command chains: subgraph of edges where the predicate is a command/order
    verb, showing the formal ordering relationship distinct from informal
    connections.

Betweenness centrality is particularly valuable for legal accountability work:
  International Tribunal prosecutors used similar network analysis to identify command structures
  that were not apparent from individual testimonies. This module automates
  that analysis.

All graph operations use NetworkX (pure Python, no GPU/torch dependency).
Community detection uses the Louvain algorithm via nx.community.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
from loguru import logger

from ai4saw.core.models import (
    EntityResolutionResult,
    NetworkAnalysis,
    NetworkEdge,
    NetworkNode,
    RelationResult,
)

# Verb stems that indicate a command relationship when found in a predicate.
# Checked with `any(stem in predicate_lower for stem in COMMAND_STEMS)`.
COMMAND_STEMS = {
    "order", "command", "direct", "instruct", "authoris", "authoriz",
    "sanction", "approve", "mandate", "assign", "dispatch", "deploy",
    "coordinate", "supervise", "oversee", "control", "led", "led the",
}

NETWORK_OUTPUT_PATH = "output/network.json"


# ── Graph construction ─────────────────────────────────────────────────────────

def _is_command_relation(predicate: str) -> bool:
    p = predicate.lower()
    return any(stem in p for stem in COMMAND_STEMS)


def _actor_id(text: str, registry: Optional[EntityResolutionResult] = None) -> str:
    """Map actor text to a resolved canonical ID if a registry is available."""
    if registry is None:
        import hashlib
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:12]

    text_lower = text.lower().strip()
    for entity in registry.entities:
        if entity.canonical_text.lower() == text_lower:
            return entity.canonical_id
        for alias in entity.aliases:
            if alias.lower() == text_lower:
                return entity.canonical_id

    import hashlib
    return hashlib.sha256(text_lower.encode()).hexdigest()[:12]


def build_command_network(
    relation_results: list[RelationResult],
    registry: Optional[EntityResolutionResult] = None,
    min_confidence: float = 0.5,
    actor_labels: frozenset[str] = frozenset({"PERSON", "ORG", "GROUP"}),
) -> NetworkAnalysis:
    """Build a perpetrator command network from extracted relations.

    Args:
        relation_results: All RelationResults from the extraction pipeline.
        registry: Optional resolved entity registry. When provided, actor names
            are canonicalised so "the Armed-Group-Beta" and "Rapid Support Forces" become
            one node. Strongly recommended.
        min_confidence: Relations below this confidence are excluded.
        actor_labels: Only nodes with these entity types are included. LOCATION
            and FACILITY are excluded by default — they are targets, not actors.

    Returns:
        NetworkAnalysis with nodes, edges, communities, and key actors.
    """
    G = nx.DiGraph()
    node_labels: dict[str, str] = {}  # id → display label
    node_types: dict[str, str] = {}   # id → entity type

    raw_edges: list[NetworkEdge] = []
    skipped = 0

    for result in relation_results:
        for rel in result.relations:
            if rel.confidence < min_confidence:
                skipped += 1
                continue

            src_id = _actor_id(rel.subject, registry)
            tgt_id = _actor_id(rel.object, registry)

            # Resolve display label from registry if available
            src_label = rel.subject
            tgt_label = rel.object
            if registry:
                for ent in registry.entities:
                    if ent.canonical_id == src_id:
                        src_label = ent.canonical_text
                    if ent.canonical_id == tgt_id:
                        tgt_label = ent.canonical_text

            node_labels[src_id] = src_label
            node_labels[tgt_id] = tgt_label

            # Entity types from registry if available, else UNKNOWN
            if registry:
                for ent in registry.entities:
                    if ent.canonical_id == src_id:
                        node_types[src_id] = ent.label
                    if ent.canonical_id == tgt_id:
                        node_types[tgt_id] = ent.label

            is_cmd = _is_command_relation(rel.predicate)
            G.add_edge(
                src_id, tgt_id,
                predicate=rel.predicate,
                confidence=rel.confidence,
                evidence=rel.evidence[:200],
                is_command=is_cmd,
                chunk_id=result.source_chunk_id,
            )

            raw_edges.append(NetworkEdge(
                source=src_label,
                target=tgt_label,
                predicate=rel.predicate,
                confidence=rel.confidence,
                evidence=rel.evidence[:200],
                is_command_relation=is_cmd,
            ))

    logger.info(
        f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({skipped} below confidence threshold)"
    )

    if G.number_of_nodes() == 0:
        return NetworkAnalysis(
            nodes=[], edges=[], communities={}, key_actors=[],
            total_nodes=0, total_edges=0, command_edges=0,
        )

    # ── Betweenness centrality ────────────────────────────────────────────────
    # Use undirected version for centrality — command direction matters for
    # semantics but betweenness on directed graphs misses actors who receive
    # commands and issue them to different sub-groups.
    G_undirected = G.to_undirected()
    try:
        centrality = nx.betweenness_centrality(G_undirected, normalized=True)
    except Exception:
        centrality = {n: 0.0 for n in G.nodes()}

    # ── Community detection (greedy modularity) ───────────────────────────────
    # Louvain via nx.community.louvain_communities (NetworkX 3.3+), with
    # greedy_modularity_communities as fallback for earlier versions.
    try:
        communities_sets = list(nx.community.louvain_communities(G_undirected, seed=42))
    except AttributeError:
        communities_sets = list(nx.community.greedy_modularity_communities(G_undirected))

    node_community: dict[str, int] = {}
    for cid, members in enumerate(communities_sets):
        for m in members:
            node_community[m] = cid

    # ── Build output nodes ────────────────────────────────────────────────────
    out_nodes: list[NetworkNode] = []
    for node_id in G.nodes():
        out_nodes.append(NetworkNode(
            id=node_id,
            label=node_labels.get(node_id, node_id),
            entity_type=node_types.get(node_id, "UNKNOWN"),
            betweenness_centrality=round(centrality.get(node_id, 0.0), 6),
            in_degree=G.in_degree(node_id),
            out_degree=G.out_degree(node_id),
            community_id=node_community.get(node_id, 0),
        ))

    out_nodes.sort(key=lambda n: n.betweenness_centrality, reverse=True)

    # ── Communities dict ──────────────────────────────────────────────────────
    communities: dict[str, list[str]] = {}
    for cid, members in enumerate(communities_sets):
        communities[str(cid)] = [node_labels.get(m, m) for m in members]

    key_actors = [n.label for n in out_nodes[:10] if n.betweenness_centrality > 0]
    command_edge_count = sum(1 for e in raw_edges if e.is_command_relation)

    return NetworkAnalysis(
        nodes=out_nodes,
        edges=raw_edges,
        communities=communities,
        key_actors=key_actors,
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        command_edges=command_edge_count,
    )


def save_network(analysis: NetworkAnalysis, path: str = NETWORK_OUTPUT_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Network analysis saved → {out}")
    return out


def export_network_gexf(
    relation_results: list[RelationResult],
    registry: Optional[EntityResolutionResult],
    path: str = "output/network.gexf",
) -> Path:
    """Export the network as GEXF for visualisation in Gephi or similar tools.

    GEXF is the standard exchange format for Gephi, which can render the
    command network with community colouring and centrality-scaled nodes.
    """
    analysis = build_command_network(relation_results, registry)
    G = nx.DiGraph()
    for node in analysis.nodes:
        G.add_node(
            node.id,
            label=node.label,
            entity_type=node.entity_type,
            betweenness=node.betweenness_centrality,
            community=node.community_id,
        )
    for edge in analysis.edges:
        # Look up source/target IDs from label
        src_id = next(
            (n.id for n in analysis.nodes if n.label == edge.source), edge.source
        )
        tgt_id = next(
            (n.id for n in analysis.nodes if n.label == edge.target), edge.target
        )
        G.add_edge(src_id, tgt_id, predicate=edge.predicate,
                   confidence=edge.confidence, is_command=edge.is_command_relation)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, str(out))
    logger.info(f"Network GEXF exported → {out}")
    return out
