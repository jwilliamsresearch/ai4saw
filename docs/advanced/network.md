# Perpetrator Command Network

## What it does

Turns relation triples into a directed social network, then applies network science to identify command structures, key actors, and operational clusters.

```bash
ai4saw analyze network --gexf
```

## Why betweenness centrality

Betweenness centrality measures how often a node lies on the shortest path between other nodes. In command structures, high-betweenness nodes are typically:

- **Mid-level commanders** who relay orders between senior leadership and field units
- **Liaison officers** who connect otherwise separate operational groups
- **Logistical coordinators** who appear in both supply chains and command records

These actors are often overlooked in favour of top-level commanders (high out-degree) or direct perpetrators (high in-degree). ICTY prosecutors found that mid-level commanders were frequently critical to attribution chains precisely because they had both the knowledge and the control to enable or prevent violations.

## Community detection

Louvain community detection identifies clusters of actors who interact more with each other than with actors outside the cluster. In conflict networks, communities typically correspond to:

- Geographic commands (Drina Corps, Sarajevo-Romanija Corps)
- Operational units (execution squads, logistics, security services)
- Allied organisations (military + paramilitary coordination)

## Command edges

Not all relations are command relations. The network distinguishes "commanded", "ordered", "directed" (command edges) from "transported", "detained", "executed" (action edges). The `--command-edges` count in the output shows how many verified ordering relationships were extracted.

## Build and export

```bash
# JSON output
ai4saw analyze network \
  --relations-file output/relation_results.json \
  --registry-file data/entity_registry.json \
  --min-confidence 0.5 \
  --output output/network.json

# With Gephi GEXF export
ai4saw analyze network --gexf
```

The GEXF file can be opened directly in [Gephi](https://gephi.org/) for interactive visualisation with community colouring and centrality-scaled nodes.

## Output

```json title="output/network.json"
{
  "nodes": [
    {
      "id": "a3f1b2c4d5e6",
      "label": "Ratko Mladić",
      "entity_type": "PERSON",
      "betweenness_centrality": 0.847,
      "in_degree": 2,
      "out_degree": 14,
      "community_id": 0
    }
  ],
  "key_actors": ["Ratko Mladić", "Radislav Krstić", "Drina Corps"],
  "communities": {
    "0": ["Ratko Mladić", "VRS Main Staff", "Radovan Karadžić"],
    "1": ["Drina Corps", "Radislav Krstić", "Zdravko Tolimir"]
  },
  "total_nodes": 47,
  "total_edges": 89,
  "command_edges": 23
}
```

## Programmatic use

```python
from ai4saw.synthesis.network import build_command_network
from ai4saw.retrieval.graph_rag import to_networkx

analysis = build_command_network(relation_results, registry, min_confidence=0.5)

# Top actors by betweenness
for node in analysis.nodes[:5]:
    print(f"{node.label}: centrality={node.betweenness_centrality:.3f}")

# Communities
for cid, members in analysis.communities.items():
    print(f"Community {cid}: {', '.join(members[:5])}")
```
