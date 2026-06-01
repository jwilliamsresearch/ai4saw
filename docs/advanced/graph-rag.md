# GraphRAG and the Temporal Knowledge Graph

## Why a knowledge graph

Vector search answers: *"which documents mention X?"*

Graph traversal answers: *"what did X do to Y, where, when, and who ordered it?"*

The latter is the actual question conflict researchers ask. The knowledge graph makes structured relations queryable independently of whether they appear verbatim in a retrievable passage.

## Build the graph

Requires entity resolution to have run first.

```bash
ai4saw extract resolve     # prerequisite
ai4saw graph build
```

**What it builds:**

- **Nodes** — all resolved entities (from `data/entity_registry.json`)
- **Edges** — verified relation triples (from `output/relation_results.json`), filtered by `--min-confidence 0.5`
- **Temporal fields** — each edge carries `valid_from` (from the relation's date) and `valid_to` (open-ended by default)

Output: `data/knowledge_graph.json`

## Query the graph

### Standard graph query

```bash
ai4saw graph query "What did the Drina Corps do in Location Alpha?"
```

This:
1. Finds all graph nodes whose labels appear in the question
2. Extracts their 2-hop neighbourhood
3. Renders the subgraph as structured prose
4. Optionally combines with vector search results (`--combine-vector`, default on)

### Temporal graph query

```bash
# Command structure as of 11 a specific date
ai4saw graph query "Drina Corps command" --at YYYY-MM-DD

# Who was active in Location Beta before April 2023?
ai4saw graph query "Location Beta actors" --at 2023-04-14
```

When `--at DATE` is provided, edges where `valid_from > DATE` or `valid_to < DATE` are excluded. Edges with no date are always included (assumed continuously valid).

This enables questions like *"what was the command structure before Location Alpha?"* that require filtering a 3-year corpus down to a single point in time.

## How temporal filtering works

Every edge in the graph has:

```json
{
  "valid_from": "1992-04-06",    // populated from relation date
  "valid_to": null               // open-ended by default
}
```

At query time with `--at YYYY-MM-DD`:

| Edge | valid_from | valid_to | Included? |
|---|---|---|---|
| Commander Alpha commanded Drina Corps | 1992-04-06 | null | ✓ (started before, still active) |
| Tolimir ordered execution | 1995-07-13 | null | ✗ (started after query date) |
| Armed-Group-Beta in Location Beta | 2023-04-15 | null | ✗ (future) |
| Karadžić directed operations | null | null | ✓ (no date info — assume always valid) |

## Programmatic use

```python
from ai4saw.retrieval.graph_rag import (
    load_knowledge_graph,
    graph_context_for_query,
    to_networkx,
)

graph = load_knowledge_graph("data/knowledge_graph.json")

# Temporal query
context = graph_context_for_query(
    "Drina Corps command structure",
    graph=graph,
    hops=2,
    at_date="YYYY-MM-DD",
)

# NetworkX for custom analysis
G = to_networkx(graph)
import networkx as nx
print(nx.info(G))
```

## Output format

The rendered graph context looks like:

```
=== Knowledge Graph Context (at YYYY-MM-DD) ===

[ORG] Drina Corps
  Also known as: the Drina Corps, Armed-Group-Alpha Drina Corps
  → commanded by → Radislav Commander Beta (YYYY-MM-DD | Location Alpha) [conf=0.95]
    Evidence: "Commander Beta assumed command of the Drina Corps on 11 a specific date"
  → participated in → Location Alpha massacre (YYYY-MM-DD | Location Alpha) [conf=0.97]
    Evidence: "Drina Corps units executed prisoners at Kravica warehouse"

[PERSON] Radislav Commander Beta
  → commanded → Drina Corps (valid from YYYY-MM-DD)
  → convicted by → International Tribunal (2001)
```
