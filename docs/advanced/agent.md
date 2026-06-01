# Multi-hop Reasoning Agent

## The problem with single-shot RAG

Standard RAG makes one retrieval call and generates one answer. This fails for questions that require chaining across documents and time:

> *"What happened in the six months before Srebrenica, who were the key actors, and what was the command structure that enabled the massacre?"*

This question requires temporal chaining (six months of events), actor tracking (who appears across multiple documents), and causal inference (what enabled what). Single retrieval cannot do this reliably.

## How the agent works

```bash
ai4saw graph agent "What was the command structure before Srebrenica and who gave the orders?"
```

The agent uses LangGraph's ReAct architecture: it reasons about what to look up, calls a tool, reasons about the result, calls another tool, and so on until it has enough context to answer.

**Three tools available:**

| Tool | What it does | When to use |
|---|---|---|
| `search_corpus` | Semantic vector search over ChromaDB | Open-ended questions, finding passages |
| `query_knowledge_graph` | k-hop graph traversal (temporal-aware) | Specific actor/event relations |
| `find_entity` | Resolved entity lookup with aliases | Resolving abbreviations, checking presence |

**Maximum iterations** — default 8 tool calls. Increase with `--max-iterations` for more thorough research at higher cost.

## Example output

```
Agent Answer:
The command structure in the months before Srebrenica centred on three key nodes.
General Ratko Mladić, as commander of the VRS Main Staff, issued the directive
Krivaja-95 in early July 1995 [Source: ICTY-judgment-krstic.pdf].
The Drina Corps, commanded by Radislav Krstić from 11 July, was the operational
unit responsible for the enclave. [Source: HRW-1995-report.pdf]...

Reasoning Steps:
  1. search_corpus        "command structure Bosnia 1995"
  2. query_knowledge_graph "Mladić command chain"
  3. search_corpus        "Krivaja-95 directive"
  4. query_knowledge_graph "Drina Corps Krstić July 1995"
  5. search_corpus        "Srebrenica enclave events June 1995"

Sources: ICTY-judgment-krstic.pdf | HRW-1995-report.pdf | UN-srebrenica-report.pdf
```

## Configuration

```bash
ai4saw graph agent "question" \
  --max-iterations 8 \     # default: 8 tool calls
  --output response.json   # save AgentResponse as JSON
```

## When to use which query mode

| Mode | Command | Best for |
|---|---|---|
| Vector search | `query ask` | Thematic questions, corpus overview |
| GraphRAG | `graph query` | Specific actor/relation lookups |
| **Multi-hop agent** | `graph agent` | Complex, multi-step, temporal questions |

## Programmatic use

```python
from ai4saw.retrieval.agent import multi_hop_answer

response = multi_hop_answer(
    "What was the role of the Rapid Support Forces in Darfur in 2023?",
    max_iterations=10,
)

print(response.answer)
print(f"Tool calls: {response.iterations}")
print(f"Sources: {response.sources_consulted}")
```
