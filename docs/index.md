# AI4SAW

**AI for Slavery and War — Open Intelligence Extraction Pipeline**

AI4SAW is an open-source Python pipeline that automates intelligence extraction from unstructured conflict and human rights corpora. It turns NGO reports, legal filings, UN documents, and news archives into structured datasets that feed academic and policy research.

**Primary domains:** conflict-related slavery in Bosnia and Sudan  
**Designed to feed:** CDISaW, ACLED, PostGIS, and QGIS workflows

---

## What it does

```
Unstructured text  →  Entities  →  Relations  →  Events
                   →  Knowledge Graph  →  Network Analysis
                   →  RAG Q&A  →  Contradiction Detection
                   →  Silence Detection  →  Corpus Discovery
```

### Core pipeline

| Stage | What it does |
|---|---|
| **Ingestion** | PDF, HTML, DOCX, plaintext → chunked → embedded → ChromaDB |
| **NER** | Few-shot entity extraction: PERSON, ORG, LOCATION, FACILITY, GROUP, LEGAL_INSTRUMENT |
| **Relations** | Chain-of-thought subject–predicate–object triple extraction |
| **Events** | Zero-shot → few-shot event classification (forced labour, trafficking, displacement…) |
| **RAG Q&A** | MMR retrieval → cross-encoder reranking → cited answers |
| **Silence detection** | Expectation-gap and density-map approaches for documentation gaps |

### Advanced features

| Feature | What it does |
|---|---|
| **Entity Resolution** | Cross-document deduplication: "RSF" = "Rapid Support Forces" |
| **GraphRAG** | Knowledge graph from relation triples; structural + semantic retrieval |
| **Temporal Graph** | Time-filtered graph queries: "command structure as of July 1995" |
| **Multi-hop Agent** | LangGraph ReAct agent for complex temporal/causal questions |
| **Contradiction Detection** | Surfaces conflicting claims across sources |
| **Network Analysis** | Betweenness centrality + Louvain community detection on actor graph |
| **Corpus Discovery** | ReliefWeb + GDELT search for documents not yet in the corpus |
| **MCP Server** | Exposes the corpus as a tool inside Claude Desktop |
| **LLM-as-Judge** | Automated extraction quality scoring without hand labels |

---

## Quick install

```bash
git clone https://github.com/jwilliamsresearch/ai4saw
cd ai4saw
uv sync
cp .env.example .env
ollama pull mistral && ollama pull nomic-embed-text
```

→ [Full getting started guide](getting-started.md)

---

## One-minute demo

```bash
# Ingest a document
ai4saw ingest file report.pdf --geography Bosnia

# Extract entities, relations, events
ai4saw extract pipeline

# Resolve entity aliases
ai4saw extract resolve

# Build the knowledge graph
ai4saw graph build

# Ask a question
ai4saw query ask "What forms of forced labour were documented in Bosnian detention camps?"

# Ask a complex multi-hop question
ai4saw graph agent "What was the command structure before Srebrenica and who gave the orders?"

# Check what's available to query
ai4saw info
```

---

## Design principles

**Model-agnostic.** Switch between Ollama (local, free), OpenRouter (cloud, pay-per-token), and HuggingFace (serverless) by changing one environment variable.

**Schema-first.** All outputs are Pydantic v2 validated. No silent data loss.

**Prompts as first-class artefacts.** All prompts live in versioned YAML files, not Python strings. Every extraction run logs the prompt version used.

**Researcher-facing.** The pipeline surfaces results for researcher review — it doesn't make decisions autonomously. Contradictions are reported, not resolved. Silence candidates are ranked, not interpreted.
