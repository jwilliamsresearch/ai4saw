# AI4SAW — AI for Slavery and War

**Open intelligence extraction pipeline for conflict-related slavery research.**

Version 0.1 | James Williams, University of Nottingham  
Research Fellow in Slavery and War | James.williams4@nottingham.ac.uk

---

## What this is

AI4SAW automates intelligence extraction from unstructured conflict and human rights corpora — NGO reports, legal filings, news archives, UN documents, grey literature. It is modular, model-agnostic, and designed to feed structured datasets including CDISaW and ACLED.

**Primary pilot domains:** conflict-related slavery in Bosnia and Sudan.

### Core pipeline
1. **Ingestion** — PDF, HTML, DOCX, plaintext → chunked + embedded → ChromaDB
2. **Extraction** — NER (few-shot), relation triples (chain-of-thought), event classification (zero-shot → few-shot fallback)
3. **RAG Q&A** — MMR retrieval → cross-encoder re-ranking → cited answers
4. **Silence detection** — expectation-gap and density-map approaches

### Advanced features
5. **Entity Resolution** — cross-document entity deduplication using embedding similarity + string matching
6. **GraphRAG + Temporal Graph** — knowledge graph from relation triples with time-filtered queries (`--at 1995-07-11`)
7. **Contradiction Detection** — surfaces conflicting claims across sources for researcher review
8. **Multi-hop Agent** — LangGraph ReAct agent that chains tool calls for complex temporal/causal questions
9. **Active Corpus Discovery** — queries ReliefWeb and GDELT to find documents not yet in the corpus
10. **Automated Corpus Fetch** — discovers, downloads, registers, and ingests from six free APIs: OpenAlex, Semantic Scholar, arXiv, Internet Archive, UN Digital Library, GDELT (news, batched)
10. **Perpetrator Command Network** — betweenness centrality + community detection on the actor graph
11. **MCP Server** — exposes the corpus as a tool inside Claude Desktop (`ai4saw-mcp`)
12. **LLM-as-Judge** — automated extraction quality scoring by a frontier model without hand labels

---

## Quick start

### Prerequisites

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) (recommended)
- [Ollama](https://ollama.ai/) for free local inference

### Install

```bash
git clone <repo>
cd ai4saw
uv sync
cp .env.example .env
```

Pull the two local models:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Optional extras (require PyTorch — ARM Mac / Linux only)

```bash
uv sync --extra rerank   # cross-encoder re-ranking
uv sync --extra eval     # RAGAS evaluation
```

---

## Full workflow

```
ingest → extract pipeline → extract resolve → graph build
                                                    ↓
                         query ask    graph query    graph agent
                                                    ↓
                    analyze contradictions    analyze network
                                                    ↓
                              discover run    export all
```

### Step 1 — Ingest

```bash
# Single document
ai4saw ingest file report.pdf --doc-type report --geography Bosnia --date-published 1996-01-01

# Whole directory
ai4saw ingest corpus ./corpus --geography Sudan
```

### Step 2 — Extract

```bash
# NER + relations + events on all indexed chunks
ai4saw extract pipeline

# Resolve entity aliases across the corpus ("RSF" = "Rapid Support Forces")
# Required before graph build and network analysis
ai4saw extract resolve
```

### Step 3 — Build the knowledge graph

```bash
ai4saw graph build
```

Reads `output/relation_results.json` and `data/entity_registry.json` → writes `data/knowledge_graph.json`.

### Step 4 — Query

```bash
# Standard vector-search RAG
ai4saw query ask "What forms of forced labour were documented in Bosnian camps?"

# GraphRAG — structural + semantic
ai4saw graph query "What did the RSF do in El Geneina in 2023?"

# Multi-hop agent — complex temporal/causal questions
ai4saw graph agent "What happened in the six months before Srebrenica and who were the key actors?"
```

### Step 5 — Analyze

```bash
# Contradiction detection across sources
ai4saw analyze contradictions

# Perpetrator command network (+ Gephi GEXF export)
ai4saw analyze network --gexf
```

### Step 6 — Discover and fetch

```bash
# Passive discovery: find candidates for manual review
ai4saw discover run "El Geneina" "Masalit" "Rapid Support Forces"

# Auto-discover from entity registry
ai4saw discover run --from-registry --label LOCATION --top-n 10

# Active fetch: discover + download + register + ingest in one command
ai4saw discover fetch "El Geneina" "Masalit" --geography Sudan --max-docs 20

# Preview candidates without downloading
ai4saw discover fetch "Srebrenica" "Mladic" --geography Bosnia --dry-run

# Silence-driven fetch (targets coverage gaps detected by silence detector)
ai4saw discover fetch "Foča" "Prijedor" --geography Bosnia --silence-mode --yes
```

### Step 7 — Export

```bash
ai4saw export all
# → output/events.geojson, output/relations.json,
#   output/entities.json, output/silences.json, output/corpus_stats.json
```

### Check status

```bash
ai4saw info
```

---

## Providers

Switch provider by setting `PROVIDER` in `.env` — no code changes required.

| Provider | Cost | Best for |
|---|---|---|
| `ollama` (default) | Free | Bulk runs, development, offline |
| `openrouter` | Pay per token | Evaluation, frontier model comparison |
| `huggingface` | Free tier | Serverless, no local GPU |

```bash
# .env
PROVIDER=ollama
DEFAULT_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# For cloud:
PROVIDER=openrouter
OPENROUTER_API_KEY=sk-...
DEFAULT_MODEL=mistralai/mistral-7b-instruct
```

All LLM and embedding calls route through `ai4saw/core/providers.py`. No module references a provider directly.

---

## Architecture

```
Ingestion → Extraction → Synthesis
    ↓            ↓           ↓
ChromaDB    Pydantic    JSON/GeoJSON/GEXF
    ↑            ↑           ↑
         Provider Layer
  Ollama | OpenRouter | HuggingFace

Advanced layer (builds on above):
  EntityResolution → KnowledgeGraph → NetworkAnalysis
                          ↓
                    GraphRAG / Agent
                          ↓
                    ContradictionDetection
                          ↓
                    CorpusDiscovery
```

### Data flow

```
corpus/          → ingest          → ChromaDB (vector store)
                 → extract pipeline → output/{ner,relation,event}_results.json
                 → extract resolve  → data/entity_registry.json
                 → graph build      → data/knowledge_graph.json
                 → analyze network  → output/network.json + output/network.gexf
                 → analyze contradictions → output/contradictions.json
                 → discover run     → output/discovered_documents.json
                 → export all       → output/{events.geojson, relations.json, ...}
```

---

## Feature reference

### Entity Resolution (`ai4saw extract resolve`)

Merges aliases across the corpus into canonical entities. "The RSF", "Rapid Support Forces", and "the Khartoum paramilitaries" become one node with aliases tracked.

**Algorithm:** batch embeddings → cosine similarity matrix → union-find clustering (cosine ≥ 0.88 OR fuzzy string ratio ≥ 72). Conservative defaults; tune with `--cosine-threshold` and `--fuzzy-threshold`.

**Output:** `data/entity_registry.json`

---

### Knowledge Graph + GraphRAG (`ai4saw graph build`, `ai4saw graph query`)

Builds a directed graph where nodes are resolved entities and edges are verified relation triples. At query time, extracts the k-hop neighbourhood of entities mentioned in the question and combines this structural context with vector search results.

**Why this matters:** vector search answers "which documents mention X?"; graph traversal answers "what did X do to Y, where, when, and who ordered it?" — which is the actual question conflict researchers ask.

**Output:** `data/knowledge_graph.json` (also loadable as NetworkX DiGraph for custom analysis)

---

### Contradiction Detection (`ai4saw analyze contradictions`)

Surfaces conflicting claims across source documents. Uses a two-pass approach:

1. **Candidate generation** (cheap): group events by location/year, group relations by (subject, object). Flag pairs with mismatched event types or predicates.
2. **LLM verification** (per pair): calls the LLM with both claims and the `contradiction_check.yaml` prompt. Only pairs above the confidence threshold enter the report.

Contradictions are not errors — in conflict research they signal contested narratives, propaganda, or different witness perspectives. The pipeline surfaces them; researchers interpret them.

**Output:** `output/contradictions.json`

---

### Multi-hop Reasoning Agent (`ai4saw graph agent`)

A LangGraph ReAct agent with three tools:

| Tool | Does |
|---|---|
| `search_corpus` | Semantic vector search over ChromaDB |
| `query_knowledge_graph` | 2-hop subgraph retrieval from the knowledge graph |
| `find_entity` | Resolved entity lookup (aliases, frequency, source chunks) |

The agent decomposes the question, calls tools iteratively, and synthesises across all accumulated context. Handles temporal chaining, actor tracking, and cross-document questions that single-shot RAG cannot answer reliably.

**Max iterations** (default 8) caps LLM cost. Adjust with `--max-iterations`.

---

### Active Corpus Discovery (`ai4saw discover run`)

Queries two free, no-authentication-required APIs:

- **ReliefWeb** (`reliefweb.int`) — UNOCHA's humanitarian information platform: NGO reports, UN situation reports, assessments
- **GDELT** (`gdeltproject.org`) — global news event database, 65+ languages

Discovered documents are deduplicated against `corpus/sources.csv` (URLs already registered). The pipeline never auto-ingests — output is for researcher review.

**Silence-driven discovery:** run discovery on locations identified by the silence detector to directly target the gaps in your corpus.

---

### Perpetrator Command Network (`ai4saw analyze network`)

Builds a directed graph from relation triples and applies:

- **Betweenness centrality** — identifies actors who sit on the most paths between other nodes. In command structures, these are typically mid-level commanders critical to attribution chains.
- **Community detection** (Louvain algorithm) — reveals operational clusters, geographic commands, or allied groups.
- **Command edge filtering** — isolates the formal ordering relationship (predicates containing "ordered", "commanded", "directed", etc.) from informal connections.

Exports JSON and optionally GEXF for visualisation in [Gephi](https://gephi.org/).

**Output:** `output/network.json`, `output/network.gexf` (with `--gexf`)

---

## Repository structure

```
ai4saw/
├── ai4saw/                  # Python package
│   ├── core/
│   │   ├── config.py        # Env var loading (pydantic-settings)
│   │   ├── models.py        # All Pydantic v2 schemas — single source of truth
│   │   └── providers.py     # get_llm() / get_embedder() — provider abstraction
│   ├── ingestion/
│   │   ├── loaders.py       # PDF, HTML, DOCX, plaintext, URL
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter (1000 tok / 200 overlap)
│   │   └── embedder.py      # Deterministic-ID ChromaDB upsert (idempotent)
│   ├── extraction/
│   │   ├── ner.py           # Few-shot NER, retry on JSON parse failure
│   │   ├── relations.py     # Chain-of-thought triple extraction
│   │   └── events.py        # Zero-shot → few-shot fallback at 0.6 confidence
│   ├── retrieval/
│   │   ├── qa.py            # MMR retrieval → cross-encoder rerank → cited answer
│   │   ├── reranker.py      # cross-encoder/ms-marco-MiniLM-L-6-v2 wrapper
│   │   ├── graph_rag.py     # Knowledge graph build + graph-augmented retrieval
│   │   └── agent.py         # LangGraph ReAct multi-hop reasoning agent
│   ├── synthesis/
│   │   ├── entity_resolution.py  # Cross-document entity deduplication
│   │   ├── contradiction.py      # Two-pass contradiction detection
│   │   ├── network.py            # Perpetrator command network + Gephi export
│   │   ├── silence.py            # Expectation-gap + density-map silence detection
│   │   ├── export.py             # GeoJSON, JSON, corpus stats
│   │   └── geocoder.py           # Stub — deferred to Phase 5
│   ├── discovery/
│   │   └── discovery.py     # ReliefWeb + GDELT corpus discovery
│   └── cli.py               # Typer CLI entry point
├── prompts/                 # Versioned YAML prompt templates
│   ├── ner_few_shot.yaml
│   ├── relations_cot.yaml
│   ├── events_zero_shot.yaml
│   └── contradiction_check.yaml
├── eval/
│   ├── ner_benchmark.py     # Precision / recall / F1 vs gold labels
│   ├── rag_eval.py          # RAGAS faithfulness + relevance
│   └── testdata/
│       ├── ner_gold.json    # 5 hand-labelled gold chunks (Bosnia/Sudan)
│       └── rag_questions.json  # 10 benchmark Q&A pairs
├── notebooks/
│   ├── 01_ingestion_demo.ipynb
│   ├── 02_extraction_demo.ipynb
│   └── 03_rag_silence_demo.ipynb
├── corpus/
│   └── sources.csv          # Provenance + licence register (fill before ingesting)
├── data/                    # Generated — entity registry, knowledge graph
├── output/                  # Generated — extraction results, exports
├── pyproject.toml
├── .env.example
└── .python-version          # Pinned to 3.12 (onnxruntime Intel Mac compatibility)
```

---

## Prompts

All prompts live in `prompts/` as versioned YAML files. No prompt strings are hardcoded in Python. Each file contains:

| Field | Purpose |
|---|---|
| `version` | Semver — logged alongside extraction results for reproducibility |
| `task` | Human-readable description |
| `system` | System prompt |
| `examples` | Input/output pairs for few-shot prompts |
| `template` | User turn with `{variable}` placeholders |
| `model_notes` | Known behaviours and failure modes per model |

---

## Evaluation

```bash
# NER precision/recall/F1 against gold labels
ai4saw eval ner eval/testdata/ner_gold.json

# RAGAS RAG quality (requires --extra eval)
ai4saw eval rag eval/testdata/rag_questions.json
```

Results are written to `eval/results/` as JSON for reproducibility. The zero-shot vs. few-shot comparison in event classification is a publishable finding — strategy is logged per chunk.

---

## Corpus

Place documents in `corpus/` and register them in `corpus/sources.csv`.

```
filename, title, source_url, doc_type, language, date_published, geography, licence, notes
```

**Every document must have a verified licence before ingestion.** Supported formats: PDF, HTML, DOCX, plaintext.

---

### MCP Server (`ai4saw-mcp`)

Exposes the full corpus + knowledge graph as a Model Context Protocol server, making it a first-class tool inside Claude Desktop and any MCP-compatible AI client.

**Setup:** Copy `mcp_config.example.json` into your Claude Desktop config and restart. Claude can then call `search_corpus`, `query_knowledge_graph`, `find_entity`, and `ask_question` directly.

```json
{
  "mcpServers": {
    "ai4saw": {
      "command": "uv",
      "args": ["run", "ai4saw-mcp"],
      "cwd": "/path/to/ai4saw"
    }
  }
}
```

The `query_knowledge_graph` tool accepts an `at_date` parameter, enabling temporal queries from within Claude Desktop.

---

### Temporal Knowledge Graph (`ai4saw graph query --at DATE`)

Every edge in the knowledge graph carries `valid_from` and `valid_to` fields (ISO 8601). This enables time-filtered graph queries:

```bash
# Command structure as of 11 July 1995
ai4saw graph query "Drina Corps command" --at 1995-07-11

# RSF operations before April 2023
ai4saw graph query "RSF El Geneina" --at 2023-04-14
```

Edges where `valid_from > DATE` or `valid_to < DATE` are excluded. Edges with no date information are always included (assumed continuously valid). This enables questions like *"what was the command structure before Srebrenica?"* without rebuilding the graph.

---

### LLM-as-Judge (`ai4saw eval judge`)

Automated extraction quality scoring using a frontier model — no hand labels required.

```bash
# Set judge model in .env:  JUDGE_MODEL=anthropic/claude-3-5-sonnet
ai4saw eval judge --sample 20
```

Three dimensions scored 0.0–1.0: NER accuracy, relation groundedness (triple supported by source text), event classification correctness. Uses a different model from the extraction LLM to avoid circular self-assessment bias.

Output: `eval/results/judge_report.json` with per-chunk scores, aggregate means, and a ranked list of most common issues — the highest-yield targets for prompt improvement.

---

## Build phases

| Phase | Scope | Milestone |
|---|---|---|
| 1 | Ingestion, ChromaDB, provider abstraction | Index 50 documents, retrieve by query |
| 2 | NER + few-shot prompts | Extract entities, eval against gold |
| 3 | Relation extraction + event classification | Full extraction pipeline |
| 4 | RAG Q&A + re-ranker | Answer 20 benchmark questions |
| 5 | Silence detection, entity resolution, GraphRAG | Graph built, silence candidates ranked |
| 6 | Agent, contradiction detection, network, discovery | Full advanced pipeline |
| 7 | Eval suite + write-up | External-facing artefact |

---

## Open questions (deferred)

- **Geocoding** — LOCATION entities → lat/lon for GeoJSON. Options: Nominatim, GeoNames, few-shot LLM geocoder. See `ai4saw/synthesis/geocoder.py`.
- **Language** — Bosnian/Arabic sources require multilingual models or translation pre-processing.
- **CDISaW write-back** — schema alignment work deferred to post-v0.1.

---

## Licence

MIT. See `corpus/sources.csv` for individual document licences.
