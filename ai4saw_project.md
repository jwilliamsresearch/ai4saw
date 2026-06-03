# AI4SAW — Autonomous Intelligence Extraction Pipeline

**University of Nottingham**

---

## What it does

AI4SAW is an end-to-end agentic system that autonomously discovers, ingests, and analyses documents related to conflict, war crimes, human trafficking, and human rights violations. Given a natural language research query, it runs continuously — finding sources, building a corpus, extracting structured intelligence, and surfacing findings — without further human input.

---

## Architecture

**Continuous parallel pipeline:**

```
Seeding (DDG · Wikipedia · CrossRef · OpenAlex · 82 RSS feeds · GDELT · Internet Archive)
    ↓  LLM-generated targeted queries
4× Parallel HTTP fetchers  →  LLM prescreen (qwen2.5:0.5b)  →  ChromaDB embedding
                                                                        ↓
                                              Reasoning (qwen2.5:7b) fires every N ingested docs
                                                                        ↓
                                              Novel entities + queries → back into seeding
```

Three models run concurrently within an 8GB VRAM budget:
- **qwen2.5:0.5b** — fast yes/no document relevance screening (~0.3s per doc)
- **qwen2.5:7b** — deep reasoning, entity extraction, goal-setting, narration
- **nomic-embed-text** — semantic embedding into ChromaDB

**Discovery sources (no API keys required):**
OpenAlex · Semantic Scholar · arXiv · Internet Archive · GDELT · CrossRef · DOAJ · Europe PMC · PubMed · World Bank Documents · HDX (OCHA) · 82 verified RSS feeds across 6 continents

---

## Key features

**Agentic snowball discovery**  
The LLM reads each ingested document, extracts novel entities (commanders, units, locations, legal cases), generates targeted search queries, and feeds them back into the discovery loop — finding documents no fixed template would reach.

**LLM memory and goal-setting**  
The system sets and persists 3-5 concrete research goals based on current findings. Goals survive restarts, guide future search queries, and are included in every reasoning prompt — giving the agent a sense of direction across sessions.

**Multi-stage extraction pipeline**  
After corpus collection: NER extracts structured entities, relation extraction builds subject-predicate-object triples, event classification produces geolocated incidents with timestamps — all exportable as GeoJSON, GEXF (Gephi), and JSON.

**RAG Q&A**  
The full corpus is queryable via semantic retrieval: *"Who commanded the forces at X?"*, *"What legislation was passed between 2018 and 2022?"*

**Real-time terminal dashboard**  
Rich Live UI with live feed, entity network chart, relevance trend, query pipeline, LLM model status indicators (running/idle per model), research goals panel, and AI-generated research summaries.

---

## Stack

Python · ChromaDB · LangChain · Ollama · PostgreSQL (optional) · Rich TUI · httpx · threading · queue · Docker-compatible

---

## Results

- Autonomous corpus construction across domains: Bosnian War / Srebrenica, North Korea nuclear programme, US human trafficking 2014–2025
- 82-source RSS network covering Africa, Middle East, Asia, Latin America, Europe, with no API keys
- Multi-model parallel inference achieving 4× fetch throughput vs sequential baseline
- Persistent agent state: frontier, discovered entities, executed queries, goals — all survive restarts

---

## Why it matters for industry

AI4SAW demonstrates the full engineering stack of a production agentic AI system:

- **Agentic loop design** — the system decides what to search for next based on what it finds, not fixed templates
- **Multi-model orchestration** — separate models for different tasks, tuned to VRAM constraints
- **Production-quality engineering** — threading, queue management, error recovery, persistent state, timeout handling
- **Domain-specific LLM integration** — prescreen, structured extraction, goal-setting, narration as distinct LLM roles
- **Data engineering at scale** — 10+ API sources, 82 RSS feeds, ChromaDB vector store, parallel ingestion

Directly transferable to: legal tech, OSINT tooling, compliance/AML, investigative journalism platforms, humanitarian intelligence systems.
