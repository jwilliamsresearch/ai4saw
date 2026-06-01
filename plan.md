# AI4SAW: Setup to Hero

A complete walkthrough from a fresh clone to a working, demonstrable intelligence extraction pipeline on conflict-related corpora.

---

## Evaluation Criteria

Before anything else — what does "hero" mean for this project?

| Signal | How to measure |
|---|---|
| Extraction quality | NER F1 ≥ 0.70 against gold labels; events classification accuracy |
| RAG faithfulness | RAGAS faithfulness ≥ 0.75; answer relevance ≥ 0.70 |
| Knowledge graph coverage | ≥ 3 entity types, temporal edges populated, k-hop queries return results |
| Agent reasoning | Multi-hop questions answered with source citations (not hallucinations) |
| Silence & contradiction | At least one silence gap and one contradiction surfaced from corpus |
| Discovery pipeline | ReliefWeb/GDELT returns new candidate documents |
| End-to-end runtime | Full pipeline on a 10-document corpus completes in < 30 min on CPU |

Run `ai4saw eval ner eval/testdata/ner_gold.json` and `ai4saw eval rag eval/testdata/rag_questions.json` to score yourself at any point.

---

## Phase 0 — Prerequisites (Day 0, ~1 hour)

### 0.1 System requirements

- Python 3.12 (pinned in `.python-version`)
- [uv](https://github.com/astral-sh/uv) — `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.ai) — local LLM runner (free, no API key needed for default path)
- Git

### 0.2 Clone and install

```bash
git clone <repo-url> ai4saw
cd ai4saw
uv sync                          # installs all core deps into .venv
uv sync --extra rerank           # cross-encoder re-ranking (recommended)
uv sync --extra eval             # RAGAS + benchmark deps
```

### 0.3 Pull local models

```bash
ollama pull mistral              # extraction + reasoning (~4 GB)
ollama pull nomic-embed-text     # embeddings (~270 MB)
```

> **Alternative (cloud):** Set `PROVIDER=openrouter` and `OPENROUTER_API_KEY=sk-...` in `.env` to use any frontier model instead. Costs ~$0.10–$2.00 for a 10-doc pilot run depending on model.

### 0.4 Configure environment

```bash
cp .env.example .env
```

Edit `.env` — minimum required fields:

```dotenv
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./data/chroma
LOG_LEVEL=INFO
```

### 0.5 Smoke test

```bash
ai4saw info
```

Expected: green table showing provider, model, artefact paths. If any path shows red, check `.env`.

**Gate:** `ai4saw info` exits 0 with no errors before proceeding.

---

## Phase 1 — Corpus Preparation (Day 0–1, ~30 min automated)

### 1.1 Automated fetch (ReliefWeb + GDELT)

For sources available via open APIs, Phase 1 is now a single command. The `discover fetch` pipeline discovers, downloads, registers in `corpus/sources.csv`, and ingests in one step:

```bash
# Bosnia pilot
ai4saw discover fetch "Srebrenica" "Mladic" "Bosnian Serb Army" "VRS" \
  --geography Bosnia \
  --max-docs 25 \
  --min-relevance 0.5

# Sudan/Darfur pilot
ai4saw discover fetch "El Geneina" "Masalit" "Rapid Support Forces" "RSF" "Darfur" \
  --geography Sudan \
  --max-docs 25 \
  --min-relevance 0.5
```

Preview candidates first with `--dry-run`. Confirm with `--yes` to skip the interactive prompt.

Sources fetched automatically:
- **ReliefWeb** — UN and NGO humanitarian reports (PDFs and HTML)
- **GDELT** — global news articles in 65+ languages (HTML)

### 1.2 Manual documents (ICTY, ICC, HRW PDFs)

Sources without open APIs require manual download. These are high-value but need custom scrapers (deferred post-v0.1):

- **ICTY/IRMCT** — trial judgements, indictments, evidence transcripts (icty.org)
- **ICC** — situation documents, warrants, judgements (icc-cpi.int)
- **HRW** — country reports 1995–2007 (hrw.org/reports)
- **OHCHR** — special rapporteur reports, treaty body documents

For these, download PDFs manually, place in `corpus/`, then register and ingest:

```bash
# Register in sources.csv (add a row manually):
# filename,source_url,date_accessed,licence,geography,notes
# icty_krstic_judgement.pdf,https://icty.org/...,2026-06-01,public,Bosnia,ICTY trial judgement

# Then ingest:
ai4saw ingest file corpus/icty_krstic_judgement.pdf \
  --doc-type report --geography Bosnia --date-published 2001-08-02
```

> **Why geography matters:** The pipeline uses geography tags to weight entity resolution and silence detection. Set it correctly on ingest; it's expensive to change later.

### 1.3 Verify corpus size

```bash
ai4saw info
```

Target: **≥ 200 chunks** in ChromaDB before moving to extraction. For a 10-document corpus (mix of PDFs and HTML), expect 300–800 chunks.

---

## Phase 2 — Ingestion (Day 1, ~30 min)

### 2.1 Ingest corpus

```bash
ai4saw ingest corpus ./corpus --geography Bosnia
```

This:
- Loads every file in `./corpus/` (PDF, DOCX, HTML, TXT)
- Chunks into ~1000-token windows with 200-token overlap
- Embeds chunks and upserts into ChromaDB (idempotent — safe to re-run)

Watch for:
- `Loaded N chunks` — should be hundreds to thousands for a 10-doc corpus
- Any `WARNING` about failed loaders (usually malformed PDFs — exclude or re-export)

### 2.2 Verify

```bash
ai4saw info
```

`Chunks in ChromaDB` should now show a non-zero count.

**Gate:** At least 200 chunks in ChromaDB before extraction.

---

## Phase 3 — Extraction Pipeline (Day 1–2, ~2–4 hours depending on hardware)

This is the slowest phase. Each chunk goes through three LLM calls (NER → relations → events). On Mistral-7B via Ollama on CPU, expect ~30s per chunk. For 500 chunks, that's ~4 hours. **Run overnight or use OpenRouter for speed.**

### 3.1 Run extraction

```bash
ai4saw extract pipeline
```

Outputs written to:
- `output/ner_results.json`
- `output/relation_results.json`
- `output/event_results.json`

Watch for JSON parse failures — the pipeline retries once automatically, then logs a skip. A skip rate > 20% suggests the model needs a prompt tweak or a larger context window.

### 3.2 Entity resolution

After extraction completes, deduplicate entities across documents:

```bash
ai4saw extract resolve
```

This runs cosine similarity + fuzzy string matching via union-find clustering to merge e.g. `"Ratko Mladic"`, `"General Mladic"`, `"R. Mladic"` into one canonical entity.

Output: `data/entity_registry.json`

Tune with `ENTITY_SIM_THRESHOLD` in `.env` (default 0.85). Lower = more aggressive merging.

**Gate:** `output/ner_results.json` exists and is non-empty. Entity registry has > 20 unique entities.

---

## Phase 4 — Knowledge Graph (Day 2, ~30 min)

### 4.1 Build the graph

```bash
ai4saw graph build
```

Constructs a temporal knowledge graph from:
- Resolved entities (nodes)
- Extracted relations (edges with `valid_from` / `valid_to`)

Output: `data/knowledge_graph.json`

### 4.2 Smoke-test graph queries

```bash
# Standard GraphRAG — k-hop subgraph + vector search
ai4saw graph query "Who commanded Bosnian Serb forces at Srebrenica?"

# Temporal filter — state of relations at a specific date
ai4saw graph query "Who was in command?" --at 1995-07-11
```

Expected: answer with source citations and confidence. If the graph returns nothing, increase `--hops` (default 2) or check that entity resolution produced nodes.

### 4.3 Multi-hop agent

For complex questions that require chaining multiple evidence sources:

```bash
ai4saw graph agent "What is the chain of command between Mladic and the execution units, and what evidence links them?"
```

The LangGraph ReAct agent uses three tools iteratively: vector search, graph query, entity lookup. Expect 3–6 tool calls before a final answer.

**Gate:** At least one graph query returns a cited answer before proceeding.

---

## Phase 5 — Advanced Analysis (Day 2–3)

### 5.1 Contradiction detection

```bash
ai4saw analyze contradictions
```

Two-pass approach:
1. Cheap candidate generation (embedding similarity between conflicting claim pairs)
2. LLM verification pass (confirms genuine contradiction vs. complementary detail)

Output: `output/contradictions.json` — ranked by confidence. In conflict corpora, contradiction is analytically meaningful (contested narratives, cover-up vs. disclosure).

### 5.2 Command network analysis

```bash
ai4saw analyze network --gexf
```

Extracts perpetrator command networks from relation triples. Computes betweenness centrality (key brokers) and Louvain community detection (operational clusters).

Outputs:
- `output/network.json` — centrality scores
- `output/network.gexf` — Gephi-importable network file for visualization

### 5.3 Silence detection

```bash
# Expectation-gap: what entities/events are absent relative to corpus expectations?
# Density-map: where are coverage gaps in the document space?
```

Silence detection runs automatically as part of `export all`. The output flags:
- Expected event types with zero instances
- Entities referenced in relations but absent from NER results
- Geographic areas mentioned but with low information density

---

## Phase 6 — Corpus Discovery (Day 3)

Expand the corpus using top entities from your registry:

```bash
# Discover using specific entity names
ai4saw discover run "Srebrenica" "Mladic" "VRS"

# Auto-discover using top entities from registry
ai4saw discover run --from-registry
```

Queries ReliefWeb (humanitarian reports) and GDELT (global news) APIs. Results written to `output/discovered_documents.json`.

**Do not auto-ingest.** Review discovered documents for relevance and licence before adding to `corpus/sources.csv` and running `ingest file` on them.

---

## Phase 7 — Export & Evaluation (Day 3)

### 7.1 Export all artefacts

```bash
ai4saw export all
```

Produces:
- `output/events.geojson` — geocoded events (if geocoder is wired up; otherwise coordinates stub to None)
- `output/corpus_stats.json` — coverage metrics

### 7.2 Evaluate extraction quality

```bash
# NER precision/recall/F1 against gold labels
ai4saw eval ner eval/testdata/ner_gold.json

# RAGAS RAG quality (faithfulness, answer relevance, context precision/recall)
ai4saw eval rag eval/testdata/rag_questions.json

# LLM-as-Judge (frontier model grades extraction quality without hand labels)
ai4saw eval judge
```

Target thresholds:
- NER F1 ≥ 0.70
- RAGAS faithfulness ≥ 0.75
- RAGAS answer relevance ≥ 0.70

If below threshold: tune prompt YAML files in `prompts/`, adjust `ENTITY_SIM_THRESHOLD`, or switch to a larger model via `PROVIDER=openrouter`.

---

## Phase 8 — Hero Demo (Day 3–4)

A complete demonstration sequence that shows the full pipeline capability:

### 8.1 The demo script

```bash
# 1. Show corpus status
ai4saw info

# 2. Ask a factual question with vector RAG
ai4saw query ask "What methods of forced labour were documented in Darfur between 2003 and 2005?"

# 3. Ask a command-chain question with GraphRAG
ai4saw graph query "Who gave orders for population displacement in Eastern Bosnia in 1995?" --at 1995-07-15

# 4. Ask a multi-hop question with the agent
ai4saw graph agent "What is the chain of command linking senior RSF leadership to documented atrocities in El Geneina?"

# 5. Show a contradiction
cat output/contradictions.json | head -50

# 6. Show the command network centrality
cat output/network.json | python -c "import json,sys; d=json.load(sys.stdin); print(sorted(d['centrality'].items(), key=lambda x:-x[1])[:10])"

# 7. Open network.gexf in Gephi for visual impact
```

### 8.2 What makes it impressive

- **Citation trail**: Every answer cites chunk IDs, source documents, page numbers. Not a chatbot — an evidence pipeline.
- **Temporal precision**: `--at 1995-07-11` narrows the knowledge graph to what was true on the day of the Srebrenica massacre.
- **Contested narratives surfaced, not suppressed**: Contradictions between witness accounts and official records are explicitly ranked and reported.
- **Silence is evidence**: The pipeline can tell you what *isn't* documented — a research-grade finding, not a retrieval failure.
- **Network visualization**: Gephi renders the command structure. Betweenness centrality identifies key brokers that may not appear prominently in raw text.

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `ai4saw info` shows red artefact paths | ChromaDB not initialized | Run `ai4saw ingest corpus ./corpus` first |
| JSON parse failures > 20% | Model context too short or prompt too complex | Switch to `mistral:7b-instruct` or use OpenRouter with a larger model |
| Entity resolution merges wrong entities | Similarity threshold too low | Increase `ENTITY_SIM_THRESHOLD` to 0.90 |
| Graph queries return empty | No relations extracted, or graph not built | Check `output/relation_results.json`, re-run `ai4saw graph build` |
| Agent loops forever | LangGraph max iterations hit | Set `AGENT_MAX_STEPS=10` in `.env` |
| RAGAS score very low | RAG retrieving irrelevant chunks | Tune `MMR_FETCH_K` and `MMR_LAMBDA` in `.env`; enable re-ranking |
| Slow extraction on CPU | Mistral-7B is large for CPU | Use `--max-chunks 100` to test on a subset; or switch to OpenRouter |

---

## Known Gaps (Post-v0.1 Work)

- **Geocoding**: `ai4saw/synthesis/geocoder.py` is a stub. GeoJSON events export latitude/longitude as `null` until this is implemented. Options: Nominatim (free, slow), GeoNames (free API key), or few-shot LLM geocoder.
- **Multilingual**: Framework exists (`language` field in `ChunkMetadata`), but extraction prompts are English-only. Bosnian/Arabic sources need translation pre-processing or multilingual models.
- **CDISaW write-back**: Schema alignment between AI4SAW's entity/event model and the CDISaW dataset format is deferred post-v0.1.
- **Unit tests**: `eval/` covers benchmarking, not unit tests. No `test_*.py` files exist. Add if contributing to shared research infrastructure.

---

## Timeline Summary

| Day | Phase | Deliverable |
|---|---|---|
| 0 | Prerequisites + smoke test | `ai4saw info` green |
| 0–1 | Corpus preparation | `discover fetch` run; `corpus/sources.csv` auto-populated; manual PDFs registered |
| 1 | Ingestion | ≥ 200 chunks in ChromaDB |
| 1–2 | Extraction (overnight if CPU) | `ner_results.json`, `relation_results.json`, `event_results.json` |
| 2 | Entity resolution + graph build | `entity_registry.json`, `knowledge_graph.json` |
| 2 | Graph queries + agent | Cited answers on factual and command-chain questions |
| 2–3 | Advanced analysis | `contradictions.json`, `network.gexf` |
| 3 | Discovery + export + eval | RAGAS score, NER F1, `events.geojson` |
| 3–4 | Hero demo | Full pipeline walkthrough, Gephi network, live agent queries |
