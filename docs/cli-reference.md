# CLI Reference

All commands follow the pattern `ai4saw <group> <command> [OPTIONS] [ARGS]`.

Run `ai4saw --help` or `ai4saw <group> --help` for live help text.

## `ai4saw info`

Show current configuration and status of all data artefacts.

```bash
ai4saw info
```

---

## `ai4saw ingest`

### `ingest file`

```bash
ai4saw ingest file SOURCE [OPTIONS]

Arguments:
  SOURCE  Path to document or URL

Options:
  --doc-type      report|news|legal|grey_literature  [default: report]
  --language      ISO 639-1 code                     [default: en]
  --geography     Geographic tag (e.g. conflict-region)
  --date-published ISO date (e.g. YYYY-MM-DD)
```

### `ingest corpus`

```bash
ai4saw ingest corpus CORPUS_DIR [OPTIONS]

Arguments:
  CORPUS_DIR  Directory to ingest recursively

Options:
  --doc-type   report|news|legal|grey_literature  [default: report]
  --language   ISO 639-1 code                     [default: en]
  --geography  Geographic tag applied to all docs
```

---

## `ai4saw extract`

### `extract pipeline`

```bash
ai4saw extract pipeline [OPTIONS]

Options:
  --output-dir  Output directory    [default: ./output]
  --delay       Seconds between LLM calls [default: 0.25]
```

Runs NER + relation extraction + event classification on all indexed chunks. Outputs `ner_results.json`, `relation_results.json`, `event_results.json`.

### `extract resolve`

```bash
ai4saw extract resolve [OPTIONS]

Options:
  --ner-file          Path to ner_results.json  [default: output/ner_results.json]
  --cosine-threshold  0.0–1.0  [default: 0.88]
  --fuzzy-threshold   0.0–100  [default: 72.0]
  --output            Output path  [default: data/entity_registry.json]
```

### `extract ner`

```bash
ai4saw extract ner CHUNK_ID --text "raw text" [--output result.json]
```

Single-chunk NER for testing.

---

## `ai4saw graph`

### `graph build`

```bash
ai4saw graph build [OPTIONS]

Options:
  --relations-file  [default: output/relation_results.json]
  --registry-file   [default: data/entity_registry.json]
  --min-confidence  [default: 0.5]
  --output          [default: data/knowledge_graph.json]
```

### `graph query`

```bash
ai4saw graph query QUESTION [OPTIONS]

Arguments:
  QUESTION  Question or entity name to query

Options:
  --hops          Graph neighbourhood depth  [default: 2]
  --at            ISO date for temporal filtering (e.g. YYYY-MM-DD)
  --graph-file    [default: data/knowledge_graph.json]
  --combine-vector  Also run vector search and combine  [default: True]
```

### `graph agent`

```bash
ai4saw graph agent QUESTION [OPTIONS]

Arguments:
  QUESTION  Complex research question

Options:
  --max-iterations  Maximum tool calls  [default: 8]
  --output          Save AgentResponse as JSON
```

---

## `ai4saw query`

### `query ask`

```bash
ai4saw query ask QUESTION [OPTIONS]

Arguments:
  QUESTION  Natural language question

Options:
  --top-k  Chunks to retrieve  [default: 8]
  --top-n  Chunks after reranking  [default: 3]
```

---

## `ai4saw analyze`

### `analyze contradictions`

```bash
ai4saw analyze contradictions [OPTIONS]

Options:
  --events-file      [default: output/event_results.json]
  --relations-file   [default: output/relation_results.json]
  --min-confidence   LLM confidence threshold  [default: 0.65]
  --max-pairs        Cap on LLM calls  [default: 100]
  --output           [default: output/contradictions.json]
```

### `analyze network`

```bash
ai4saw analyze network [OPTIONS]

Options:
  --relations-file   [default: output/relation_results.json]
  --registry-file    Optional entity registry for canonicalisation
  --min-confidence   [default: 0.5]
  --output           [default: output/network.json]
  --gexf             Also export GEXF for Gephi  [flag]
```

---

## `ai4saw discover`

### `discover run`

```bash
ai4saw discover run [ENTITIES...] [OPTIONS]

Arguments:
  ENTITIES  Entity names to search for (space-separated)

Options:
  --from-registry     Use top entities from entity registry  [flag]
  --top-n             Top N entities from registry  [default: 10]
  --label             Filter registry by label (repeatable, e.g. --label LOCATION)
  --per-entity-limit  Max results per entity per API  [default: 10]
  --output            [default: output/discovered_documents.json]
```

---

## `ai4saw export`

### `export all`

```bash
ai4saw export all [OPTIONS]

Options:
  --ner-file        [default: output/ner_results.json]
  --relations-file  [default: output/relation_results.json]
  --events-file     [default: output/event_results.json]
```

Outputs: `events.geojson`, `relations.json`, `entities.json`, `silences.json`, `corpus_stats.json`

---

## `ai4saw eval`

### `eval ner`

```bash
ai4saw eval ner GOLD_FILE [--output result.json]
```

### `eval rag`

```bash
ai4saw eval rag QUESTIONS_FILE [--output result.json]
```

Requires `--extra eval`.

### `eval judge`

```bash
ai4saw eval judge [OPTIONS]

Options:
  --ner-file        [default: output/ner_results.json]
  --relations-file  [default: output/relation_results.json]
  --events-file     [default: output/event_results.json]
  --sample          Number of chunks to evaluate  [default: 20]
  --output          [default: eval/results/judge_report.json]
  --seed            Random seed  [default: 42]
```
