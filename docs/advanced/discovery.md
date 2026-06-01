# Active Corpus Discovery

## The idea

The silence detector identifies what's missing from the corpus. This module fills those gaps by querying six free, no-registration APIs for documents not yet ingested.

## APIs

All six sources are **free and require no authentication**.

| API | Coverage | Rate limit |
|---|---|---|
| **OpenAlex** | Open-access academic papers and working papers (200M+ works) | Generous; use `CONTACT_EMAIL` for polite pool |
| **Semantic Scholar** | AI-curated academic index — distinct corpus from OpenAlex | 100 req/5 min unauthenticated |
| **arXiv** | Preprints in political science, peace studies, economics | Generous |
| **Internet Archive** | NGO reports, historical news, HRW/Amnesty PDFs, grey literature | Generous |
| ~~**UN Digital Library**~~ | ~~UN documents~~ | Deferred — API requires authentication |
| **GDELT** | Global news in 65+ languages, real-time | **Strict: 1 req/5s per IP** — batched |

> **OpenAlex polite pool:** Set `CONTACT_EMAIL=your@email.com` in `.env` for faster responses.

> **GDELT rate limiting:** GDELT enforces 1 request per 5 seconds per IP. The pipeline sends **one batched OR query** for all entities rather than one query per entity. Do not call the GDELT endpoint outside this module.

## Usage

```bash
# Search by entity names — queries all six sources
ai4saw discover run "Location Alpha" "Commander Alpha" "Armed-Group-Alpha"

# Auto-discover from entity registry (top-N by frequency)
ai4saw discover run --from-registry --top-n 10

# Filter by entity label
ai4saw discover run --from-registry --label LOCATION --top-n 5

# Silence-driven discovery (highest yield)
ai4saw discover run "Location C" "Location D" "Location E"
```

## Deduplication

Results are checked against `corpus/sources.csv` by URL. Documents already registered are excluded. The `new_documents` count shows how many are genuinely new.

## Output

```json title="output/discovered_documents.json"
{
  "trigger_entities": ["Location Alpha", "Commander Alpha"],
  "query_count": 11,
  "new_documents": 47,
  "documents": [
    {
      "title": "Location Alpha: A 'Safe' Area",
      "url": "https://archive.org/details/srebrenica-report",
      "source": "internetarchive",
      "date": "1995-11-01",
      "relevance_score": 0.80,
      "trigger_entity": "Location Alpha"
    }
  ]
}
```

## Workflow

### Passive (review mode)

`discover run` is for researcher review only — it never ingests automatically.

```
discover run → review output/discovered_documents.json
             → add selected URLs to corpus/sources.csv (with licence)
             → ai4saw ingest file <url>
```

### Active (automated fetch)

`discover fetch` extends this into a full pipeline: discover → download → register → ingest in one command. See [Automated Corpus Fetch](corpus-fetch.md).

```bash
ai4saw discover fetch "Location Alpha" "Commander Alpha" --geography conflict-region --max-docs 30
```

## Silence-driven discovery

The most effective strategy is to query locations identified by the silence detector:

```bash
# Get silence candidates first (run after graph build)

# Then target the gaps directly
ai4saw discover run "Location F" "Location G" "Location Beta" --per-entity-limit 25
```

High-silence, high-conflict locations are where corpus gaps are largest and new evidence most valuable.
