# Active Corpus Discovery

## The idea

The silence detector identifies what's missing from the corpus. This module tries to fill those gaps by searching external APIs for documents not yet ingested.

## APIs

Both APIs are **free and require no authentication**.

| API | Coverage | URL |
|---|---|---|
| **ReliefWeb** | UN and NGO humanitarian reports, situation assessments | `api.reliefweb.int` |
| **GDELT** | Global news in 65+ languages, real-time | `api.gdeltproject.org` |

## Usage

```bash
# Search by entity names
ai4saw discover run "El Geneina" "Masalit" "Rapid Support Forces"

# Auto-discover from entity registry (top-N by frequency)
ai4saw discover run --from-registry --top-n 10

# Filter by entity label
ai4saw discover run --from-registry --label LOCATION --top-n 5

# Silence-driven discovery (highest yield)
ai4saw discover run "Foča" "Prijedor" "Brčko"
```

## Deduplication

Results are checked against `corpus/sources.csv`. Documents already registered (by URL) are excluded. The `new_documents` count in the output shows how many are genuinely new.

## Output

```json title="output/discovered_documents.json"
{
  "trigger_entities": ["El Geneina", "Masalit"],
  "query_count": 4,
  "new_documents": 12,
  "documents": [
    {
      "title": "Sudan: RSF Attacks on Masalit in El Geneina",
      "url": "https://reliefweb.int/report/sudan/...",
      "source": "reliefweb",
      "date": "2023-05-02",
      "relevance_score": 0.82,
      "trigger_entity": "El Geneina"
    }
  ]
}
```

## Workflow

Discovery output is for **researcher review only**. The pipeline never ingests automatically.

```
discover run → review output/discovered_documents.json
             → add selected URLs to corpus/sources.csv (with licence)
             → ai4saw ingest file <url>
```

## Silence-driven discovery

The most effective discovery strategy is to query locations identified by the silence detector:

```bash
# Get silence candidates first
# (run after graph build)

# Then discover for the silenced locations
ai4saw discover run "Nyala" "Zalingei" "Geneina" --per-entity-limit 15
```

High-silence, high-conflict locations are where the corpus gap is largest and the potential new evidence is most valuable.
