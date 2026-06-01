# Entity Resolution

## The problem

NER processes each chunk independently. A single actor may appear across the corpus as:

- "the Armed-Group-Beta"
- "Rapid Support Forces"
- "the Khartoum paramilitaries"
- "Armed-Group-Beta fighters"

Without resolution, these are four separate entities. Network analysis, silence detection, and any aggregation across documents produce fragmented results.

## How it works

```bash
ai4saw extract resolve
```

**Algorithm:**

1. Collect all `(text, label)` mentions from NER results, counting frequency per text.
2. Batch-embed all unique entity texts using the configured embedder.
3. Group by label — PERSON is never merged with ORG.
4. Within each label group, compute pairwise cosine similarity via numpy.
5. Also compute fuzzy string ratio (stdlib `difflib.SequenceMatcher`).
6. **Union-find clustering:** merge two mentions if `cosine_similarity ≥ threshold OR fuzzy_ratio ≥ fuzzy_threshold`.
7. Each cluster → one `ResolvedEntity`: canonical = most frequent form, aliases = others.

The dual threshold (cosine + fuzzy) is deliberate. Embedding similarity alone conflates different organisations in the same domain ("International Tribunal" and "ICTR" have high semantic similarity but are distinct entities). String similarity alone misses "Armed-Group-Beta" ↔ "Rapid Support Forces". Together they are much more precise.

## Configuration

```bash
ai4saw extract resolve \
  --cosine-threshold 0.88 \    # default: conservative
  --fuzzy-threshold 72.0 \     # default: catches abbreviations
  --output data/entity_registry.json
```

**Tuning guidance:**

| Scenario | Adjustment |
|---|---|
| Too many false merges | Raise `--cosine-threshold` (0.92+) |
| Missing obvious aliases | Lower `--fuzzy-threshold` (65) |
| Merging different orgs in same domain | Raise both thresholds |

## Output

```json title="data/entity_registry.json"
{
  "entities": [
    {
      "canonical_id": "a3f1b2c4d5e6",
      "canonical_text": "Rapid Support Forces",
      "label": "ORG",
      "aliases": ["Armed-Group-Beta", "the Armed-Group-Beta", "Khartoum paramilitaries"],
      "occurrence_count": 47,
      "source_chunks": ["abc123", "def456"],
      "mean_confidence": 0.94
    }
  ],
  "total_mentions": 312,
  "unique_texts_before": 89,
  "resolved_count": 41
}
```

## Downstream effects

Entity resolution is a prerequisite for:

- **Knowledge graph build** — nodes are resolved entities
- **Network analysis** — accurate actor counts and centrality
- **Corpus discovery** — queries use canonical entity names
- **MCP `find_entity` tool** — alias lookup
