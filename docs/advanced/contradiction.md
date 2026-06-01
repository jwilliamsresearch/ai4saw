# Contradiction Detection

## Why contradictions matter

In conflict research, contradictions between sources are not errors to discard — they are often the most analytically important signal.

Two sources giving different death tolls for the same massacre may reflect:
- Different witness perspectives
- Propaganda from one or both parties
- Selective documentation by different organisations
- Genuinely different incidents being conflated

The pipeline surfaces contradictions for researcher review. It does not resolve them.

## How it works

```bash
ai4saw analyze contradictions --output output/contradictions.json
```

**Two-pass approach:**

**Pass 1 — Candidate generation (cheap, no LLM):**

- Group `EventResult`s by `(location, year)`. Flag pairs with different event types for the same location/year.
- Group `RelationResult`s by `(subject, object)`. Flag pairs where the same actor acted on the same target with a different predicate across different source chunks.

This generates candidate pairs without calling the LLM — keeping costs manageable on large corpora.

**Pass 2 — LLM verification (per candidate pair):**

For each candidate pair, retrieve the source chunk text and call the LLM with `prompts/contradiction_check.yaml`. Only pairs where the LLM confirms a contradiction above the confidence threshold enter the report.

## Contradiction types

| Type | Example |
|---|---|
| `factual` | Source A: "200 killed"; Source B: "fewer than 50 killed" |
| `temporal` | Source A: "a specific date"; Source B: "August 1995" |
| `attribution` | Source A: "ordered by Commander Alpha�"; Source B: "local commanders acting alone" |
| `numerical` | Source A: "detained for 6 months"; Source B: "detained for 3 weeks" |

## Configuration

```bash
ai4saw analyze contradictions \
  --min-confidence 0.65 \   # LLM confidence threshold (default 0.65)
  --max-pairs 100 \          # cap on LLM calls to control cost
  --output output/contradictions.json
```

## Output

```json title="output/contradictions.json"
{
  "pairs": [
    {
      "chunk_id_a": "abc123",
      "chunk_id_b": "def456",
      "source_a": "HRW-1992-report.pdf",
      "source_b": "SerbiaMOD-1992-statement.pdf",
      "contradiction_type": "numerical",
      "confidence": 0.89,
      "explanation": "HRW reports 3,000 detained at Manjača; Serbian MoD states capacity was 300..."
    }
  ],
  "total_chunks_analysed": 847,
  "candidate_pairs_assessed": 23,
  "high_confidence_count": 4
}
```

## Programmatic use

```python
from ai4saw.synthesis.contradiction import detect_contradictions

report = detect_contradictions(
    event_results,
    relation_results,
    llm_confidence_threshold=0.65,
    max_pairs_to_assess=100,
)

for pair in report.pairs:
    print(f"{pair.contradiction_type}: {pair.source_a} vs {pair.source_b}")
    print(f"  Confidence: {pair.confidence:.2f}")
    print(f"  {pair.explanation}")
```
