# LLM-as-Judge

## Why this matters

Hand-labelling 200 chunks is expensive and slow. The LLM-as-Judge approach uses a frontier model to score extraction quality at scale, providing automated feedback without requiring ground truth labels.

This is not a replacement for the gold-standard benchmark — it is a complement. Use it for:

- Development feedback loops (does changing the prompt improve quality?)
- Comparing model performance across providers
- Detecting regressions when prompt templates change
- Early quality signal before the full benchmark is ready

The methodology itself is publishable: automated evaluation of conflict NLP extraction is an open research problem.

## Configuration

The judge should be a *different* model from the extraction model. Self-judging introduces circular bias — the model that produced the extraction is unlikely to reliably critique it.

```bash title=".env"
# Extraction (bulk, cheap)
PROVIDER=ollama
DEFAULT_MODEL=mistral

# Judge (frontier, small sample)
JUDGE_MODEL=anthropic/claude-3-5-sonnet    # via OpenRouter
```

If `JUDGE_MODEL` is unset, the judge uses `DEFAULT_MODEL` with a bias warning.

## Run

```bash
ai4saw eval judge \
  --sample 20 \                # number of chunks to evaluate
  --output eval/results/judge_report.json
```

For final benchmarks, use 50+ samples. 20 is appropriate for development.

## What is assessed

Three dimensions, scored 0.0–1.0:

| Dimension | Assesses |
|---|---|
| `ner_score` | Entity accuracy: correct labels, no hallucinations, no missed entities |
| `relation_score` | Triple groundedness: every triple directly supported by source text evidence |
| `event_score` | Event classification: correct type, accurate date/location/perpetrator/victim |

`overall_score` = weighted mean (NER × 0.3, relation × 0.4, event × 0.3). Relations are weighted highest because hallucinated triples are the most harmful output — they could enter the knowledge graph as false facts.

## Output

```json title="eval/results/judge_report.json"
{
  "aggregate": {
    "ner": 0.84,
    "relation": 0.71,
    "event": 0.79,
    "overall": 0.77
  },
  "sample_size": 20,
  "model_used": "anthropic/claude-3-5-sonnet",
  "scores": [
    {
      "chunk_id": "abc123",
      "source_filename": "icty-krstic.pdf",
      "ner_score": 0.9,
      "relation_score": 0.6,
      "event_score": 0.85,
      "overall_score": 0.75,
      "issues": ["Relation 'commanded' has no evidence span", "Missed entity: International Tribunal"],
      "explanation": "..."
    }
  ]
}
```

## Interpreting scores

| Score range | Interpretation |
|---|---|
| 0.85 – 1.0 | Good — publishable quality |
| 0.70 – 0.85 | Acceptable — minor issues, usable for research |
| 0.50 – 0.70 | Needs improvement — review prompt templates |
| < 0.50 | Poor — likely model or prompt mismatch |

## Most common issues

The judge report aggregates issues across all chunks. The most common issues are the highest-yield targets for prompt improvement. Typical patterns:

- **"Relation has no evidence span"** → CoT prompt not forcing evidence grounding
- **"Missed entity: FACILITY"** → Few-shot examples don't cover this entity type well
- **"Event type `no_event` but text clearly describes forced labour"** → Threshold too high, zero-shot underperforming
