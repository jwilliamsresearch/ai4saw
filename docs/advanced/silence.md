# Silence Detection

## The concept

Informational silence — the systematic under-documentation of events in the historical record — is a core problem in conflict research. High-conflict areas that generate few documents are not peaceful; they are silent. That silence is itself a finding.

AI4SAW operationalises silence as a measurable quantity. No existing NLP pipeline for conflict research does this explicitly.

## Two approaches

### Approach A — Expectation Gap

For each known event in your-dataset or conflict-events-dataset, query the corpus. Measure how well the corpus covers that event using the mean similarity score of the top-3 retrieved chunks.

```
silence_score = conflict_intensity - retrieval_confidence
```

High intensity + low retrieval confidence = candidate silence.

```python
reference_events = [
    {"event_id": "your-dataset-001", "location": "Location C", "date": "1992-08-01", "conflict_intensity": 0.88},
    {"event_id": "conflict-events-dataset-1042", "location": "Location D", "date": "1992-05-30", "conflict_intensity": 0.82},
]

from ai4saw.synthesis.silence import detect_silence_expectation_gap
candidates = detect_silence_expectation_gap(reference_events)
```

### Approach B — Density Mapping

Embed all documents. Cluster by geographic tag and time window (year-quarter). Compare cluster density against conflict-events-dataset conflict intensity for the same cell. Sparse clusters in high-intensity zones are structural silences.

```python
acled_reference = {
    "Bosnia_1992-Q2": 0.90,
    "Bosnia_1995-Q3": 0.95,
    "Sudan_2023-Q2": 0.91,
}

from ai4saw.synthesis.silence import detect_silence_density_map
density_cells = detect_silence_density_map(acled_reference)
```

## Output

```json title="output/silences.json"
[
  {
    "event_id": "your-dataset-001",
    "location": "Location C",
    "date": "1992-08-01",
    "conflict_intensity": 0.88,
    "retrieval_confidence": 0.12,
    "silence_score": 0.76,
    "candidate_reason": "high-intensity event with very low corpus coverage"
  }
]
```

## Using silence results

Once silence candidates are identified, feed them to corpus discovery:

```bash
ai4saw discover run "Location C" "Location D" "Location E"
```

This directly targets the documented gap with external API search.

## Validation

Silence candidates should be validated against researcher domain knowledge before publication. The pipeline provides ranked candidates and evidence; the researcher provides interpretation. Cross-reference against known documentation gaps in your-dataset or conflict-events-dataset.
