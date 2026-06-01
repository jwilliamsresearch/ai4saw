# Evaluation

AI4SAW has three evaluation tracks: NER benchmark, RAG quality (RAGAS), and LLM-as-Judge automated scoring.

## NER Benchmark

Precision, recall, and F1 per entity type against hand-labelled gold chunks.

```bash
ai4saw eval ner eval/testdata/ner_gold.json
```

Gold standard format (`eval/testdata/ner_gold.json`):

```json
[
  {
    "chunk_id": "gold_001",
    "chunk_text": "...",
    "entities": [
      {"text": "International Tribunal", "label": "ORG"},
      {"text": "Commander Alpha", "label": "PERSON"}
    ]
  }
]
```

Output: `eval/results/ner_benchmark.json`

**Expanding the gold set:** The current gold set has 5 hand-labelled chunks. For publishable benchmarks, label 200 chunks spanning both conflict-region and conflict-region domains. The evaluation script handles any number of gold chunks.

## RAG Evaluation (RAGAS)

Measures four dimensions of RAG quality:

| Metric | Measures |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved chunks relevant to the question? |
| `context_recall` | Are all relevant chunks retrieved? (requires ground truth) |

```bash
# Requires --extra eval (PyTorch, ARM/Linux only)
ai4saw eval rag eval/testdata/rag_questions.json
```

Question file format:

```json
[
  {
    "question": "What forms of forced labour were documented in Bosnian camps?",
    "ground_truth": "..."   // optional, needed for context_recall
  }
]
```

Output: `eval/results/rag_eval.json`

## LLM-as-Judge

Automated extraction quality scoring without hand labels. Uses a frontier model to assess NER, relation, and event extraction quality on a random sample.

```bash
ai4saw eval judge --sample 20 --output eval/results/judge_report.json
```

Requires `JUDGE_MODEL` set in `.env`. See [LLM-as-Judge](advanced/llm-judge.md).

## Zero-shot vs few-shot comparison

Event classification uses zero-shot first, falling back to few-shot when confidence < 0.6. The strategy used per chunk is logged. To compare:

```bash
# Check the event results for strategy distribution
python -c "
import json
events = json.load(open('output/event_results.json'))
# Strategy is logged per-chunk in logs/ai4saw.log
"
```

This comparison is a publishable finding — systematic measurement of zero-shot vs few-shot performance on conflict-domain text.

## Model comparison

To compare extraction quality across models:

1. Set `DEFAULT_MODEL=mistral` in `.env`, run `extract pipeline`, save results
2. Set `DEFAULT_MODEL=llama3`, re-run, save results  
3. Run `eval judge` on both result sets with the same seed
4. Compare aggregate scores

Commit all results to `eval/results/` for reproducibility.

## Results are committed

All evaluation results in `eval/results/` are committed to git. This makes all performance claims verifiable — a reviewer can inspect the exact scores rather than taking the paper's word for it.
