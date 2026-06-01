"""NER benchmark — precision, recall, F1 per entity type against hand-labelled gold standard.

Usage:
  python eval/ner_benchmark.py --gold eval/testdata/ner_gold.json

Gold standard format (ner_gold.json):
  [
    {
      "chunk_id": "abc123",
      "chunk_text": "...",
      "entities": [
        {"text": "International Tribunal", "label": "ORG"},
        ...
      ]
    }
  ]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ai4saw.extraction.ner import extract_entities
from ai4saw.core.models import Entity

app = typer.Typer(help="Evaluate NER precision/recall/F1 against gold labels.")
console = Console()


def _normalise_entity(text: str, label: str) -> str:
    return f"{label}::{text.lower().strip()}"


def compute_metrics(
    gold_entities: list[dict],
    pred_entities: list[Entity],
) -> dict[str, dict[str, float]]:
    """Return per-label precision, recall, F1."""
    gold_set: dict[str, set] = defaultdict(set)
    pred_set: dict[str, set] = defaultdict(set)

    for g in gold_entities:
        gold_set[g["label"]].add(g["text"].lower().strip())

    for p in pred_entities:
        pred_set[p.label].add(p.text.lower().strip())

    all_labels = set(gold_set.keys()) | set(pred_set.keys())
    metrics: dict[str, dict[str, float]] = {}

    for label in sorted(all_labels):
        gold = gold_set[label]
        pred = pred_set[label]
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return metrics


def run_benchmark(
    gold_path: Path,
    output_path: Optional[Path] = None,
    delay: float = 0.25,
) -> dict:
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    all_metrics: dict[str, list] = defaultdict(list)
    chunk_results = []

    for item in gold_data:
        chunk_id = item["chunk_id"]
        chunk_text = item["chunk_text"]
        gold_entities = item["entities"]

        try:
            result = extract_entities(chunk_text, chunk_id)
            pred_entities = result.entities
        except Exception as exc:
            logger.error(f"Extraction failed for chunk {chunk_id}: {exc}")
            pred_entities = []

        metrics = compute_metrics(gold_entities, pred_entities)
        chunk_results.append({"chunk_id": chunk_id, "metrics": metrics})

        for label, m in metrics.items():
            all_metrics[label].append(m)

    # Macro-average across chunks
    summary: dict[str, dict] = {}
    for label, ms in all_metrics.items():
        summary[label] = {
            "precision": round(sum(m["precision"] for m in ms) / len(ms), 4),
            "recall": round(sum(m["recall"] for m in ms) / len(ms), 4),
            "f1": round(sum(m["f1"] for m in ms) / len(ms), 4),
            "chunk_count": len(ms),
        }

    report = {
        "gold_file": str(gold_path),
        "chunk_count": len(gold_data),
        "per_label": summary,
        "per_chunk": chunk_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Benchmark results written to {output_path}")

    return report


def _display_report(report: dict) -> None:
    table = Table(title="NER Benchmark Results", show_header=True)
    table.add_column("Label", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold")
    table.add_column("Chunks", justify="right")

    for label, m in sorted(report["per_label"].items()):
        table.add_row(
            label,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            str(m["chunk_count"]),
        )

    console.print(table)
    console.print(f"\n[dim]Evaluated on {report['chunk_count']} chunks.[/dim]")


@app.command()
def benchmark(
    gold: Path = typer.Argument(..., help="Path to gold-standard JSON file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON results to this path"
    ),
) -> None:
    """Run NER precision/recall/F1 benchmark against gold labels."""
    if not gold.exists():
        typer.echo(f"Gold file not found: {gold}", err=True)
        raise typer.Exit(1)

    output = output or Path("eval/results/ner_benchmark.json")
    report = run_benchmark(gold, output_path=output)
    _display_report(report)


if __name__ == "__main__":
    app()
