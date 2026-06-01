"""RAG evaluation using RAGAS metrics.

Evaluates: faithfulness, answer relevance, context precision, context recall.

Usage:
  python eval/rag_eval.py --questions eval/testdata/rag_questions.json

Question file format (rag_questions.json):
  [
    {
      "question": "...",
      "ground_truth": "..."   // optional — needed for context_recall
    }
  ]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ai4saw.retrieval.qa import answer

app = typer.Typer(help="Evaluate RAG pipeline quality using RAGAS metrics.")
console = Console()


def _build_ragas_dataset(questions: list[dict]) -> "Dataset":  # type: ignore[name-defined]
    """Run the Q&A pipeline and collect results for RAGAS evaluation."""
    from datasets import Dataset

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in questions:
        q = item["question"]
        gt = item.get("ground_truth", "")

        logger.info(f"Running Q&A for: {q!r}")
        try:
            resp = answer(q)
            rows["question"].append(q)
            rows["answer"].append(resp.answer)
            rows["contexts"].append([s.source_filename for s in resp.sources])
            rows["ground_truth"].append(gt)
        except Exception as exc:
            logger.error(f"Q&A failed for question {q!r}: {exc}")

    return Dataset.from_dict(rows)


def run_ragas_eval(
    questions_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        raise ImportError(
            "ragas is required for RAG evaluation. Install with: uv add ragas"
        )

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    dataset = _build_ragas_dataset(questions)

    metrics = [faithfulness, answer_relevancy, context_precision]
    has_ground_truth = any(q.get("ground_truth") for q in questions)
    if has_ground_truth:
        metrics.append(context_recall)

    logger.info(f"Running RAGAS evaluation on {len(questions)} questions...")
    results = evaluate(dataset, metrics=metrics)

    report = {
        "question_count": len(questions),
        "metrics": {
            "faithfulness": round(float(results["faithfulness"]), 4),
            "answer_relevancy": round(float(results["answer_relevancy"]), 4),
            "context_precision": round(float(results["context_precision"]), 4),
        },
        "questions_file": str(questions_path),
    }

    if has_ground_truth:
        report["metrics"]["context_recall"] = round(
            float(results["context_recall"]), 4
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"RAGAS results written to {output_path}")

    return report


def _display_report(report: dict) -> None:
    table = Table(title="RAGAS Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold")

    for metric, score in report["metrics"].items():
        table.add_row(metric.replace("_", " ").title(), f"{score:.4f}")

    console.print(table)
    console.print(f"\n[dim]Evaluated on {report['question_count']} questions.[/dim]")


@app.command()
def evaluate_rag(
    questions: Path = typer.Argument(..., help="Path to JSON questions file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON results to this path"
    ),
) -> None:
    """Evaluate RAG pipeline with RAGAS faithfulness, relevancy, and precision metrics."""
    if not questions.exists():
        typer.echo(f"Questions file not found: {questions}", err=True)
        raise typer.Exit(1)

    output = output or Path("eval/results/rag_eval.json")
    report = run_ragas_eval(questions, output_path=output)
    _display_report(report)


if __name__ == "__main__":
    app()
