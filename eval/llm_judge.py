"""LLM-as-Judge evaluation — automated extraction quality scoring without hand labels.

Why LLM-as-Judge:
  Hand-labelling 200 chunks is expensive. A frontier model acting as judge can
  score extraction quality at scale with reasonable calibration, providing early
  signal about which extraction tasks degrade on a specific corpus before the
  full gold-standard benchmark is ready.

  This is not a replacement for the gold-standard benchmark — it's a complement.
  Use it for: development feedback loops, comparing model performance across
  providers, detecting regressions when prompt templates change.

Methodology:
  1. Sample N chunks randomly from the indexed corpus (default 20).
  2. Retrieve the extracted NER, relation, and event results for each chunk.
  3. Call the judge LLM with the source text + extractions.
  4. Parse the score (0–1 per dimension) and aggregate.

Judge model configuration:
  Set JUDGE_MODEL in .env to use a different model for judging than extraction.
  Example: JUDGE_MODEL=anthropic/claude-3-5-sonnet (via OpenRouter)
  If unset, falls back to DEFAULT_MODEL (self-judging, introduces bias).

Usage:
  python eval/llm_judge.py --sample 20 --output eval/results/judge_report.json
  ai4saw eval judge --sample 20
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

import typer

# Lazy import to avoid loading providers at import time
console = Console()
app = typer.Typer(help="LLM-as-Judge extraction quality evaluation.")


def _get_judge_llm():
    """Return a judge LLM — uses JUDGE_MODEL if set, else DEFAULT_MODEL."""
    from ai4saw.core.config import settings
    from ai4saw.core.providers import get_llm

    judge_model = os.environ.get("JUDGE_MODEL", "")
    if judge_model and judge_model != settings.default_model:
        # Construct a separate LLM instance for the judge model
        # Uses the same provider as configured
        match settings.provider:
            case "openrouter":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=judge_model,
                    api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
                    base_url=settings.openrouter_base_url,
                    temperature=0.0,
                )
            case "huggingface":
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                endpoint = HuggingFaceEndpoint(
                    repo_id=judge_model,
                    huggingfacehub_api_token=settings.hf_api_key,
                    temperature=0.01,
                )
                return ChatHuggingFace(llm=endpoint)
            case _:
                # Ollama or fallback
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=judge_model,
                    base_url=settings.ollama_base_url,
                    temperature=0.0,
                )

    logger.warning(
        "JUDGE_MODEL not set — using DEFAULT_MODEL as judge. "
        "Self-judging introduces bias; set JUDGE_MODEL to a different model for reliable eval."
    )
    return get_llm()


def _load_prompt() -> dict:
    from ai4saw.core.config import settings
    with open(settings.prompts_dir / "llm_judge.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_chunk_text(chunk_id: str) -> str:
    from ai4saw.ingestion.embedder import get_vector_store
    try:
        store = get_vector_store()
        result = store._collection.get(ids=[chunk_id], include=["documents"])
        docs = result.get("documents") or []
        return docs[0][:1500] if docs else f"[chunk {chunk_id} not found]"
    except Exception:
        return f"[chunk {chunk_id} — retrieval failed]"


def _get_chunk_meta(chunk_id: str) -> dict:
    from ai4saw.ingestion.embedder import get_vector_store
    try:
        store = get_vector_store()
        result = store._collection.get(ids=[chunk_id], include=["metadatas"])
        metas = result.get("metadatas") or [{}]
        return metas[0]
    except Exception:
        return {}


def _judge_chunk(
    chunk_id: str,
    ner_data: Optional[dict],
    rel_data: Optional[dict],
    event_data: Optional[dict],
    prompt: dict,
    judge_llm,
) -> Optional[dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    source_text = _get_chunk_text(chunk_id)
    meta = _get_chunk_meta(chunk_id)

    ner_str = json.dumps(ner_data.get("entities", []), indent=2) if ner_data else "not available"
    rel_str = json.dumps(rel_data.get("relations", []), indent=2) if rel_data else "not available"
    event_str = json.dumps({
        k: v for k, v in (event_data or {}).items()
        if k != "source_chunk_id"
    }, indent=2) if event_data else "not available"

    template: str = prompt["template"]
    user_content = (
        template
        .replace("{source_text}", source_text)
        .replace("{ner_result}", ner_str[:800])
        .replace("{relation_result}", rel_str[:800])
        .replace("{event_result}", event_str[:400])
    )

    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]

    try:
        response = judge_llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        data["chunk_id"] = chunk_id
        data["source_filename"] = meta.get("source_filename", "unknown")
        return data
    except Exception as exc:
        logger.warning(f"Judge failed for chunk {chunk_id}: {exc}")
        return None


def run_judge(
    ner_file: Path,
    relation_file: Path,
    event_file: Path,
    sample_size: int = 20,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> dict:
    """Run the LLM-as-Judge evaluation on a random sample of chunks.

    Args:
        ner_file: Path to output/ner_results.json
        relation_file: Path to output/relation_results.json
        event_file: Path to output/event_results.json
        sample_size: Number of chunks to evaluate. 20 is a good development
            default; 50+ for final benchmarks.
        output_path: Where to write the JSON report.
        seed: Random seed for reproducible sampling.

    Returns:
        JudgeReport-compatible dict with per-chunk scores and aggregate.
    """
    from ai4saw.core.models import JudgeReport, JudgeScore

    ner_results = json.loads(ner_file.read_text())
    rel_results = json.loads(relation_file.read_text())
    event_results = json.loads(event_file.read_text())

    # Index by chunk_id
    ner_by_id = {r["source_chunk_id"]: r for r in ner_results}
    rel_by_id = {r["source_chunk_id"]: r for r in rel_results}
    event_by_id = {r["source_chunk_id"]: r for r in event_results}

    all_ids = list(set(ner_by_id) | set(rel_by_id) | set(event_by_id))
    random.seed(seed)
    sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

    prompt = _load_prompt()
    judge_llm = _get_judge_llm()
    judge_model = os.environ.get("JUDGE_MODEL", "default")

    logger.info(f"LLM-as-Judge: evaluating {len(sample_ids)} chunks with model={judge_model!r}")

    scores: list[JudgeScore] = []
    for chunk_id in sample_ids:
        result = _judge_chunk(
            chunk_id,
            ner_by_id.get(chunk_id),
            rel_by_id.get(chunk_id),
            event_by_id.get(chunk_id),
            prompt,
            judge_llm,
        )
        if result:
            scores.append(JudgeScore(
                chunk_id=result["chunk_id"],
                source_filename=result.get("source_filename", "unknown"),
                ner_score=float(result.get("ner_score", 0.0)),
                relation_score=float(result.get("relation_score", 0.0)),
                event_score=float(result.get("event_score", 0.0)),
                overall_score=float(result.get("overall_score", 0.0)),
                issues=result.get("issues", []),
                explanation=result.get("explanation", ""),
            ))

    if not scores:
        return {"error": "No chunks successfully evaluated"}

    aggregate = {
        "ner": round(sum(s.ner_score for s in scores) / len(scores), 4),
        "relation": round(sum(s.relation_score for s in scores) / len(scores), 4),
        "event": round(sum(s.event_score for s in scores) / len(scores), 4),
        "overall": round(sum(s.overall_score for s in scores) / len(scores), 4),
    }

    report = JudgeReport(
        scores=scores,
        aggregate=aggregate,
        sample_size=len(scores),
        model_used=judge_model,
    )

    report_dict = report.model_dump()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        logger.info(f"Judge report written → {output_path}")

    return report_dict


def display_report(report: dict) -> None:
    agg = report.get("aggregate", {})
    table = Table(title="LLM-as-Judge Results", show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Mean Score", justify="right", style="bold")
    table.add_column("Interpretation")

    def interp(score: float) -> str:
        if score >= 0.85: return "[green]Good[/green]"
        if score >= 0.70: return "[yellow]Acceptable[/yellow]"
        if score >= 0.50: return "[orange3]Needs improvement[/orange3]"
        return "[red]Poor[/red]"

    for dim in ["ner", "relation", "event", "overall"]:
        s = agg.get(dim, 0.0)
        table.add_row(dim.title(), f"{s:.3f}", interp(s))

    console.print(table)

    # Show top issues
    all_issues = [
        issue
        for score in report.get("scores", [])
        for issue in score.get("issues", [])
    ]
    if all_issues:
        from collections import Counter
        top = Counter(all_issues).most_common(5)
        console.print("\n[bold]Most common issues:[/bold]")
        for issue, count in top:
            console.print(f"  × {issue} ({count}×)")

    console.print(
        f"\n[dim]Evaluated {report.get('sample_size', '?')} chunks "
        f"with {report.get('model_used', '?')}.[/dim]"
    )


@app.command()
def evaluate(
    ner_file: Path = typer.Option(Path("output/ner_results.json")),
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    events_file: Path = typer.Option(Path("output/event_results.json")),
    sample: int = typer.Option(20, help="Number of chunks to evaluate"),
    output: Optional[Path] = typer.Option(None),
    seed: int = typer.Option(42, help="Random seed for reproducible sampling"),
) -> None:
    """Score extraction quality using a frontier LLM as judge."""
    for p in [ner_file, relations_file, events_file]:
        if not p.exists():
            typer.echo(f"File not found: {p} — run `ai4saw extract pipeline` first.", err=True)
            raise typer.Exit(1)

    output = output or Path("eval/results/judge_report.json")
    report = run_judge(ner_file, relations_file, events_file, sample, output, seed)
    display_report(report)


if __name__ == "__main__":
    app()
