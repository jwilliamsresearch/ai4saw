"""AI4SAW — command-line entry point.

All commands are type-annotated and auto-documented via Typer.
Run `ai4saw --help` for the full command reference.

Command groups:
  ingest   — load documents into ChromaDB
  extract  — NER, relations, events, entity resolution
  graph    — build knowledge graph, GraphRAG queries, multi-hop agent
  query    — standard vector-search RAG Q&A
  analyze  — contradiction detection, perpetrator network
  discover — active corpus discovery via ReliefWeb and GDELT
  export   — structured JSON/GeoJSON/GEXF outputs
  eval     — benchmark evaluation (NER, RAG)
  info     — show current configuration
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="ai4saw",
    help="AI for Slavery and War — open intelligence extraction pipeline.",
    rich_markup_mode="rich",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def _app_startup(ctx: typer.Context) -> None:
    """Auto-load active project context before every command.

    If a project is active (set via `ai4saw project switch`), this silently:
      - Overrides settings.chroma_collection to the project collection
      - Sets corpus dir and sources.csv to project paths
      - Loads the search graph as active graph

    This means ALL commands (query, extract, graph, analyze, export) automatically
    operate on the active project without any extra flags.
    """
    if ctx.invoked_subcommand is None:
        return
    try:
        from ai4saw.core.project import load_active_project_context
        load_active_project_context()
    except Exception:
        pass  # never block a command due to project context failure


# ── Project-aware path helpers ─────────────────────────────────────────────────
# Use these instead of hard-coded Path("output/foo") or Path("data/bar") in
# every command.  When an active project exists they return project-scoped paths;
# otherwise they fall back to the legacy global layout.

def _op(*parts: str) -> Path:
    """Resolve an output/ path against the active project (or global output/)."""
    from ai4saw.core.project import get_output_dir
    return get_output_dir().joinpath(*parts)


def _dp(*parts: str) -> Path:
    """Resolve a data/ path against the active project (or global data/)."""
    from ai4saw.core.project import get_data_dir
    return get_data_dir().joinpath(*parts)

# ── Sub-app groups ──────────────────────────────────────────────────────────

ingest_app  = typer.Typer(help="Ingest documents into ChromaDB.")
extract_app = typer.Typer(help="Run extraction (NER, relations, events, entity resolution).")
graph_app   = typer.Typer(help="Knowledge graph: build, query (GraphRAG), multi-hop agent.")
query_app   = typer.Typer(help="Standard vector-search RAG Q&A.")
analyze_app = typer.Typer(help="Contradiction detection and perpetrator network analysis.")
discover_app = typer.Typer(help="Active corpus discovery via ReliefWeb and GDELT.")
export_app  = typer.Typer(help="Export structured outputs (JSON, GeoJSON, GEXF).")
eval_app    = typer.Typer(help="Run evaluation benchmarks.")

project_app = typer.Typer(help="Manage research projects (namespaced corpus + graph).")

app.add_typer(ingest_app,   name="ingest")
app.add_typer(extract_app,  name="extract")
app.add_typer(graph_app,    name="graph")
app.add_typer(query_app,    name="query")
app.add_typer(analyze_app,  name="analyze")
app.add_typer(discover_app, name="discover")
app.add_typer(export_app,   name="export")
app.add_typer(eval_app,     name="eval")
app.add_typer(project_app,  name="project")


# ── Ingest ─────────────────────────────────────────────────────────────────

@ingest_app.command("file")
def ingest_file(
    source: str = typer.Argument(..., help="Path or URL to document"),
    doc_type: str = typer.Option("report", help="report|news|legal|grey_literature"),
    language: str = typer.Option("en", help="ISO 639-1 language code"),
    geography: Optional[str] = typer.Option(None, help="Geographic tag"),
    date_published: Optional[str] = typer.Option(None, help="ISO date, e.g. YYYY-MM-DD"),
) -> None:
    """Load a single document, chunk it, and store in ChromaDB."""
    from ai4saw.ingestion.chunker import chunk_documents
    from ai4saw.ingestion.embedder import embed_and_store
    from ai4saw.ingestion.loaders import load_document

    pub_date = date.fromisoformat(date_published) if date_published else None
    with console.status(f"Loading {source!r}..."):
        docs = load_document(source, doc_type=doc_type, language=language,
                             date_published=pub_date, geography=geography)
    console.print(f"[green]Loaded[/green] {len(docs)} page(s).")
    with console.status("Chunking..."):
        chunks = chunk_documents(docs)
    console.print(f"[green]Chunked[/green] → {len(chunks)} chunk(s).")
    with console.status("Embedding and storing..."):
        embed_and_store(chunks)
    console.print("[bold green]Done.[/bold green] Document indexed in ChromaDB.")


@ingest_app.command("corpus")
def ingest_corpus(
    corpus_dir: Path = typer.Argument(..., help="Directory of documents to ingest"),
    doc_type: str = typer.Option("report", help="report|news|legal|grey_literature"),
    language: str = typer.Option("en", help="ISO 639-1 language code"),
    geography: Optional[str] = typer.Option(None, help="Geographic tag for all docs"),
) -> None:
    """Recursively ingest all documents in a directory."""
    from ai4saw.ingestion.chunker import chunk_documents
    from ai4saw.ingestion.embedder import embed_and_store
    from ai4saw.ingestion.loaders import load_corpus

    if not corpus_dir.is_dir():
        typer.echo(f"Directory not found: {corpus_dir}", err=True)
        raise typer.Exit(1)
    with console.status(f"Loading corpus from {corpus_dir}..."):
        docs = load_corpus(corpus_dir, doc_type=doc_type, language=language, geography=geography)
    console.print(f"[green]Loaded[/green] {len(docs)} document page(s).")
    with console.status("Chunking..."):
        chunks = chunk_documents(docs)
    console.print(f"[green]Chunked[/green] → {len(chunks)} chunk(s).")
    with console.status("Embedding and storing..."):
        embed_and_store(chunks)
    console.print("[bold green]Corpus ingestion complete.[/bold green]")


# ── Extract ────────────────────────────────────────────────────────────────

@extract_app.command("ner")
def run_ner(
    chunk_id: str = typer.Argument(..., help="Chunk ID to run NER on"),
    text: Optional[str] = typer.Option(None, help="Raw text to extract from"),
    output: Optional[Path] = typer.Option(None, help="Save result as JSON"),
) -> None:
    """Extract named entities from a single chunk of text."""
    from ai4saw.extraction.ner import extract_entities

    if not text:
        typer.echo("Provide --text or use `extract pipeline` for batch runs.", err=True)
        raise typer.Exit(1)
    with console.status("Running NER..."):
        result = extract_entities(text, chunk_id)
    console.print(f"\n[bold]Entities extracted:[/bold] {len(result.entities)}")
    for entity in result.entities:
        console.print(f"  [{entity.label}] {entity.text!r}  (conf={entity.confidence:.2f})")
    if output:
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\nSaved to {output}")


@extract_app.command("pipeline")
def run_extraction_pipeline(
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: project or ./output)"),
    delay: float = typer.Option(0.25, help="Delay between LLM calls (seconds)"),
    max_chunks: int = typer.Option(0, "--max", "-n", help="Max chunks to process (0 = all)"),
) -> None:
    """Run full extraction pipeline (NER + relations + events) on all indexed chunks."""
    from ai4saw.extraction.events import classify_events_batch
    from ai4saw.extraction.ner import extract_entities_batch
    from ai4saw.extraction.relations import extract_relations_batch
    from ai4saw.ingestion.embedder import get_vector_store

    output_dir = output_dir or _op()
    store = get_vector_store()
    result = store._collection.get(include=["documents", "metadatas"])
    texts: list[str] = result.get("documents") or []
    ids: list[str] = result.get("ids") or []
    pairs = list(zip(texts, ids))

    if not pairs:
        typer.echo("No documents in ChromaDB. Run `ai4saw ingest` first.", err=True)
        raise typer.Exit(1)

    if max_chunks > 0:
        pairs = pairs[:max_chunks]
        console.print(f"Running extraction on {len(pairs)} chunk(s) (limited from {len(texts)})...")
    else:
        console.print(f"Running extraction on {len(pairs)} chunk(s)...")
    ner_results = extract_entities_batch(pairs, delay_between=delay)
    rel_results = extract_relations_batch(pairs, delay_between=delay)
    event_results = classify_events_batch(pairs, delay_between=delay)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ner_results.json").write_text(
        json.dumps([r.model_dump() for r in ner_results], indent=2), encoding="utf-8"
    )
    (output_dir / "relation_results.json").write_text(
        json.dumps([r.model_dump() for r in rel_results], indent=2), encoding="utf-8"
    )
    (output_dir / "event_results.json").write_text(
        json.dumps([r.model_dump() for r in event_results], indent=2), encoding="utf-8"
    )
    console.print(Panel(
        f"NER: {len(ner_results)} chunks\nRelations: {len(rel_results)} chunks\n"
        f"Events: {len(event_results)} chunks\nOutput: {output_dir.resolve()}",
        title="[bold green]Extraction complete[/bold green]",
    ))


@extract_app.command("resolve")
def run_entity_resolution(
    ner_file: Optional[Path] = typer.Option(None),
    cosine_threshold: float = typer.Option(0.88, help="Cosine similarity threshold (0–1)"),
    fuzzy_threshold: float = typer.Option(72.0, help="String fuzzy match threshold (0–100)"),
    output: Optional[Path] = typer.Option(None, help="Output path"),
) -> None:
    """Resolve NER entities across the corpus into a canonical registry.

    Merges aliases ("Armed-Group-Beta", "Rapid Support Forces", "the paramilitaries") into
    single canonical entities using embedding similarity + string matching.
    Required before building the knowledge graph.
    """
    from ai4saw.core.models import NERResult
    from ai4saw.synthesis.entity_resolution import resolve_entities, save_entity_registry

    ner_file = ner_file or _op("ner_results.json")
    output   = output   or _dp("entity_registry.json")
    if not ner_file.exists():
        typer.echo(f"NER results not found: {ner_file} — run `extract pipeline` first.", err=True)
        raise typer.Exit(1)

    ner_results = [NERResult(**r) for r in json.loads(ner_file.read_text())]
    with console.status(f"Resolving entities across {len(ner_results)} chunks..."):
        result = resolve_entities(ner_results, cosine_threshold, fuzzy_threshold)

    save_entity_registry(result, str(output))
    console.print(Panel(
        f"Mentions:        {result.total_mentions}\n"
        f"Unique texts:    {result.unique_texts_before}\n"
        f"Canonical ents:  {result.resolved_count}\n"
        f"Registry saved:  {output}",
        title="[bold green]Entity resolution complete[/bold green]",
    ))


# ── Graph ──────────────────────────────────────────────────────────────────

@graph_app.command("build")
def graph_build(
    relations_file: Optional[Path] = typer.Option(None),
    registry_file: Optional[Path] = typer.Option(None),
    min_confidence: float = typer.Option(0.5, help="Minimum relation confidence to include"),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Build the knowledge graph from extracted relations and resolved entities.

    Run `extract pipeline` and `extract resolve` before this command.
    The knowledge graph is used by `graph query`, `graph agent`, and network analysis.
    """
    from ai4saw.core.models import RelationResult
    from ai4saw.retrieval.graph_rag import build_knowledge_graph, save_knowledge_graph
    from ai4saw.synthesis.entity_resolution import load_entity_registry

    relations_file = relations_file or _op("relation_results.json")
    registry_file  = registry_file  or _dp("entity_registry.json")
    output         = output         or _dp("knowledge_graph.json")
    for p in [relations_file, registry_file]:
        if not p.exists():
            typer.echo(f"Required file not found: {p}", err=True)
            raise typer.Exit(1)

    rel_results = [RelationResult(**r) for r in json.loads(relations_file.read_text())]
    registry = load_entity_registry(str(registry_file))

    with console.status("Building knowledge graph..."):
        graph = build_knowledge_graph(rel_results, registry, min_confidence=min_confidence)

    save_knowledge_graph(graph, str(output))
    console.print(Panel(
        f"Nodes:  {graph.node_count}\nEdges:  {graph.edge_count}\nSaved:  {output}",
        title="[bold green]Knowledge graph built[/bold green]",
    ))


@graph_app.command("query")
def graph_query(
    question: str = typer.Argument(..., help="Question or entity name to query"),
    hops: int = typer.Option(2, help="Graph neighbourhood depth"),
    at: Optional[str] = typer.Option(
        None, "--at", help="ISO date for temporal filtering, e.g. YYYY-MM-DD"
    ),
    graph_file: Optional[Path] = typer.Option(None),
    combine_vector: bool = typer.Option(True, help="Also run vector search and combine"),
) -> None:
    """Query the knowledge graph (GraphRAG) — structural + semantic retrieval.

    Identifies entities mentioned in the question, extracts their neighbourhood
    from the graph, and combines with vector search results.

    Use --at to filter the graph to a specific point in time:
      ai4saw graph query "Drina Corps command" --at YYYY-MM-DD
    """
    from ai4saw.retrieval.graph_rag import graph_context_for_query, load_knowledge_graph

    graph_file = graph_file or _dp("knowledge_graph.json")
    if not graph_file.exists():
        typer.echo("Knowledge graph not found. Run `ai4saw graph build` first.", err=True)
        raise typer.Exit(1)

    at_label = f" (at {at})" if at else ""
    with console.status(f"Querying knowledge graph{at_label}..."):
        graph = load_knowledge_graph(str(graph_file))
        graph_ctx = graph_context_for_query(question, graph=graph, hops=hops, at_date=at)

    if not graph_ctx:
        console.print("[yellow]No matching entities found in knowledge graph.[/yellow]")
        if at:
            console.print(f"[dim]Temporal filter applied: {at} — try without --at for all edges.[/dim]")
        console.print("Try `ai4saw query ask` for pure vector search.")
        return

    if combine_vector:
        from ai4saw.retrieval.qa import answer as qa_answer
        with console.status("Running vector search..."):
            qa_resp = qa_answer(question)
        console.print(Panel(qa_resp.answer, title=f"[bold cyan]GraphRAG Answer{at_label}[/bold cyan]"))
    else:
        console.print(Panel(graph_ctx, title=f"[bold cyan]Graph Context{at_label}[/bold cyan]"))


@graph_app.command("agent")
def graph_agent(
    question: str = typer.Argument(..., help="Complex research question"),
    max_iterations: int = typer.Option(8, help="Maximum agent tool calls"),
    output: Optional[Path] = typer.Option(None, help="Save AgentResponse as JSON"),
) -> None:
    """Multi-hop reasoning agent — decomposes and chains retrieval for complex questions.

    Unlike `query ask` (single retrieval), the agent iteratively calls tools
    (vector search, graph query, entity lookup) until it has enough context.
    Best for questions requiring temporal chaining, actor tracking, or
    cross-document synthesis.
    """
    from ai4saw.retrieval.agent import multi_hop_answer

    with console.status("Agent reasoning (this may take a moment)..."):
        response = multi_hop_answer(question, max_iterations=max_iterations)

    console.print(Panel(response.answer, title="[bold cyan]Agent Answer[/bold cyan]"))

    if response.steps:
        table = Table(title="Reasoning Steps", show_header=True)
        table.add_column("#", width=3)
        table.add_column("Tool")
        table.add_column("Query")
        table.add_column("Result (preview)")
        for i, step in enumerate(response.steps, 1):
            table.add_row(
                str(i), step.tool_used,
                step.sub_question[:50],
                step.result_summary[:60],
            )
        console.print(table)

    console.print(
        f"\n[dim]{response.iterations} tool calls. "
        f"Sources: {', '.join(response.sources_consulted[:5]) or 'none'}[/dim]"
    )

    if output:
        output.write_text(response.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Saved → {output}")


# ── Query ─────────────────────────────────────────────────────────────────

@query_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Natural language question"),
    top_k: int = typer.Option(8, help="Chunks to retrieve"),
    top_n: int = typer.Option(3, help="Chunks to keep after re-ranking"),
) -> None:
    """Ask a question over the indexed corpus and receive a cited answer.

    For complex multi-step questions, use `ai4saw graph agent` instead.
    """
    from ai4saw.retrieval.qa import answer as qa_answer

    with console.status("Retrieving and generating answer..."):
        response = qa_answer(question, top_k=top_k, top_n=top_n)

    console.print(Panel(response.answer, title="[bold cyan]Answer[/bold cyan]"))

    table = Table(title="Sources", show_header=True)
    table.add_column("#", width=3)
    table.add_column("File")
    table.add_column("Geography")
    table.add_column("Date")
    table.add_column("Type")
    for i, src in enumerate(response.sources, start=1):
        table.add_row(
            str(i), src.source_filename,
            src.geography or "—",
            str(src.date_published) if src.date_published else "—",
            src.doc_type,
        )
    console.print(table)
    console.print(
        f"\n[dim]Retrieved {response.retrieved_chunks} chunks, "
        f"re-ranked to {response.reranked_to}. Confidence: {response.confidence:.2f}[/dim]"
    )


# ── Analyze ───────────────────────────────────────────────────────────────

@analyze_app.command("contradictions")
def analyze_contradictions(
    events_file: Optional[Path] = typer.Option(None),
    relations_file: Optional[Path] = typer.Option(None),
    min_confidence: float = typer.Option(0.65, help="Minimum LLM confidence to report a pair"),
    max_pairs: int = typer.Option(100, help="Maximum candidate pairs to assess (controls cost)"),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Detect conflicting claims across source documents.

    Uses a two-pass approach: cheap candidate generation (grouping by
    location/time), then LLM verification per candidate pair. Contradictions
    are not discarded — they are ranked by confidence and reported for
    researcher review.
    """
    from ai4saw.core.models import EventResult, RelationResult
    from ai4saw.synthesis.contradiction import detect_contradictions

    events_file    = events_file    or _op("event_results.json")
    relations_file = relations_file or _op("relation_results.json")
    output         = output         or _op("contradictions.json")
    for p in [events_file, relations_file]:
        if not p.exists():
            typer.echo(f"Required file not found: {p} — run `extract pipeline` first.", err=True)
            raise typer.Exit(1)

    event_results = [EventResult(**r) for r in json.loads(events_file.read_text())]
    rel_results = [RelationResult(**r) for r in json.loads(relations_file.read_text())]

    with console.status("Detecting contradictions..."):
        report = detect_contradictions(
            event_results, rel_results,
            llm_confidence_threshold=min_confidence,
            max_pairs_to_assess=max_pairs,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"pairs": [p.model_dump() for p in report.pairs],
             "total_chunks_analysed": report.total_chunks_analysed,
             "candidate_pairs_assessed": report.candidate_pairs_assessed,
             "high_confidence_count": report.high_confidence_count},
            indent=2
        ),
        encoding="utf-8",
    )

    if report.pairs:
        table = Table(title="Top Contradictions", show_header=True)
        table.add_column("Type")
        table.add_column("Confidence", justify="right")
        table.add_column("Source A")
        table.add_column("Source B")
        table.add_column("Explanation")
        for pair in report.pairs[:10]:
            table.add_row(
                pair.contradiction_type.value,
                f"{pair.confidence:.2f}",
                pair.source_a[:30],
                pair.source_b[:30],
                pair.explanation[:60],
            )
        console.print(table)
    else:
        console.print("[green]No contradictions detected above confidence threshold.[/green]")

    console.print(
        f"\n[dim]{report.candidate_pairs_assessed} pairs assessed, "
        f"{len(report.pairs)} confirmed, "
        f"{report.high_confidence_count} high-confidence. Saved → {output}[/dim]"
    )


@analyze_app.command("network")
def analyze_network(
    relations_file: Optional[Path] = typer.Option(None),
    registry_file: Optional[Path] = typer.Option(
        None, help="Entity registry (from extract resolve) — enables canonicalisation"
    ),
    min_confidence: float = typer.Option(0.5, help="Minimum relation confidence"),
    output: Optional[Path] = typer.Option(None),
    gexf: bool = typer.Option(False, help="Also export GEXF for Gephi visualisation"),
) -> None:
    """Build and analyse the perpetrator command network.

    Constructs a directed graph from extracted relations. Computes betweenness
    centrality (identifies critical nodes in command chains) and community
    detection (identifies operational clusters). Exports JSON and optionally
    GEXF for Gephi.
    """
    from ai4saw.core.models import RelationResult
    from ai4saw.synthesis.network import (
        build_command_network,
        export_network_gexf,
        save_network,
    )

    relations_file = relations_file or _op("relation_results.json")
    output         = output         or _op("network.json")

    if not relations_file.exists():
        typer.echo(f"Relations file not found: {relations_file}", err=True)
        raise typer.Exit(1)

    rel_results = [RelationResult(**r) for r in json.loads(relations_file.read_text())]

    registry = None
    if registry_file and registry_file.exists():
        from ai4saw.synthesis.entity_resolution import load_entity_registry
        registry = load_entity_registry(str(registry_file))

    with console.status("Building command network..."):
        analysis = build_command_network(rel_results, registry, min_confidence)

    save_network(analysis, str(output))

    if gexf:
        gexf_path = output.with_suffix(".gexf")
        export_network_gexf(rel_results, registry, str(gexf_path))
        console.print(f"GEXF exported → {gexf_path}")

    console.print(Panel(
        f"Nodes:          {analysis.total_nodes}\n"
        f"Edges:          {analysis.total_edges}\n"
        f"Command edges:  {analysis.command_edges}\n"
        f"Communities:    {len(analysis.communities)}\n"
        f"Key actors:     {', '.join(analysis.key_actors[:5])}\n"
        f"Saved:          {output}",
        title="[bold green]Network analysis complete[/bold green]",
    ))


# ── Discover ───────────────────────────────────────────────────────────────

@discover_app.command("run")
def discover_run(
    entities: Optional[list[str]] = typer.Argument(
        None, help="Entity names to search for (e.g. 'Location Beta' 'Rapid Support Forces')"
    ),
    from_registry: bool = typer.Option(
        False, "--from-registry", help="Use top entities from the entity registry"
    ),
    top_n: int = typer.Option(10, help="Top N entities from registry to query"),
    labels: Optional[list[str]] = typer.Option(
        None, "--label", help="Filter registry entities by label (e.g. LOCATION ORG)"
    ),
    per_entity_limit: int = typer.Option(25, help="Max results per entity per source"),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Discover documents not yet in the corpus via ReliefWeb and GDELT.

    Queries two free, no-auth APIs. Results are deduplicated against
    corpus/sources.csv. Review the output before ingesting any document.

    Examples:
      ai4saw discover run "Location Beta" "Group Beta" "Armed-Group-Beta"
      ai4saw discover run --from-registry --label LOCATION --top-n 5
    """
    from ai4saw.discovery.discovery import (
        discover_for_entities,
        discover_from_registry,
    )

    output = output or _op("discovered_documents.json")
    if from_registry:
        registry_path = _dp("entity_registry.json")
        if not registry_path.exists():
            typer.echo("Entity registry not found. Run `extract resolve` first.", err=True)
            raise typer.Exit(1)
        from ai4saw.synthesis.entity_resolution import load_entity_registry
        registry = load_entity_registry(str(registry_path))
        with console.status("Discovering from entity registry..."):
            result = discover_from_registry(
                registry,
                top_n=top_n,
                entity_labels=labels or None,
                per_entity_limit=per_entity_limit,
            )
    elif entities:
        with console.status(f"Discovering for {len(entities)} entity/ies..."):
            result = discover_for_entities(
                list(entities), per_entity_limit=per_entity_limit
            )
    else:
        typer.echo("Provide entity names or --from-registry.", err=True)
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    output.write_text(
        json.dumps(
            {"trigger_entities": result.trigger_entities,
             "query_count": result.query_count,
             "new_documents": result.new_documents,
             "documents": [d.model_dump() for d in result.documents]},
            indent=2
        ),
        encoding="utf-8",
    )

    if result.documents:
        table = Table(title="Discovered Documents", show_header=True)
        table.add_column("Source", width=10)
        table.add_column("Relevance", justify="right", width=9)
        table.add_column("Date", width=12)
        table.add_column("Entity")
        table.add_column("Title")
        for doc in result.documents[:20]:
            table.add_row(
                doc.source,
                f"{doc.relevance_score:.2f}",
                doc.date or "—",
                doc.trigger_entity[:20],
                doc.title[:60],
            )
        console.print(table)
    else:
        console.print("[yellow]No new documents found.[/yellow]")

    console.print(
        f"\n[dim]{result.query_count} API queries, "
        f"{result.new_documents} new documents. Saved → {output}[/dim]"
    )


@discover_app.command("fetch")
def discover_fetch(
    entities: list[str] = typer.Argument(
        ..., help="Entity names to search for (e.g. 'Location Alpha' 'Armed-Group-Alpha' 'Commander Alpha')"
    ),
    geography: str = typer.Option(..., help="Geography tag for chunk metadata and sources.csv"),
    max_docs: int = typer.Option(100, help="Hard cap on documents to ingest in one run"),
    min_relevance: float = typer.Option(0.5, help="Minimum relevance score (0–1)"),
    per_entity_limit: int = typer.Option(25, help="Max results per entity per source"),
    silence_mode: bool = typer.Option(
        False, "--silence-mode", help="Treat entities as silence candidates (higher per-entity limit)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show candidates only — download nothing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Discover, download, and ingest documents from ReliefWeb and GDELT.

    Automates corpus preparation: queries both APIs, filters by relevance,
    downloads documents, registers them in corpus/sources.csv, and ingests
    them into ChromaDB — in a single command.

    Examples:

      ai4saw discover fetch "Location Alpha" "Commander Alpha" --geography conflict-region --max-docs 20

      ai4saw discover fetch "Location Beta" "Group Beta" --geography conflict-region --dry-run

      ai4saw discover fetch "Location C" "Location D" --geography conflict-region --silence-mode --yes
    """
    from ai4saw.agents.fetch_agent import fetch_corpus

    # ── Dry run: discover and show candidates, then exit ──────────────────────
    if dry_run:
        console.print("[dim]Dry run — no files will be downloaded.[/dim]\n")
        result = fetch_corpus(
            entities=list(entities),
            geography=geography,
            min_relevance=min_relevance,
            max_docs=max_docs,
            per_entity_limit=per_entity_limit,
            silence_mode=silence_mode,
            dry_run=True,
        )
        console.print(
            f"[bold]{result.candidates_found}[/bold] candidates found, "
            f"[bold]{result.candidates_above_threshold}[/bold] above relevance threshold {min_relevance}."
        )
        return

    # ── Discovery pass to show candidates before committing ───────────────────
    if not yes:
        from ai4saw.agents.fetch_agent import fetch_corpus as _fc
        from ai4saw.discovery.discovery import discover_for_entities, discover_for_silences
        from ai4saw.agents.fetch_agent import _is_registered

        if silence_mode:
            discovery = discover_for_silences(list(entities), per_entity_limit=max(per_entity_limit, 15))
        else:
            discovery = discover_for_entities(list(entities), per_entity_limit=per_entity_limit)

        candidates = [
            d for d in discovery.documents
            if d.relevance_score >= min_relevance and not _is_registered(d.url)
        ][:max_docs]

        if not candidates:
            console.print("[yellow]No new candidates found above threshold.[/yellow]")
            return

        table = Table(title="Candidates to fetch", show_header=True)
        table.add_column("Source", width=11)
        table.add_column("Relevance", justify="right", width=9)
        table.add_column("Date", width=12)
        table.add_column("Title")
        for c in candidates:
            table.add_row(c.source, f"{c.relevance_score:.2f}", c.date or "—", c.title[:70])
        console.print(table)

        confirmed = typer.confirm(f"\nFetch {len(candidates)} document(s)?", default=False)
        if not confirmed:
            raise typer.Exit(0)

    # ── Run the full fetch pipeline ───────────────────────────────────────────
    with console.status("[bold cyan]Fetching and ingesting…[/bold cyan]"):
        result = fetch_corpus(
            entities=list(entities),
            geography=geography,
            min_relevance=min_relevance,
            max_docs=max_docs,
            per_entity_limit=per_entity_limit,
            silence_mode=silence_mode,
            dry_run=False,
        )

    if result.fetched:
        table = Table(title="Fetched Documents", show_header=True)
        table.add_column("Source", width=11)
        table.add_column("Chunks", justify="right", width=7)
        table.add_column("Licence", width=8)
        table.add_column("Title")
        for f in result.fetched:
            table.add_row(f.source, str(f.chunks_added), f.licence, f.title[:70])
        console.print(table)

    console.print(Panel(
        f"Candidates found:    {result.candidates_found}\n"
        f"Above threshold:     {result.candidates_above_threshold}\n"
        f"Documents fetched:   {result.documents_fetched}\n"
        f"Chunks added:        {result.chunks_added}\n"
        f"Skipped (errors):    {len(result.skipped)}",
        title="[bold green]Fetch complete[/bold green]",
    ))

    if result.skipped:
        console.print("[dim]Skipped URLs:[/dim]")
        for url in result.skipped:
            console.print(f"  [dim]{url}[/dim]")


@discover_app.command("web")
def discover_web(
    entities: list[str] = typer.Argument(
        ..., help="Entity names to search for (e.g. 'Location Alpha' 'Commander Alpha')"
    ),
    geography: str = typer.Option(..., help="Geography tag for chunk metadata and sources.csv"),
    min_relevance: float = typer.Option(0.4, help="Minimum relevance score (0–1)"),
    per_entity_limit: int = typer.Option(10, help="Max DDG results per template per entity"),
    frontier_batch: int = typer.Option(30, help="Frontier URLs to visit per session"),
    rediscover_every: int = typer.Option(4, help="Run fresh DDG/Wikipedia/CrossRef every N sessions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show state and candidates — ingest nothing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    loop: bool = typer.Option(False, "--loop", help="Run continuously until Ctrl-C"),
    interval: int = typer.Option(900, help="Seconds between sessions (default: 15 min)"),
    show_state: bool = typer.Option(False, "--state", help="Print agent state summary and exit"),
) -> None:
    """Autonomous web agent — continuously searches and ingests with persistent memory.

    The agent maintains a state file (output/web_agent_state.json) tracking every
    URL visited, a priority frontier of pending URLs, domain hit-rates, and query
    template yield scores.  It adapts: high-yield queries run first; unproductive
    domains are deprioritised.

    Each session:
      1. Drain frontier — visit queued URLs, extract new links, ingest quality docs
      2. Every --rediscover-every sessions, run fresh DuckDuckGo / Wikipedia / CrossRef
      3. All new URLs go into the frontier for future sessions

    With --loop it runs indefinitely, sleeping --interval seconds between sessions.
    State persists across restarts — safe to stop and resume at any time.

    Examples:

      # First run: discover and start building frontier
      ai4saw discover web "Location Alpha" "Commander Alpha" "Armed-Group-Alpha" "conflict-region" --geography conflict-region --yes

      # Let it run overnight, re-querying every hour
      ai4saw discover web "Location Alpha" "Commander Alpha" --geography conflict-region --loop --interval 900 --yes

      # Check what the agent knows
      ai4saw discover web "Location Alpha" --geography conflict-region --state
    """
    from ai4saw.agents.web_agent import (
        web_discover, drain_frontier, load_state, save_state,
        get_state_summary,
    )
    from ai4saw.core.config import settings

    contact_email = getattr(settings, "contact_email", "")
    state = load_state()

    # ── State summary mode ────────────────────────────────────────────────────
    if show_state:
        summary = get_state_summary(state)
        console.print(Panel(
            f"Sessions run:      {summary['sessions']}\n"
            f"URLs visited:      {summary['urls_visited']}\n"
            f"Frontier size:     {summary['frontier_size']}\n"
            f"Docs ingested:     {summary['docs_ingested']}\n"
            f"Chunks added:      {summary['chunks_added']}\n"
            f"Last run:          {state.last_run or 'never'}",
            title="[bold cyan]Web Agent State[/bold cyan]",
        ))
        if summary["top_domains"]:
            t = Table(title="Top domains by hit rate")
            t.add_column("Domain"); t.add_column("Score", justify="right")
            t.add_column("Hits", justify="right"); t.add_column("Attempts", justify="right")
            for domain, score, hits, attempts in summary["top_domains"]:
                t.add_row(domain, f"{score:.2f}", str(hits), str(attempts))
            console.print(t)
        if summary["top_queries"]:
            t = Table(title="Top query templates by yield")
            t.add_column("Template key"); t.add_column("Yield/run", justify="right"); t.add_column("Runs", justify="right")
            for key, rate, runs in summary["top_queries"]:
                t.add_row(key, f"{rate:.2f}", str(runs))
            console.print(t)
        return

    session = 0

    while True:
        session += 1
        run_discovery = (session == 1) or (session % rediscover_every == 0)

        if loop or session > 1:
            console.print(f"\n[bold cyan]── Web agent session #{state.session_count + 1} ──[/bold cyan]")

        # ── Phase 1: Drain frontier ───────────────────────────────────────────
        if state.frontier:
            console.print(
                f"[dim]Frontier: {len(state.frontier)} pending URLs. "
                f"Draining {min(frontier_batch, len(state.frontier))}…[/dim]"
            )
            if not dry_run:
                import httpx as _httpx
                with _httpx.Client(timeout=20.0) as client:
                    docs_in, chunks_in = drain_frontier(
                        state, geography, client,
                        batch_size=frontier_batch,
                        min_relevance=min_relevance,
                    )
                state.total_docs_ingested += docs_in
                state.total_chunks_added += chunks_in
                if docs_in:
                    console.print(f"  Frontier drained: [bold]{docs_in}[/bold] docs, [bold]{chunks_in}[/bold] chunks")

        # ── Phase 2: Fresh discovery (every N sessions) ───────────────────────
        if run_discovery:
            console.print("[dim]Running fresh discovery (DDG · Wikipedia · CrossRef)…[/dim]")
            with console.status("[bold cyan]Searching…[/bold cyan]"):
                result, state = web_discover(
                    entities=list(entities),
                    per_entity_limit=per_entity_limit,
                    contact_email=contact_email,
                    state=state,
                )

            console.print(
                f"  Discovery: [bold]{result.new_documents}[/bold] new URLs → "
                f"frontier now [bold]{len(state.frontier)}[/bold]"
            )

            if dry_run:
                above = [d for d in result.documents if d.relevance_score >= min_relevance]
                if above:
                    t = Table(title="Candidates (dry run — not ingested)")
                    t.add_column("Source", width=11); t.add_column("Rel.", justify="right", width=5); t.add_column("Title")
                    for d in above[:25]:
                        t.add_row(d.source, f"{d.relevance_score:.2f}", d.title[:70])
                    console.print(t)

        # ── Save state ────────────────────────────────────────────────────────
        if not dry_run:
            save_state(state)

        # ── Summary ───────────────────────────────────────────────────────────
        console.print(Panel(
            f"Session:           #{state.session_count}\n"
            f"URLs visited:      {len(state.visited_urls)}\n"
            f"Frontier pending:  {len(state.frontier)}\n"
            f"Total ingested:    {state.total_docs_ingested} docs / {state.total_chunks_added} chunks",
            title="[bold green]Session complete[/bold green]",
        ))

        if not loop:
            break

        console.print(f"[dim]Next session in {interval}s (Ctrl-C to stop).[/dim]")
        time.sleep(interval)


@discover_app.command("agent")
def discover_agent(
    entities: list[str] = typer.Argument(
        ..., help="Seed entity names (e.g. 'Location Alpha' 'Commander Alpha')"
    ),
    geography: str = typer.Option(..., help="Geography tag for chunk metadata and sources.csv"),
    min_relevance: float = typer.Option(0.4, help="Minimum frontier priority to ingest"),
    frontier_batch: int = typer.Option(20, help="Frontier URLs to visit per session"),
    per_entity_limit: int = typer.Option(10, help="Max DDG results per template per entity (seed)"),
    seed_every: int = typer.Option(6, help="Re-seed frontier via web discovery every N sessions"),
    max_reasoning: int = typer.Option(8, help="Max docs to run LLM reasoning on per session"),
    loop: bool = typer.Option(False, "--loop", help="Run continuously until Ctrl-C"),
    interval: int = typer.Option(1200, help="Seconds between sessions (default: 20 min)"),
    show_state: bool = typer.Option(False, "--state", help="Print agent state and exit"),
    show_log: int = typer.Option(0, "--log", help="Print last N reasoning entries and exit"),
    project: Optional[str] = typer.Option(None, "--project", "-p",
                                           help="Project slug (overrides active project)"),
) -> None:
    """Agentic discovery — LLM reads each document and decides what to search for next.

    This is a true reasoning loop, not a template crawler.  After ingesting each
    document the LLM extracts novel entities and generates specific search queries
    based on what it found.  Those queries feed the next session's frontier.

    Example snowball:
      Seed: "Location Alpha"  →  finds International Tribunal Commander Beta judgment
      LLM reads it  →  extracts "Witness Alpha", "Unit Alpha"
      LLM generates:
        "Witness Alpha plea agreement International Tribunal 1996"
        "Unit Alpha Armed-Group-Alpha Site Alpha executions"
      →  3 queries no template would have produced

    State:  output/agent_discover_state.json
    Log:    output/agent_discover_log.jsonl

    Examples:

      ai4saw discover agent "Location Alpha" "Commander Alpha" --geography conflict-region --yes

      ai4saw discover agent "Location Beta" "Group Beta" "Armed-Group-Beta" --geography conflict-region --loop --yes

      ai4saw discover agent "Location Alpha" --geography conflict-region --state

      ai4saw discover agent "Location Alpha" --geography conflict-region --log 5
    """
    from ai4saw.agents.agent_discover import (
        load_agent_state, save_agent_state, run_agent_session,
        get_agent_summary, top_novel_entities, _seed_frontier, AGENT_LOG_FILE,
    )
    from ai4saw.core.config import settings
    from ai4saw.core.project import (
        resolve_project, get_active_project, create_project,
        set_active_project, get_project_paths, set_active_paths,
    )
    from ai4saw.core.search_graph import SearchGraph, set_active_graph

    # ── Project context — always required, auto-create if none active ──────────
    proj_paths = resolve_project(project)
    if proj_paths is None:
        # Auto-create from entity list + geography
        _auto_name = (", ".join(entities[:2]) + (f" ({geography})" if geography else ""))[:48]
        try:
            _meta = create_project(
                name=_auto_name,
                research_query=" ".join(entities),
                geography=geography,
            )
            set_active_project(_meta.slug)
            proj_paths = get_project_paths(_meta.slug)
            console.print(f"[green]✓[/green] Auto-created project: [bold]{_meta.slug}[/bold]")
        except Exception as _pe:
            console.print(f"[yellow]⚠ Project auto-create failed: {_pe} — using global paths[/yellow]")

    _state_path = proj_paths["agent_state"] if proj_paths else None
    _log_path   = proj_paths["agent_log"]   if proj_paths else None
    _graph_port_da: Optional[int] = None

    if proj_paths:
        set_active_paths(proj_paths)
        _sg = SearchGraph(proj_paths["search_graph"])
        set_active_graph(_sg)
        settings.chroma_collection = proj_paths["chroma_collection"]
        try:
            from ai4saw.ui.graph_server import start_graph_server
            _graph_port_da = start_graph_server(proj_paths["search_graph"], project_name=proj_paths["dir"].name)
            console.print(f"[dim]Project: {proj_paths['dir'].name} | Graph: http://localhost:{_graph_port_da}[/dim]")
        except Exception:
            console.print(f"[dim]Project: {proj_paths['dir'].name} | Collection: {proj_paths['chroma_collection']}[/dim]")
    else:
        set_active_graph(None)

    contact_email = getattr(settings, "contact_email", "")
    state = load_agent_state(path=_state_path)

    # Keep initial_entities updated in state (idempotent)
    for e in entities:
        if e not in state.initial_entities:
            state.initial_entities.append(e)

    # ── State summary ─────────────────────────────────────────────────────────
    if show_state:
        summary = get_agent_summary(state)
        console.print(Panel(
            f"Sessions:            {summary['sessions']}\n"
            f"URLs visited:        {summary['urls_visited']}\n"
            f"Frontier pending:    {summary['frontier_size']}\n"
            f"Docs ingested:       {summary['docs_ingested']}\n"
            f"Chunks added:        {summary['chunks_added']}\n"
            f"Docs reasoned (LLM): {summary['docs_reasoned']}\n"
            f"Novel entities:      {summary['novel_entities']}\n"
            f"Queries queued:      {summary['queries_queued']}\n"
            f"Queries executed:    {summary['queries_executed']}\n"
            f"Last run:            {summary['last_run'] or 'never'}",
            title="[bold cyan]Agent Discover State[/bold cyan]",
        ))
        if summary["top_novel_entities"]:
            t = Table(title="Top novel entities (by mention frequency)")
            t.add_column("Entity"); t.add_column("Mentions", justify="right")
            for entity, count in summary["top_novel_entities"]:
                t.add_row(entity, str(count))
            console.print(t)
        return

    # ── Log viewer ────────────────────────────────────────────────────────────
    if show_log > 0:
        if not AGENT_LOG_FILE.exists():
            console.print("[yellow]No reasoning log yet.[/yellow]")
            return
        entries = AGENT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        for line in entries[-show_log:]:
            try:
                entry = json.loads(line)
                console.print(Panel(
                    f"[bold]{entry.get('source_title', '?')[:80]}[/bold]\n"
                    f"URL: {entry.get('source_url', '')[:80]}\n\n"
                    f"Novel entities: {entry.get('novel_entities', [])}\n\n"
                    f"Queries generated:\n" +
                    "\n".join(f"  • {q}" for q in entry.get("generated_queries", [])) +
                    f"\n\nReasoning: {entry.get('reasoning', '')}",
                    title=f"[cyan]{entry.get('timestamp', '')[:19]}[/cyan]",
                ))
            except Exception:
                pass
        return

    session_number = 0

    while True:
        session_number += 1
        console.print(
            f"\n[bold cyan]── Agent session #{state.session_count + 1} "
            f"(frontier: {len(state.frontier)}, queued queries: {len(state.query_queue)}) ──[/bold cyan]"
        )

        # ── Seed frontier on first run or every N sessions ────────────────────
        needs_seed = (len(state.frontier) < 5) or (state.session_count % seed_every == 0)
        if needs_seed:
            console.print("[dim]Seeding frontier via web discovery…[/dim]")
            with console.status("[bold cyan]Searching (DDG · Wikipedia · CrossRef)…[/bold cyan]"):
                new_count = _seed_frontier(
                    list(state.initial_entities) + top_novel_entities(state, n=10),
                    state,
                    per_entity_limit=per_entity_limit,
                    contact_email=contact_email,
                )
            console.print(f"  Frontier seeded: [bold]{len(state.frontier)}[/bold] pending URLs")

        # ── Run agent session ─────────────────────────────────────────────────
        with console.status("[bold cyan]Draining frontier · reasoning · executing novel queries…[/bold cyan]"):
            docs_in, chunks_in, reasonings = run_agent_session(
                state=state,
                geography=geography,
                frontier_batch=frontier_batch,
                min_relevance=min_relevance,
                max_reasoning=max_reasoning,
            )

        # ── Display reasoning results ─────────────────────────────────────────
        if reasonings:
            t = Table(title="LLM Reasoning this session", show_header=True)
            t.add_column("Doc", max_width=40)
            t.add_column("Novel entities", max_width=30)
            t.add_column("Queries generated", max_width=50)
            for r in reasonings:
                t.add_row(
                    r.source_title[:40],
                    ", ".join(r.novel_entities[:3]),
                    "\n".join(r.generated_queries[:2]),
                )
            console.print(t)

        save_agent_state(state, path=_state_path)
        if proj_paths:
            _sg.save()
            if _graph_port_da:
                try:
                    from ai4saw.ui.graph_server import push_update
                    _gdata = json.loads(proj_paths["search_graph"].read_text(encoding="utf-8"))
                    _nodes = [{"id": n["id"], "label": n.get("label",""), "type": n.get("type",""), "ingested": n.get("ingested", False)} for n in _gdata.get("nodes", [])]
                    _edges = [{"source": e["src"], "target": e["dst"], "type": e.get("type","")} for e in _gdata.get("edges", [])]
                    push_update(json.dumps({"nodes": _nodes, "edges": _edges, "stats": _gdata.get("stats", {})}))
                except Exception:
                    pass

        console.print(Panel(
            f"Docs ingested:       {docs_in}\n"
            f"Chunks added:        {chunks_in}\n"
            f"LLM reasonings:      {len(reasonings)}\n"
            f"Novel entities (total): {len(state.discovered_entities)}\n"
            f"Frontier pending:    {len(state.frontier)}\n"
            f"Queries queued:      {len(state.query_queue)}",
            title="[bold green]Session complete[/bold green]",
        ))

        if not loop:
            break

        console.print(f"[dim]Next session in {interval}s (Ctrl-C to stop).[/dim]")
        time.sleep(interval)


# ── Project ────────────────────────────────────────────────────────────────

@project_app.command("new")
def project_new(
    name: str = typer.Argument(..., help="Project name (e.g. 'Sudan Conflict 2023')"),
    query: str = typer.Option(..., "--query", "-q", help="Research query for this project"),
    geography: str = typer.Option("", "--geography", "-g", help="Geographic focus"),
    switch: bool = typer.Option(True, help="Make this the active project immediately"),
) -> None:
    """Create a new research project and (optionally) make it active.

    Each project gets its own isolated corpus, agent state, sources register,
    ChromaDB collection, and search provenance graph.

    Example:
      ai4saw project new "Sudan Conflict" --query "Sudan civil war RSF atrocities 2023" --geography Sudan
    """
    from ai4saw.core.project import create_project, set_active_project, get_project_paths

    meta = create_project(name, research_query=query, geography=geography)
    paths = get_project_paths(meta.slug)
    console.print(Panel(
        f"Slug:         {meta.slug}\n"
        f"Query:        {meta.research_query}\n"
        f"Geography:    {meta.geography or '(not set)'}\n"
        f"Directory:    {paths['dir']}\n"
        f"Collection:   {paths['chroma_collection']}",
        title=f"[bold green]Project created: {meta.name}[/bold green]",
    ))
    if switch:
        set_active_project(meta.slug)
        console.print(f"[green]✓[/green] Active project set to [bold]{meta.slug}[/bold]")


@project_app.command("list")
def project_list() -> None:
    """List all projects and show which is active."""
    from ai4saw.core.project import list_projects, get_active_project, get_project_paths

    projects = list_projects()
    active = get_active_project()

    if not projects:
        console.print("[yellow]No projects yet. Run `ai4saw project new` to create one.[/yellow]")
        return

    table = Table(title="Research Projects", show_header=True)
    table.add_column("", width=2)
    table.add_column("Slug", style="cyan")
    table.add_column("Name")
    table.add_column("Geography")
    table.add_column("Query")
    table.add_column("Created")
    for p in projects:
        marker = "[bold green]✓[/bold green]" if p.slug == active else " "
        table.add_row(
            marker,
            p.slug,
            p.name,
            p.geography or "—",
            p.research_query[:50],
            p.created_at[:10],
        )
    console.print(table)
    if active:
        console.print(f"\n[dim]Active: {active}[/dim]")
    else:
        console.print("\n[dim]No active project. Use `ai4saw project switch <slug>` to activate.[/dim]")


@project_app.command("switch")
def project_switch(
    slug: str = typer.Argument(..., help="Project slug to activate"),
) -> None:
    """Switch the active project."""
    from ai4saw.core.project import set_active_project, load_project

    try:
        meta = load_project(slug)
    except FileNotFoundError:
        typer.echo(f"Project not found: {slug!r}", err=True)
        raise typer.Exit(1)
    set_active_project(slug)
    console.print(f"[green]✓[/green] Active project: [bold]{meta.name}[/bold] ({slug})")


@project_app.command("status")
def project_status(
    slug: Optional[str] = typer.Argument(None, help="Project slug (defaults to active project)"),
) -> None:
    """Show detailed status for a project: corpus, agent state, search graph."""
    from ai4saw.core.project import resolve_project, get_active_project, load_project
    from ai4saw.core.search_graph import SearchGraph

    effective = slug or get_active_project()
    if not effective:
        typer.echo("No project specified and no active project.", err=True)
        raise typer.Exit(1)

    try:
        meta = load_project(effective)
    except FileNotFoundError:
        typer.echo(f"Project not found: {effective!r}", err=True)
        raise typer.Exit(1)

    paths = resolve_project(effective)

    # Agent state stats
    agent_stats: dict = {}
    if paths["agent_state"].exists():
        try:
            from ai4saw.agents.agent_discover import load_agent_state, get_agent_summary
            state = load_agent_state(path=paths["agent_state"])
            agent_stats = get_agent_summary(state)
        except Exception:
            pass

    # Corpus stats
    corpus_files = list(paths["corpus"].glob("*.pdf")) if paths["corpus"].exists() else []
    sources_count = 0
    if paths["sources_csv"].exists():
        import csv as _csv
        with open(paths["sources_csv"], encoding="utf-8") as f:
            sources_count = sum(1 for _ in _csv.DictReader(f))

    # Search graph stats
    graph_stats: dict = {}
    if paths["search_graph"].exists():
        try:
            g = SearchGraph(paths["search_graph"])
            graph_stats = g.stats()
        except Exception:
            pass

    console.print(Panel(
        f"Name:          {meta.name}\n"
        f"Slug:          {meta.slug}\n"
        f"Query:         {meta.research_query}\n"
        f"Geography:     {meta.geography or '(not set)'}\n"
        f"Created:       {meta.created_at[:19]}\n"
        f"Directory:     {paths['dir']}",
        title="[bold cyan]Project[/bold cyan]",
    ))

    if agent_stats:
        console.print(Panel(
            f"Sessions:        {agent_stats.get('sessions', 0)}\n"
            f"URLs visited:    {agent_stats.get('urls_visited', 0)}\n"
            f"Docs ingested:   {agent_stats.get('docs_ingested', 0)}\n"
            f"Novel entities:  {agent_stats.get('novel_entities', 0)}\n"
            f"Queries queued:  {agent_stats.get('queries_queued', 0)}\n"
            f"Last run:        {agent_stats.get('last_run') or 'never'}",
            title="[bold cyan]Agent State[/bold cyan]",
        ))

    console.print(Panel(
        f"PDFs downloaded: {len(corpus_files)}\n"
        f"Sources registered: {sources_count}",
        title="[bold cyan]Corpus[/bold cyan]",
    ))

    if graph_stats:
        console.print(Panel(
            f"Total nodes:  {graph_stats['total_nodes']}\n"
            f"Total edges:  {graph_stats['total_edges']}\n"
            f"Node types:   {graph_stats['node_types']}\n"
            f"Edge types:   {graph_stats['edge_types']}",
            title="[bold cyan]Search Graph[/bold cyan]",
        ))
    else:
        console.print("[dim]Search graph: not built yet (run `discover agent` or `research`).[/dim]")


@project_app.command("graph")
def project_graph(
    slug: Optional[str] = typer.Argument(None, help="Project slug (defaults to active)"),
    format: str = typer.Option("stats", "--format", "-f",
                               help="Output format: stats | gexf | networkx"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                          help="Write to file (default: print to console)"),
) -> None:
    """Inspect or export the search provenance graph for a project.

    The graph records every query, source, URL, entity, and LLM-generated
    query — and how they connect. Use --format gexf to export for Gephi.

    Formats:
      stats     — print node/edge counts by type (default)
      gexf      — export GEXF XML (open in Gephi, yEd, or NetworkX)
      networkx  — export NetworkX node-link JSON

    Examples:
      ai4saw project graph                          # stats for active project
      ai4saw project graph sudan_conflict_2023      # stats for specific project
      ai4saw project graph --format gexf -o g.gexf  # export for Gephi
    """
    from ai4saw.core.project import get_active_project, load_project, get_project_paths
    from ai4saw.core.search_graph import SearchGraph

    effective = slug or get_active_project()
    if not effective:
        typer.echo("No project specified and no active project.", err=True)
        raise typer.Exit(1)

    try:
        load_project(effective)
    except FileNotFoundError:
        typer.echo(f"Project not found: {effective!r}", err=True)
        raise typer.Exit(1)

    paths = get_project_paths(effective)
    if not paths["search_graph"].exists():
        console.print("[yellow]No search graph yet — run `ai4saw research` or `ai4saw discover agent` first.[/yellow]")
        raise typer.Exit(0)

    g = SearchGraph(paths["search_graph"])

    if format == "stats":
        stats = g.stats()
        console.print(Panel(
            f"Nodes:  {stats['total_nodes']}\n"
            f"Edges:  {stats['total_edges']}\n\n"
            f"Node types:\n" +
            "\n".join(f"  {t}: {n}" for t, n in stats["node_types"].items()) +
            "\n\nEdge types:\n" +
            "\n".join(f"  {t}: {n}" for t, n in stats["edge_types"].items()),
            title=f"[bold cyan]Search Graph — {effective}[/bold cyan]",
        ))
        # Top entities by hits
        top_ents = g.top_nodes("entity", n=10)
        if top_ents:
            t = Table(title="Top discovered entities (by mention count)")
            t.add_column("Entity"); t.add_column("Hits", justify="right")
            for ent, hits in top_ents:
                t.add_row(ent, str(hits))
            console.print(t)
        # Top sources by hits
        top_srcs = g.top_nodes("source", n=10)
        if top_srcs:
            t = Table(title="Most-queried sources")
            t.add_column("Source"); t.add_column("Hits", justify="right")
            for src, hits in top_srcs:
                t.add_row(src, str(hits))
            console.print(t)

    elif format == "gexf":
        content = g.to_gexf()
        dest = output or paths["dir"] / "search_graph.gexf"
        dest.write_text(content, encoding="utf-8")
        console.print(f"[green]GEXF exported →[/green] {dest}")
        console.print(f"[dim]Open in Gephi: File → Open → select the .gexf file[/dim]")

    elif format == "networkx":
        import json as _json
        content = _json.dumps(g.to_networkx_dict(), indent=2)
        dest = output or paths["dir"] / "search_graph_nx.json"
        dest.write_text(content, encoding="utf-8")
        console.print(f"[green]NetworkX JSON exported →[/green] {dest}")
        console.print(f"[dim]Load with: import networkx as nx; G = nx.node_link_graph(json.load(open('{dest}')))[/dim]")

    else:
        typer.echo(f"Unknown format: {format!r}. Choose: stats, gexf, networkx", err=True)
        raise typer.Exit(1)


# ── Export ─────────────────────────────────────────────────────────────────

@export_app.command("all")
def export_all(
    ner_file: Optional[Path] = typer.Option(None),
    relations_file: Optional[Path] = typer.Option(None),
    events_file: Optional[Path] = typer.Option(None),
) -> None:
    """Export events (GeoJSON), relations, entities, and corpus stats."""
    from ai4saw.core.models import EventResult, NERResult, RelationResult
    from ai4saw.ingestion.embedder import get_vector_store
    from ai4saw.synthesis.export import export_all as _export_all

    ner_file       = ner_file       or _op("ner_results.json")
    relations_file = relations_file or _op("relation_results.json")
    events_file    = events_file    or _op("event_results.json")
    for p in [ner_file, relations_file, events_file]:
        if not p.exists():
            typer.echo(f"File not found: {p} — run `extract pipeline` first.", err=True)
            raise typer.Exit(1)

    ner_results = [NERResult(**r) for r in json.loads(ner_file.read_text())]
    rel_results = [RelationResult(**r) for r in json.loads(relations_file.read_text())]
    event_results = [EventResult(**r) for r in json.loads(events_file.read_text())]

    store = get_vector_store()
    raw = store._collection.get(include=["metadatas"])
    metadatas = raw.get("metadatas") or []

    with console.status("Exporting..."):
        outputs = _export_all(
            events=event_results, ner_results=ner_results,
            relation_results=rel_results, silence_candidates=[],
            chunk_metadatas=metadatas,
        )

    for name, path in outputs.items():
        console.print(f"[green]{name}[/green] → {path}")


# ── Eval ──────────────────────────────────────────────────────────────────

@eval_app.command("ner")
def eval_ner(
    gold: Path = typer.Argument(Path("eval/testdata/ner_gold.json")),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Run NER precision/recall/F1 benchmark against gold labels."""
    from eval.ner_benchmark import _display_report, run_benchmark

    output = output or Path("eval/results/ner_benchmark.json")
    report = run_benchmark(gold, output_path=output)
    _display_report(report)


@eval_app.command("rag")
def eval_rag(
    questions: Path = typer.Argument(Path("eval/testdata/rag_questions.json")),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Run RAGAS evaluation (faithfulness, relevancy, precision, recall)."""
    from eval.rag_eval import _display_report, run_ragas_eval

    output = output or Path("eval/results/rag_eval.json")
    report = run_ragas_eval(questions, output_path=output)
    _display_report(report)


@eval_app.command("judge")
def eval_judge(
    ner_file: Optional[Path] = typer.Option(None),
    relations_file: Optional[Path] = typer.Option(None),
    events_file: Optional[Path] = typer.Option(None),
    sample: int = typer.Option(20, help="Number of chunks to evaluate (20 for dev, 50+ for benchmarks)"),
    output: Optional[Path] = typer.Option(None),
    seed: int = typer.Option(42, help="Random seed for reproducible sampling"),
) -> None:
    """Score extraction quality using a frontier LLM as judge — no hand labels needed.

    Set JUDGE_MODEL in .env to use a different (stronger) model for judging
    than the extraction model. Self-judging introduces bias.

    Example: JUDGE_MODEL=anthropic/claude-3-5-sonnet (via OpenRouter)
    """
    from eval.llm_judge import display_report, run_judge

    ner_file       = ner_file       or _op("ner_results.json")
    relations_file = relations_file or _op("relation_results.json")
    events_file    = events_file    or _op("event_results.json")
    for p in [ner_file, relations_file, events_file]:
        if not p.exists():
            typer.echo(f"File not found: {p} — run `extract pipeline` first.", err=True)
            raise typer.Exit(1)

    output = output or Path("eval/results/judge_report.json")
    with console.status(f"LLM-as-Judge evaluating {sample} chunks..."):
        report = run_judge(ner_file, relations_file, events_file, sample, output, seed)
    display_report(report)


# ── Wipe ──────────────────────────────────────────────────────────────────

@app.command("wipe")
def wipe(
    project_slug: Optional[str] = typer.Option(
        None, "--project", "-p",
        help="Wipe a specific project (slug). Use 'all' to wipe all projects."
    ),
    state:    bool = typer.Option(True,  help="Wipe agent state (frontier, visited URLs, entities)"),
    chroma:   bool = typer.Option(False, "--chroma", help="Wipe ChromaDB collection(s)"),
    corpus:   bool = typer.Option(False, "--corpus", help="Delete downloaded corpus PDFs"),
    graph:    bool = typer.Option(False, "--graph",  help="Delete search provenance graph"),
    all_data: bool = typer.Option(False, "--all",    help="Wipe everything"),
    yes:      bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear session data so the next run starts fresh.

    With no --project flag, operates on the global (legacy) data only.
    With --project <slug>, wipes only that project's isolated data.
    With --project all, wipes ALL projects.

    Examples:
      ai4saw wipe                        # clear global agent state
      ai4saw wipe --all                  # clear everything global
      ai4saw wipe --project sudan_2023   # wipe one project
      ai4saw wipe --project all --all    # nuke all projects completely
    """
    import shutil
    from ai4saw.agents.agent_discover import AGENT_STATE_FILE, AGENT_LOG_FILE

    if all_data:
        state = chroma = corpus = graph = True

    # ── Project wipe ──────────────────────────────────────────────────────────
    if project_slug:
        from ai4saw.core.project import (
            list_projects, get_project_paths, load_project,
            get_active_project, clear_active_project,
        )
        import chromadb as _cdb

        if project_slug == "all":
            targets = [(p.slug, get_project_paths(p.slug)) for p in list_projects()]
            if not targets:
                console.print("[yellow]No projects to wipe.[/yellow]")
                raise typer.Exit(0)
        else:
            try:
                load_project(project_slug)
            except FileNotFoundError:
                typer.echo(f"Project not found: {project_slug!r}", err=True)
                raise typer.Exit(1)
            targets = [(project_slug, get_project_paths(project_slug))]

        console.print(f"\n[bold red]Wiping {len(targets)} project(s):[/bold red]")
        for slug, paths in targets:
            console.print(f"  [bold]{slug}[/bold]  →  {paths['dir']}")

        items_to_show = []
        if state:  items_to_show.append("agent state + log")
        if corpus: items_to_show.append("corpus PDFs")
        if graph:  items_to_show.append("search graph")
        if chroma: items_to_show.append("ChromaDB collection")
        console.print(f"  Scope: {', '.join(items_to_show) or 'nothing selected'}")

        if not items_to_show:
            console.print("[yellow]Nothing selected. Add --state/--corpus/--graph/--chroma or --all.[/yellow]")
            raise typer.Exit(0)

        if not yes:
            typer.confirm("\nProceed?", default=False, abort=True)

        for slug, paths in targets:
            if state:
                paths["agent_state"].unlink(missing_ok=True)
                paths["agent_log"].unlink(missing_ok=True)
                console.print(f"[green]✓[/green] [{slug}] Agent state cleared")
            if graph:
                paths["search_graph"].unlink(missing_ok=True)
                console.print(f"[green]✓[/green] [{slug}] Search graph cleared")
            if corpus:
                if paths["corpus"].exists():
                    pdfs = list(paths["corpus"].glob("*.pdf"))
                    for p in pdfs:
                        p.unlink(missing_ok=True)
                paths["sources_csv"].unlink(missing_ok=True)
                console.print(f"[green]✓[/green] [{slug}] Corpus PDFs + sources.csv cleared")
            if chroma:
                try:
                    from ai4saw.core.config import settings as _s
                    _client = _cdb.PersistentClient(path=str(_s.chroma_persist_dir))
                    _col = paths["chroma_collection"]
                    _client.delete_collection(_col)
                    console.print(f"[green]✓[/green] [{slug}] ChromaDB collection '{_col}' deleted")
                except Exception as exc:
                    console.print(f"[yellow]![/yellow] [{slug}] ChromaDB: {exc}")

            # If wiping all data, remove the whole project dir
            if all_data:
                shutil.rmtree(paths["dir"], ignore_errors=True)
                console.print(f"[green]✓[/green] [{slug}] Project directory deleted")
                # Clear active project pointer if we just wiped it
                if get_active_project() == slug:
                    clear_active_project()

        console.print("\n[bold green]Done.[/bold green]")
        return

    # ── Global (legacy) wipe ──────────────────────────────────────────────────
    items: list[tuple[str, Path | None]] = []
    if state:
        items += [
            ("Agent state (frontier + visited URLs + entities)", AGENT_STATE_FILE),
            ("Agent reasoning log", AGENT_LOG_FILE),
            ("Research errors log", Path("output/research_errors.log")),
            ("Research stderr log", Path("output/research_stderr.log")),
        ]
    if chroma:
        items.append(("ChromaDB (all embedded chunks)", Path("data/chroma")))
    if corpus:
        items.append(("Corpus PDFs", None))  # handled separately

    if not items and not corpus:
        console.print("[yellow]Nothing selected to wipe. Use --all for everything.[/yellow]")
        raise typer.Exit(0)

    console.print("\n[bold red]This will delete:[/bold red]")
    for label, _ in items:
        console.print(f"  [red]✗[/red] {label}")
    if corpus:
        n = len(list(Path("corpus").glob("*.pdf")))
        console.print(f"  [red]✗[/red] {n} corpus PDF files")

    if not yes:
        typer.confirm("\nProceed?", default=False, abort=True)

    for label, path in items:
        if path is None:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
        console.print(f"[green]✓[/green] Deleted {label}")

    if corpus:
        pdfs = list(Path("corpus").glob("*.pdf"))
        for p in pdfs:
            p.unlink(missing_ok=True)
        Path("corpus/sources.csv").write_text(
            "filename,title,source_url,doc_type,language,date_published,geography,licence,notes\n",
            encoding="utf-8"
        )
        console.print(f"[green]✓[/green] Deleted {len(pdfs)} corpus PDFs and reset sources.csv")

    console.print("\n[bold green]Done — next run starts fresh.[/bold green]")


# ── Info ──────────────────────────────────────────────────────────────────

@app.command("info")
def info() -> None:
    """Show current configuration and status of all data artefacts."""
    from ai4saw.core.config import settings
    from ai4saw.core.project import get_active_project, load_project

    active = get_active_project()
    if active:
        try:
            meta = load_project(active)
            console.print(Panel(
                f"Slug:     {meta.slug}\n"
                f"Name:     {meta.name}\n"
                f"Query:    {meta.research_query}\n"
                f"Geo:      {meta.geography or '—'}",
                title="[bold magenta]Active Project[/bold magenta]",
                border_style="magenta",
            ))
        except Exception:
            pass

    table = Table(title="AI4SAW Configuration", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Provider", settings.provider)
    table.add_row("Default model", settings.default_model)
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("ChromaDB", str(settings.chroma_persist_dir))
    table.add_row("Collection", settings.chroma_collection)
    table.add_row("Output dir", str(_op()))
    table.add_row("Data dir", str(_dp()))
    table.add_row("Prompts dir", str(settings.prompts_dir))
    table.add_row("Retrieval top-K", str(settings.retrieval_top_k))
    table.add_row("Re-rank top-N", str(settings.rerank_top_n))
    console.print(table)

    # Artefact status — all paths resolve to active project or legacy
    artefacts = {
        "ChromaDB": settings.chroma_persist_dir / "chroma.sqlite3",
        "Entity registry": _dp("entity_registry.json"),
        "Knowledge graph": _dp("knowledge_graph.json"),
        "NER results":     _op("ner_results.json"),
        "Relation results":_op("relation_results.json"),
        "Event results":   _op("event_results.json"),
        "Contradictions":  _op("contradictions.json"),
        "Network":         _op("network.json"),
    }
    status_table = Table(title="Data Artefacts", show_header=False)
    status_table.add_column("Artefact", style="cyan")
    status_table.add_column("Path", style="dim")
    status_table.add_column("Status")
    for name, path in artefacts.items():
        status_table.add_row(
            name,
            str(path),
            "[green]exists[/green]" if path.exists() else "[dim]not yet built[/dim]",
        )
    console.print(status_table)


# ── Research (one-command loop) ───────────────────────────────────────────────

def _parse_research_query(query: str) -> dict:
    """Use the LLM to extract entities, geography, and topics from a natural language query."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    prompt = f"""Extract research parameters from this natural language query.

Query: "{query}"

Output ONLY valid JSON — no text outside the braces:
{{
  "entities": ["entity1", "entity2"],
  "geography": "region/country name",
  "topics": ["topic1", "topic2"]
}}

Examples:
  "events in Sarajevo 1992-1995 war crimes" →
    {{"entities": ["Sarajevo", "Bosnian War"], "geography": "Bosnia", "topics": ["war crimes", "siege"]}}
  "Srebrenica massacre Mladic ICTY judgment" →
    {{"entities": ["Srebrenica", "Ratko Mladic", "ICTY"], "geography": "Bosnia", "topics": ["genocide", "tribunal"]}}
"""
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="You are a research assistant. Extract structured parameters from research queries."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        console.print(f"[yellow]Query parsing failed ({exc}), using query text as single entity.[/yellow]")
        return {"entities": [query], "geography": "unknown", "topics": []}


@app.command("research")
def research(
    query: Optional[str] = typer.Argument(None, help="Research query — omit to be prompted"),
    interval: int = typer.Option(0, help="Seconds between sessions (0 = continuous)"),
    frontier_batch: int = typer.Option(20, help="Frontier URLs to visit per session"),
    max_reasoning: int = typer.Option(8, help="Max docs to run LLM reasoning on per session"),
    seed_every: int = typer.Option(6, help="Re-seed frontier every N sessions"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    debug: bool = typer.Option(False, "--debug", help="Show all logs/errors in terminal (no Live UI)"),
    project: Optional[str] = typer.Option(None, "--project", "-p",
                                           help="Project slug (overrides active project)"),
) -> None:
    """One-command research loop — describe what you want, the system finds it."""
    import os, sys
    from loguru import logger as _logger
    from rich.live import Live
    from rich.console import Console as _Console
    from ai4saw.ui.startup import (
        make_processing_layout, prompt_query,
        show_splash_1, show_splash_2,
    )
    from ai4saw.agents.agent_discover import (
        AgentDiscoverState, DiscoveryReasoning,
        get_agent_summary, load_agent_state, run_agent_session_pipeline,
        save_agent_state, top_novel_entities, _seed_frontier,
        _llm_generate_seed_queries, prune_frontier,
    )
    from ai4saw.ui.dashboard import DashboardState, make_renderable

    from ai4saw.core.project import (
        resolve_project, create_project, set_active_project,
        get_project_paths, set_active_paths,
    )
    from ai4saw.core.search_graph import SearchGraph, set_active_graph
    from ai4saw.core.config import settings as _cfg

    # ── Project context — always required, auto-create from query if needed ────
    # First resolve explicitly supplied --project or active project
    _proj_paths = resolve_project(project)
    # If no project is active, auto-create one from the query string (after parsing)
    # We defer auto-creation until after query parsing so we have a good name.
    # Flag it so we can create it once we have the query.
    _auto_create_project = _proj_paths is None

    _state_path: Optional[Path] = None
    _graph_port: Optional[int] = None
    _sg = None

    if debug:
        # Debug mode: keep stderr, restore loguru, no splash, plain output
        _logger.remove()
        _logger.add(sys.stderr, level="DEBUG", colorize=True)
        console.print(f"[bold yellow]DEBUG MODE[/bold yellow] — all logs visible, no UI")
        if not query:
            query = console.input("[bold blue]Research query:[/bold blue] ").strip()
        _run_debug(query, yes, frontier_batch, max_reasoning, seed_every, interval, _state_path)
        return

    # Kill loguru — events surface via on_event callbacks instead.
    _logger.remove()

    # ── Splash 1 (always) ─────────────────────────────────────────────────────
    show_splash_1(console)

    # ── Query prompt (if not provided on command line) ────────────────────────
    if not query:
        query = prompt_query(console)

    # ── Splash 2 (always) ─────────────────────────────────────────────────────
    show_splash_2(console)

    # ── Initialising Research — full screen ───────────────────────────────────
    entities: list[str] = []
    geography: str = "unknown"
    topics: list[str] = []
    steps: list[tuple[str, str]] = [("dim", "  Asking LLM to extract entities and geography…")]

    with Live(make_processing_layout(query, steps), screen=True, console=console, refresh_per_second=4) as proc_live:
        try:
            parsed = _parse_research_query(query)
            entities = parsed.get("entities") or [query]
            geography = parsed.get("geography") or "unknown"
            topics = parsed.get("topics") or []
            steps = [
                ("green", f"  ✓  Entities:  {', '.join(entities)}"),
                ("green", f"  ✓  Geography: {geography}"),
            ]
            if topics:
                steps.append(("green", f"  ✓  Topics:    {', '.join(topics)}"))
            # ── Auto-create project if none active ────────────────────────────
            if _auto_create_project:
                try:
                    _meta = create_project(
                        name=query[:48],
                        research_query=query,
                        geography=geography,
                    )
                    set_active_project(_meta.slug)
                    _proj_paths = get_project_paths(_meta.slug)
                    steps.append(("green", f"  ✓  Project:   {_meta.slug}"))
                except Exception as _pe:
                    steps.append(("yellow", f"  ⚠  Project auto-create failed: {_pe}"))
            if _proj_paths:
                set_active_paths(_proj_paths)
                _state_path = _proj_paths["agent_state"]
                _cfg.chroma_collection = _proj_paths["chroma_collection"]
                _sg = SearchGraph(_proj_paths["search_graph"])
                set_active_graph(_sg)
                try:
                    from ai4saw.ui.graph_server import start_graph_server
                    _graph_port = start_graph_server(_proj_paths["search_graph"], project_name=_proj_paths["dir"].name)
                    steps.append(("green", f"  ✓  Graph UI:  http://localhost:{_graph_port}"))
                except Exception:
                    pass
            steps.append(("dim", "  ✓  Loading agent state…"))
            steps.append(("dim", "  Starting research loop…"))
        except Exception as exc:
            steps = [("red", f"  ✗  Query parsing failed: {exc}"),
                     ("dim", "  Using query text as-is…")]
            entities = [query]
            geography = "unknown"
        proc_live.update(make_processing_layout(query, steps))
        time.sleep(2.0)

    # ── Load or init agent state ──────────────────────────────────────────────
    agent_state = load_agent_state(path=_state_path)
    if not agent_state.initial_entities:
        agent_state.initial_entities = entities
    else:
        for e in entities:
            if e not in agent_state.initial_entities:
                agent_state.initial_entities.append(e)

    # ── Dashboard state — pre-populate from saved agent state ────────────────
    dash = DashboardState(
        query=query,
        geography=geography,
        entities=entities,
        graph_url=f"http://localhost:{_graph_port}" if _graph_port else "",
        project_name=_proj_paths["dir"].name if _proj_paths else "",
    )
    # Restore prior session data immediately so UI isn't empty on restart
    _prior = get_agent_summary(agent_state)
    dash.frontier_size    = _prior["frontier_size"]
    dash.novel_entities   = _prior["novel_entities"]
    dash.queries_queued   = _prior["queries_queued"]
    dash.queries_executed = _prior["queries_executed"]
    dash.top_entities     = _prior["top_novel_entities"]
    dash.docs_ingested    = agent_state.total_docs_ingested
    dash.docs_skipped     = agent_state.total_docs_skipped
    dash.urls_visited     = _prior["urls_visited"]
    dash.goals            = list(agent_state.goals)
    dash.goals_updated_at = agent_state.goals_updated_at
    dash.chunks_added     = agent_state.total_chunks_added
    dash.session          = agent_state.session_count
    # Replay last N reasoning log entries into the feed
    from ai4saw.agents.agent_discover import AGENT_LOG_FILE
    if AGENT_LOG_FILE.exists():
        try:
            _log_lines = AGENT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
            for _line in _log_lines[-5:]:
                _entry = json.loads(_line)
                dash.push("reason", f"[prior] {_entry.get('source_title','')[:55]}")
                for _q in _entry.get("generated_queries", [])[:2]:
                    dash.push("query", _q[:80])
            # Pre-fill reasoning panel with last entry
            if _log_lines:
                _last = json.loads(_log_lines[-1])
                dash.set_reasoning(
                    doc=_last.get("source_title", ""),
                    entities=_last.get("novel_entities", []),
                    queries=_last.get("generated_queries", []),
                    why=_last.get("reasoning", ""),
                )
        except Exception:
            pass

    contact_email = ""
    try:
        from ai4saw.core.config import settings as _s
        contact_email = getattr(_s, "contact_email", "")
    except Exception:
        pass

    session_number = 0

    # Route stderr through the Live feed so errors appear on screen
    _old_stderr = sys.stderr
    _stderr_log = Path("output/research_stderr.log")
    _stderr_log.parent.mkdir(parents=True, exist_ok=True)

    # Catch ALL unhandled exceptions and write to log before terminal is restored
    import sys as _sys
    def _crash_handler(exc_type, exc_val, exc_tb):
        import traceback as _tb_crash
        _crash_text = "".join(_tb_crash.format_exception(exc_type, exc_val, exc_tb))
        try:
            _error_log.write_text(_crash_text, encoding="utf-8")
        except Exception:
            pass
        _sys.__excepthook__(exc_type, exc_val, exc_tb)
    _sys.excepthook = _crash_handler

    class _FeedStderr:
        """Routes stderr writes into the Live feed error panel."""
        def write(self, msg: str) -> None:
            msg = msg.strip()
            if msg and hasattr(self, '_event_fn'):
                self._event_fn("error", msg[:120])
            _stderr_log.open("a", encoding="utf-8").write(msg + "\n")
        def flush(self) -> None: pass
    _feed_stderr = _FeedStderr()
    sys.stderr = _feed_stderr

    try:
        with Live(make_renderable(dash), refresh_per_second=8, screen=True) as live:

            _feed_stderr._event_fn = lambda etype, msg: _event("error", msg)

            def _event(etype: str, msg: str) -> None:
                # model_* events only update status indicators, never the feed
                if not etype.startswith("model_"):
                    dash.push(etype, msg)
                dash.current_action = msg if etype == "info" else dash.current_action
                if etype == "ingest":
                    dash.docs_ingested += 1
                    dash.urls_visited += 1
                elif etype == "skip":
                    dash.docs_skipped += 1
                    dash.urls_visited += 1
                elif etype == "error":
                    dash.urls_visited += 1
                elif etype == "reason":
                    # Only show "in progress" if we have no result yet — never overwrite a real result
                    if dash.reasoning_count == 0:
                        dash.last_doc = msg.replace("Reasoning: ", "")
                        dash.last_why = "⟳ Reasoning in progress…"
                        dash.last_entities = []
                        dash.last_queries = []
                elif etype == "model_prescreen":
                    dash.prescreen_status = msg
                    if msg == "running":
                        dash.prescreen_calls += 1
                elif etype == "model_reason":
                    dash.reason_status = msg
                    if msg == "running":
                        dash.reason_calls += 1
                elif etype == "model_embed":
                    dash.embed_status = "running" if msg.startswith("running") else "idle"
                    if msg.startswith("running"):
                        dash.embed_calls += 1
                elif etype == "query":
                    dash.recent_queries.append(msg)
                    if len(dash.recent_queries) > 50:
                        dash.recent_queries = dash.recent_queries[-50:]
                live.update(make_renderable(dash))

            def _on_reasoning(r: object) -> None:
                entities = getattr(r, "novel_entities", [])
                queries  = getattr(r, "generated_queries", [])
                why      = getattr(r, "reasoning", "")
                doc      = getattr(r, "source_title", "")
                dash.set_reasoning(doc=doc, entities=entities, queries=queries, why=why)
                # Refresh analysis panels
                try:
                    _summary = get_agent_summary(agent_state)
                    dash.top_entities     = _summary["top_novel_entities"]
                    dash.novel_entities   = _summary["novel_entities"]
                    dash.queries_queued   = _summary["queries_queued"]
                    dash.queries_executed = _summary["queries_executed"]
                    dash.urls_visited     = _summary["urls_visited"]
                except Exception:
                    pass
                # Confirm in feed
                if entities:
                    _event("query", f"Reasoning ×{dash.reasoning_count}: {', '.join(entities[:3])}")
                else:
                    _event("info", f"Reasoning ×{dash.reasoning_count}: no new entities — queries queued")
                # Set/update goals after every 3rd reasoning — don't wait for session end
                if dash.reasoning_count % 3 == 1 or not agent_state.goals:
                    def _update_goals() -> None:
                        try:
                            from ai4saw.agents.agent_discover import llm_set_goals as _lsg
                            new_goals = _lsg(
                                state=agent_state,
                                geography=geography,
                                research_query=query,
                                narrator_text=dash.narrator_text,
                            )
                            if new_goals:
                                agent_state.goals = new_goals
                                from datetime import datetime as _dt
                                agent_state.goals_updated_at = _dt.now().strftime("%H:%M:%S")
                                dash.goals = new_goals
                                dash.goals_updated_at = agent_state.goals_updated_at
                                live.update(make_renderable(dash))
                        except Exception:
                            pass
                    _threading.Thread(target=_update_goals, daemon=True).start()
                live.update(make_renderable(dash))

            def _narrate() -> None:
                dash.current_action = "Generating research summary…"
                live.update(make_renderable(dash))
                try:
                    from datetime import datetime as _dt
                    dash.narrator_text = _llm_narrate(
                        query=query,
                        geography=geography,
                        docs_ingested=dash.docs_ingested,
                        docs_skipped=dash.docs_skipped,
                        top_entities=dash.top_entities,
                        last_reasoning_doc=dash.last_doc,
                        last_reasoning_why=dash.last_why,
                        last_queries=dash.last_queries,
                        novel_entities=dash.last_entities,
                    )
                    dash.narrator_updated_at = _dt.now().strftime("%H:%M:%S")
                except Exception:
                    pass
                live.update(make_renderable(dash))

            # Generate opening summary + initial goals immediately on start
            def _initial_start() -> None:
                try:
                    _narrate()
                except Exception:
                    pass
                try:
                    from ai4saw.agents.agent_discover import llm_set_goals as _lsg
                    from datetime import datetime as _dt
                    new_goals = _lsg(state=agent_state, geography=geography,
                                     research_query=query, narrator_text=dash.narrator_text)
                    if new_goals:
                        agent_state.goals = new_goals
                        agent_state.goals_updated_at = _dt.now().strftime("%H:%M:%S")
                        dash.goals = new_goals
                        dash.goals_updated_at = agent_state.goals_updated_at
                        live.update(make_renderable(dash))
                except Exception:
                    pass
            import threading as _threading
            _threading.Thread(target=_initial_start, daemon=True).start()
            _error_log = Path("output/research_errors.log")
            _error_log.parent.mkdir(parents=True, exist_ok=True)
            _error_log.write_text("", encoding="utf-8")  # clear on each run
            _spinners = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

            # ── LLM generates targeted seed queries before session 1 ──────────
            # Fixed DDG templates run once for initial discovery; after that the
            # LLM controls what to search for based on research context.
            if agent_state.session_count == 0 and not agent_state.query_queue:
                dash.current_action = "LLM generating initial targeted seed queries…"
                live.update(make_renderable(dash))
                try:
                    _initial_seeds = _llm_generate_seed_queries(
                        agent_state, geography, query, n=10
                    )
                    for _sq in _initial_seeds:
                        agent_state.query_queue.append(_sq)
                    dash.push("info", f"LLM seeded {len(_initial_seeds)} targeted queries")
                    live.update(make_renderable(dash))
                except Exception:
                    pass

            while True:
                session_number += 1
                dash.session = agent_state.session_count + 1

                # Summarise what we know at the START of each session
                try:
                    _narrate()
                except Exception:
                    pass

                dash.current_action = f"Session {dash.session} — seeding frontier…"
                live.update(make_renderable(dash))

                # With 4 parallel fetchers, frontier drains fast — reseed at 50 not 5
                frontier_low = len(agent_state.frontier) < 50
                _drifting = bool(dash.drift_warning)
                needs_seed = frontier_low or (agent_state.session_count % seed_every == 0) or _drifting

                # Always define _seed_done so _run_session closure can reference it
                _seed_done = _threading.Event()
                _seed_error: list[Exception] = []
                if not needs_seed:
                    _seed_done.set()  # mark as already done if no seeding needed

                if needs_seed:
                    seed_reason = (
                        "drift — LLM taking over seeding" if _drifting
                        else ("frontier empty" if frontier_low else f"every {seed_every} sessions")
                    )
                    dash.push("info", f"Re-seeding ({seed_reason})…")

                    # On drift: prune junk frontier + LLM generates targeted queries
                    if _drifting:
                        _anchors = list(entities) + [geography]
                        pruned = prune_frontier(agent_state, _anchors, keep_top=200)
                        if pruned:
                            dash.push("info", f"Pruned {pruned} off-topic frontier items")
                        dash.current_action = "Drift detected — LLM generating targeted seed queries…"
                        live.update(make_renderable(dash))
                        llm_seeds = _llm_generate_seed_queries(
                            agent_state, geography, query, n=8
                        )
                        if llm_seeds:
                            for sq in llm_seeds:
                                if sq not in agent_state.executed_queries and sq not in agent_state.query_queue:
                                    agent_state.query_queue.append(sq)
                            dash.push("info", f"LLM generated {len(llm_seeds)} targeted seed queries")
                            dash.drift_warning = ""  # cleared — LLM has taken control
                        live.update(make_renderable(dash))

                    _use_apis = True  # always run OpenAlex/IA/arXiv/GDELT/S2 on seed
                    # Fixed DDG templates only run on session 0 as a bootstrap.
                    # From session 1 onwards, LLM-generated queries are the only DDG source.
                    _skip_fixed_seed = agent_state.session_count > 0
                    # After session 0, use top discovered entities for Wikipedia/CrossRef
                    if agent_state.session_count > 0 and agent_state.discovered_entities:
                        seed_entities = (
                            list(agent_state.initial_entities) +
                            top_novel_entities(agent_state, n=5)
                        )
                    else:
                        seed_entities = list(agent_state.initial_entities)
                    if _skip_fixed_seed:
                        seed_label = "LLM queries · Wikipedia · CrossRef…"
                    else:
                        seed_label = "DDG · Wikipedia · CrossRef…"
                    dash.current_action = f"Seeding frontier: {seed_label}"
                    live.update(make_renderable(dash))

                    # ALL seeding runs in background — session streams items as they arrive
                    from ai4saw.agents.agent_discover import (
                        _execute_novel_query, _frontier_priority, _add_to_frontier,
                    )
                    llm_qs = list(agent_state.query_queue)
                    agent_state.query_queue = []

                    def _run_seed() -> None:
                        try:
                            # DDG first (fast, ~30s) — immediately adds URLs
                            _all_ents = list(agent_state.initial_entities) + list(agent_state.discovered_entities.keys())
                            for _q in llm_qs:
                                _event("info", f"LLM query → {_q[:70]}")
                                try:
                                    _docs = _execute_novel_query(_q, _all_ents, agent_state)
                                    for _d in _docs:
                                        _pri = _frontier_priority(_d.relevance_score, _d.url, agent_state)
                                        _add_to_frontier(agent_state, _d.url, _pri, _d.trigger_entity, _d.source)
                                except Exception:
                                    pass
                            # Then slower API sources (Wikipedia/CrossRef/OpenAlex/etc)
                            _seed_frontier(
                                llm_qs or list(agent_state.initial_entities),
                                agent_state,
                                contact_email=contact_email,
                                on_event=_event,
                                use_api_sources=_use_apis,
                                use_ddg=False,
                            )
                        except Exception as _e:
                            _seed_error.append(_e)
                            import traceback as _tb
                            _error_log.write_text(_tb.format_exc(), encoding="utf-8")
                        finally:
                            _seed_done.set()

                    _threading.Thread(target=_run_seed, daemon=True).start()

                _sess_done = _threading.Event()
                _sess_result: list = []
                _sess_error: list[Exception] = []

                def _run_session() -> None:
                    # Start immediately — fetchers will wait for frontier items themselves
                    try:
                        result = run_agent_session_pipeline(
                            state=agent_state,
                            geography=geography,
                            frontier_batch=frontier_batch,
                            max_reasoning=max_reasoning,
                            on_event=_event,
                            on_reasoning=_on_reasoning,
                            seeding_done=_seed_done,
                        )
                        _sess_result.append(result)
                    except Exception as _e:
                        _sess_error.append(_e)
                        import traceback as _tb
                        _error_log.write_text(_tb.format_exc(), encoding="utf-8")
                    finally:
                        _sess_done.set()

                # Launch both concurrently — seed fills frontier while session drains it
                dash.current_action = "Seeding + draining frontier in parallel…"
                live.update(make_renderable(dash))
                _threading.Thread(target=_run_session, daemon=True).start()

                _spin_i = 0
                _ticks = 0
                _MAX_TICKS = int(600 / 0.12)
                _REFRESH_EVERY  = int(5 / 0.12)    # refresh analysis panels every ~5s
                _NARRATE_EVERY  = int(120 / 0.12)  # narrate every ~2 minutes
                _narrate_running = [False]
                while not _sess_done.wait(timeout=0.12):
                    seeding_done = _seed_done.is_set()
                    status = "draining" if seeding_done else "seeding + draining"
                    dash.current_action = f"{_spinners[_spin_i % len(_spinners)]} {status}…"
                    dash.frontier_size = len(agent_state.frontier)
                    # Refresh analysis panels from live agent state every 5s
                    if _ticks % _REFRESH_EVERY == 0:
                        _s = get_agent_summary(agent_state)
                        dash.top_entities     = _s["top_novel_entities"]
                        dash.novel_entities   = _s["novel_entities"]
                        dash.queries_queued   = _s["queries_queued"]
                        dash.queries_executed = _s["queries_executed"]
                        dash.urls_visited     = _s["urls_visited"]
                        dash.chunks_added     = agent_state.total_chunks_added
                        # Source breakdown from visited URLs
                        _src_counts: dict[str, int] = {}
                        for _info in list(agent_state.visited_urls.values())[-500:]:
                            _src = _info.get("source", "web") if isinstance(_info, dict) else "web"
                            _src_counts[_src] = _src_counts.get(_src, 0) + 1
                        dash.source_counts = _src_counts
                        # Session history milestone
                        if agent_state.session_count > len(dash.session_history):
                            dash.session_history.append((
                                agent_state.session_count,
                                dash.docs_ingested,
                                dash.docs_skipped,
                            ))
                    # Narrate every 2 minutes — not tied to session completion
                    if _ticks % _NARRATE_EVERY == 0 and not _narrate_running[0] and dash.docs_ingested > 0:
                        _narrate_running[0] = True
                        def _bg_narrate() -> None:
                            try:
                                _narrate()
                            except Exception:
                                pass
                            finally:
                                _narrate_running[0] = False
                        _threading.Thread(target=_bg_narrate, daemon=True).start()
                    live.update(make_renderable(dash))
                    _spin_i += 1
                    _ticks += 1
                    if _ticks >= _MAX_TICKS:
                        dash.push("error", "Session timed out (10 min) — killing and retrying")
                        _sess_done.set()
                        break

                if _seed_error:
                    dash.push("error", f"Seed failed: {str(_seed_error[0])[:60]}")
                dash.frontier_size = len(agent_state.frontier)

                if _sess_error:
                    import traceback as _tb2
                    _error_log.write_text(_tb2.format_exc(), encoding="utf-8")
                    dash.push("error", f"Session error: {str(_sess_error[0])[:60]}")
                    dash.current_action = f"Session {dash.session} failed — retrying next cycle"
                    live.update(make_renderable(dash))
                    time.sleep(5)
                elif not _sess_result:
                    # Session exited without result — treat as 0-doc session and continue
                    dash.push("error", "Session returned no result — continuing")
                    live.update(make_renderable(dash))
                else:
                    docs_in, chunks_in, reasonings = _sess_result[0]

                    for item in getattr(agent_state, "visited_urls", {}).values():
                        src = item.get("source", "web") if isinstance(item, dict) else "web"
                        dash.record_source(src)

                    dash.docs_ingested += docs_in
                    dash.chunks_added += chunks_in
                    dash.docs_skipped = agent_state.total_docs_skipped
                    summary = get_agent_summary(agent_state)
                    dash.frontier_size    = summary["frontier_size"]
                    dash.novel_entities   = summary["novel_entities"]
                    dash.queries_queued   = summary["queries_queued"]
                    dash.queries_executed = summary["queries_executed"]
                    dash.top_entities     = summary["top_novel_entities"]
                    dash.urls_visited     = summary["urls_visited"]
                    dash.session_history.append((dash.session, docs_in, dash.docs_skipped))

                    # Drift detection — warn if last 3 sessions were >80% skips
                    if len(dash.session_history) >= 3:
                        recent = dash.session_history[-3:]
                        skip_pct = sum(sk for _, _, sk in recent) / max(1, sum(i + sk for _, i, sk in recent))
                        if skip_pct > 0.8:
                            dash.drift_warning = (
                                f"Topic drift detected — {int(skip_pct*100)}% of recent documents "
                                f"rejected as off-topic. Consider refining the query."
                            )
                        else:
                            dash.drift_warning = ""
                    dash.current_action   = f"Session {dash.session} complete — {docs_in} ingested, {len(reasonings)} reasoned"
                    live.update(make_renderable(dash))
                    save_agent_state(agent_state, path=_state_path)
                    if _proj_paths:
                        _sg.save()
                        if _graph_port:
                            try:
                                from ai4saw.ui.graph_server import push_update
                                import json as _json
                                _gdata = json.loads(_proj_paths["search_graph"].read_text(encoding="utf-8"))
                                _nodes = [{"id": n["id"], "label": n.get("label",""), "type": n.get("type",""), "ingested": n.get("ingested", False)} for n in _gdata.get("nodes", [])]
                                _edges = [{"source": e["src"], "target": e["dst"], "type": e.get("type","")} for e in _gdata.get("edges", [])]
                                push_update(_json.dumps({"nodes": _nodes, "edges": _edges, "stats": _gdata.get("stats", {})}))
                            except Exception:
                                pass

                if interval > 0:
                    dash.current_action = f"Sleeping {interval}s…"
                    live.update(make_renderable(dash))
                    time.sleep(interval)

    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = _old_stderr


def _llm_narrate(
    query: str,
    geography: str,
    docs_ingested: int,
    docs_skipped: int,
    top_entities: list,
    last_reasoning_doc: str,
    last_reasoning_why: str,
    last_queries: list,
    novel_entities: list,
) -> str:
    """Ask the LLM to narrate current research progress in plain English."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from ai4saw.core.providers import get_llm

    entity_list = ", ".join(f"{e} ({n})" for e, n in top_entities[:8]) if top_entities else "none yet"
    query_list  = "\n".join(f"  - {q}" for q in last_queries[:3]) if last_queries else "  none yet"
    novel_list  = ", ".join(novel_entities[:6]) if novel_entities else "none"

    if docs_ingested == 0:
        prompt = f"""You are a research intelligence analyst. A researcher has just started an investigation.

Research query: "{query}"
Geography: {geography}

Write 2–3 sentences describing what this research is about, what kinds of sources and evidence will be \
sought, and what the most important questions to answer are. Be specific about the subject matter. \
Write in present tense as if briefing a colleague at the start of an investigation. No preamble."""
    else:
        prompt = f"""You are helping a researcher study: "{query}" (geography: {geography}).

Current progress:
- Documents ingested: {docs_ingested}  |  Skipped as irrelevant: {docs_skipped}
- Most-mentioned entities: {entity_list}
- Last document reasoned on: {last_reasoning_doc or "none yet"}
- Why it was significant: {last_reasoning_why or "n/a"}
- Novel entities just discovered: {novel_list}
- Search queries now queued:
{query_list}

Write 2–3 sentences summarising what has been found so far and what the most promising leads are.
Be specific — name actual entities, documents, and why they matter.
Write in present tense as if giving a live briefing. No preamble."""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a research intelligence analyst giving a concise live briefing."),
        HumanMessage(content=prompt),
    ])
    return response.content.strip()


def _run_debug(
    query: str,
    yes: bool,
    frontier_batch: int,
    max_reasoning: int,
    seed_every: int,
    interval: int,
    state_path: Optional[Path] = None,
) -> None:
    """Debug mode: no Live UI, all output to terminal, full tracebacks visible."""
    import threading as _th
    from ai4saw.agents.agent_discover import (
        get_agent_summary, load_agent_state, run_agent_session_pipeline,
        save_agent_state, top_novel_entities, _seed_frontier,
        _llm_generate_seed_queries, AGENT_LOG_FILE,
    )

    parsed = _parse_research_query(query)
    entities: list[str] = parsed.get("entities") or [query]
    geography: str = parsed.get("geography") or "unknown"
    console.print(f"[green]Entities:[/green] {entities}  [green]Geography:[/green] {geography}")

    agent_state = load_agent_state(path=state_path)
    if not agent_state.initial_entities:
        agent_state.initial_entities = entities

    if not yes:
        typer.confirm("Start debug research loop?", default=True, abort=True)

    def _on_event(etype: str, msg: str) -> None:
        colour = {"ingest": "green", "skip": "red", "error": "red", "query": "yellow",
                  "reason": "cyan", "info": "dim"}.get(etype, "white")
        console.print(f"[{colour}][{etype}][/{colour}] {msg}")

    session = 0
    while True:
        session += 1
        console.rule(f"[bold]Session {session}[/bold]")

        # Seed
        llm_qs = list(agent_state.query_queue)
        agent_state.query_queue = []
        if not llm_qs and agent_state.session_count == 0:
            console.print("[dim]Generating LLM seed queries…[/dim]")
            llm_qs = _llm_generate_seed_queries(agent_state, geography, query, n=10)

        if llm_qs or len(agent_state.frontier) < 50:
            console.print(f"[dim]Seeding with {len(llm_qs)} LLM queries…[/dim]")
            _seed_frontier(llm_qs or entities, agent_state,
                           on_event=_on_event, use_api_sources=True)

        console.print(f"[dim]Frontier: {len(agent_state.frontier)} URLs[/dim]")
        console.print("[dim]Running session pipeline…[/dim]")

        # Run session (blocking, full tracebacks visible)
        docs_in, chunks_in, reasonings = run_agent_session_pipeline(
            state=agent_state,
            geography=geography,
            frontier_batch=frontier_batch,
            max_reasoning=max_reasoning,
            on_event=_on_event,
        )

        summary = get_agent_summary(agent_state)
        console.print(
            f"[green]✓ {docs_in} ingested[/green]  "
            f"[yellow]⬡ {chunks_in} chunks[/yellow]  "
            f"[blue]⋯ {summary['frontier_size']} frontier[/blue]  "
            f"[magenta]✦ {summary['novel_entities']} entities[/magenta]"
        )
        save_agent_state(agent_state, path=state_path)
        from ai4saw.core.search_graph import g_save
        g_save()

        if interval > 0:
            console.print(f"[dim]Sleeping {interval}s…[/dim]")
            time.sleep(interval)


if __name__ == "__main__":
    app()
