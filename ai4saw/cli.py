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
)
console = Console()

# ── Sub-app groups ──────────────────────────────────────────────────────────

ingest_app  = typer.Typer(help="Ingest documents into ChromaDB.")
extract_app = typer.Typer(help="Run extraction (NER, relations, events, entity resolution).")
graph_app   = typer.Typer(help="Knowledge graph: build, query (GraphRAG), multi-hop agent.")
query_app   = typer.Typer(help="Standard vector-search RAG Q&A.")
analyze_app = typer.Typer(help="Contradiction detection and perpetrator network analysis.")
discover_app = typer.Typer(help="Active corpus discovery via ReliefWeb and GDELT.")
export_app  = typer.Typer(help="Export structured outputs (JSON, GeoJSON, GEXF).")
eval_app    = typer.Typer(help="Run evaluation benchmarks.")

app.add_typer(ingest_app,   name="ingest")
app.add_typer(extract_app,  name="extract")
app.add_typer(graph_app,    name="graph")
app.add_typer(query_app,    name="query")
app.add_typer(analyze_app,  name="analyze")
app.add_typer(discover_app, name="discover")
app.add_typer(export_app,   name="export")
app.add_typer(eval_app,     name="eval")


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
    output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
    delay: float = typer.Option(0.25, help="Delay between LLM calls (seconds)"),
) -> None:
    """Run full extraction pipeline (NER + relations + events) on all indexed chunks."""
    from ai4saw.extraction.events import classify_events_batch
    from ai4saw.extraction.ner import extract_entities_batch
    from ai4saw.extraction.relations import extract_relations_batch
    from ai4saw.ingestion.embedder import get_vector_store

    store = get_vector_store()
    result = store._collection.get(include=["documents", "metadatas"])
    texts: list[str] = result.get("documents") or []
    ids: list[str] = result.get("ids") or []
    pairs = list(zip(texts, ids))

    if not pairs:
        typer.echo("No documents in ChromaDB. Run `ai4saw ingest` first.", err=True)
        raise typer.Exit(1)

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
    ner_file: Path = typer.Option(Path("output/ner_results.json")),
    cosine_threshold: float = typer.Option(0.88, help="Cosine similarity threshold (0–1)"),
    fuzzy_threshold: float = typer.Option(72.0, help="String fuzzy match threshold (0–100)"),
    output: Path = typer.Option(Path("data/entity_registry.json"), help="Output path"),
) -> None:
    """Resolve NER entities across the corpus into a canonical registry.

    Merges aliases ("Armed-Group-Beta", "Rapid Support Forces", "the paramilitaries") into
    single canonical entities using embedding similarity + string matching.
    Required before building the knowledge graph.
    """
    from ai4saw.core.models import NERResult
    from ai4saw.synthesis.entity_resolution import resolve_entities, save_entity_registry

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
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    registry_file: Path = typer.Option(Path("data/entity_registry.json")),
    min_confidence: float = typer.Option(0.5, help="Minimum relation confidence to include"),
    output: Path = typer.Option(Path("data/knowledge_graph.json")),
) -> None:
    """Build the knowledge graph from extracted relations and resolved entities.

    Run `extract pipeline` and `extract resolve` before this command.
    The knowledge graph is used by `graph query`, `graph agent`, and network analysis.
    """
    from ai4saw.core.models import RelationResult
    from ai4saw.retrieval.graph_rag import build_knowledge_graph, save_knowledge_graph
    from ai4saw.synthesis.entity_resolution import load_entity_registry

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
    graph_file: Path = typer.Option(Path("data/knowledge_graph.json")),
    combine_vector: bool = typer.Option(True, help="Also run vector search and combine"),
) -> None:
    """Query the knowledge graph (GraphRAG) — structural + semantic retrieval.

    Identifies entities mentioned in the question, extracts their neighbourhood
    from the graph, and combines with vector search results.

    Use --at to filter the graph to a specific point in time:
      ai4saw graph query "Drina Corps command" --at YYYY-MM-DD
    """
    from ai4saw.retrieval.graph_rag import graph_context_for_query, load_knowledge_graph

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
    events_file: Path = typer.Option(Path("output/event_results.json")),
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    min_confidence: float = typer.Option(0.65, help="Minimum LLM confidence to report a pair"),
    max_pairs: int = typer.Option(100, help="Maximum candidate pairs to assess (controls cost)"),
    output: Path = typer.Option(Path("output/contradictions.json")),
) -> None:
    """Detect conflicting claims across source documents.

    Uses a two-pass approach: cheap candidate generation (grouping by
    location/time), then LLM verification per candidate pair. Contradictions
    are not discarded — they are ranked by confidence and reported for
    researcher review.
    """
    from ai4saw.core.models import EventResult, RelationResult
    from ai4saw.synthesis.contradiction import detect_contradictions

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
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    registry_file: Optional[Path] = typer.Option(
        None, help="Entity registry (from extract resolve) — enables canonicalisation"
    ),
    min_confidence: float = typer.Option(0.5, help="Minimum relation confidence"),
    output: Path = typer.Option(Path("output/network.json")),
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
    output: Path = typer.Option(Path("output/discovered_documents.json")),
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

    if from_registry:
        registry_path = Path("data/entity_registry.json")
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

    output.parent.mkdir(parents=True, exist_ok=True)
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

    contact_email = getattr(settings, "contact_email", "")
    state = load_agent_state()

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

        save_agent_state(state)

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


# ── Export ─────────────────────────────────────────────────────────────────

@export_app.command("all")
def export_all(
    ner_file: Path = typer.Option(Path("output/ner_results.json")),
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    events_file: Path = typer.Option(Path("output/event_results.json")),
) -> None:
    """Export events (GeoJSON), relations, entities, and corpus stats."""
    from ai4saw.core.models import EventResult, NERResult, RelationResult
    from ai4saw.ingestion.embedder import get_vector_store
    from ai4saw.synthesis.export import export_all as _export_all

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
    ner_file: Path = typer.Option(Path("output/ner_results.json")),
    relations_file: Path = typer.Option(Path("output/relation_results.json")),
    events_file: Path = typer.Option(Path("output/event_results.json")),
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

    for p in [ner_file, relations_file, events_file]:
        if not p.exists():
            typer.echo(f"File not found: {p} — run `extract pipeline` first.", err=True)
            raise typer.Exit(1)

    output = output or Path("eval/results/judge_report.json")
    with console.status(f"LLM-as-Judge evaluating {sample} chunks..."):
        report = run_judge(ner_file, relations_file, events_file, sample, output, seed)
    display_report(report)


# ── Info ──────────────────────────────────────────────────────────────────

@app.command("info")
def info() -> None:
    """Show current configuration and status of all data artefacts."""
    from ai4saw.core.config import settings

    table = Table(title="AI4SAW Configuration", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Provider", settings.provider)
    table.add_row("Default model", settings.default_model)
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("ChromaDB", str(settings.chroma_persist_dir))
    table.add_row("Collection", settings.chroma_collection)
    table.add_row("Output dir", str(settings.output_dir))
    table.add_row("Prompts dir", str(settings.prompts_dir))
    table.add_row("Retrieval top-K", str(settings.retrieval_top_k))
    table.add_row("Re-rank top-N", str(settings.rerank_top_n))
    console.print(table)

    # Artefact status
    artefacts = {
        "ChromaDB": settings.chroma_persist_dir / "chroma.sqlite3",
        "Entity registry": Path("data/entity_registry.json"),
        "Knowledge graph": Path("data/knowledge_graph.json"),
        "NER results": Path("output/ner_results.json"),
        "Relation results": Path("output/relation_results.json"),
        "Event results": Path("output/event_results.json"),
        "Contradictions": Path("output/contradictions.json"),
        "Network": Path("output/network.json"),
    }
    status_table = Table(title="Data Artefacts", show_header=False)
    status_table.add_column("Artefact", style="cyan")
    status_table.add_column("Status")
    for name, path in artefacts.items():
        status_table.add_row(name, "[green]exists[/green]" if path.exists() else "[dim]not yet built[/dim]")
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
        get_agent_summary, load_agent_state, run_agent_session,
        save_agent_state, top_novel_entities, _seed_frontier,
        _llm_generate_seed_queries, prune_frontier,
    )
    from ai4saw.ui.dashboard import DashboardState, make_renderable

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
    agent_state = load_agent_state()
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

    # Only suppress stderr now — splash and processing screens need it for errors.
    _old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    try:
        with Live(make_renderable(dash), refresh_per_second=8, screen=True) as live:

            def _event(etype: str, msg: str) -> None:
                dash.push(etype, msg)
                dash.current_action = msg if etype == "info" else dash.current_action
                if etype == "ingest":
                    dash.docs_ingested += 1
                elif etype == "skip":
                    dash.docs_skipped += 1
                elif etype == "query":
                    dash.recent_queries.append(msg)
                    if len(dash.recent_queries) > 50:
                        dash.recent_queries = dash.recent_queries[-50:]
                live.update(make_renderable(dash))

            def _on_reasoning(r: object) -> None:
                dash.set_reasoning(
                    doc=getattr(r, "source_title", ""),
                    entities=getattr(r, "novel_entities", []),
                    queries=getattr(r, "generated_queries", []),
                    why=getattr(r, "reasoning", ""),
                )
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

            # Generate opening summary immediately on start
            _narrate()

            import threading as _threading
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

                frontier_low = len(agent_state.frontier) < 5
                _drifting = bool(dash.drift_warning)
                needs_seed = frontier_low or (agent_state.session_count % seed_every == 0) or _drifting

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
                    _seed_done = _threading.Event()
                    _seed_error: list[Exception] = []

                    def _run_seed() -> None:
                        try:
                            from ai4saw.agents.agent_discover import (
                                _execute_novel_query, _frontier_priority, _add_to_frontier,
                            )
                            # Use LLM queries as the search terms for ALL sources:
                            # DDG, Wikipedia, CrossRef, OpenAlex, Internet Archive, GDELT.
                            llm_qs = list(agent_state.query_queue)
                            agent_state.query_queue = []
                            if llm_qs:
                                _seed_frontier(
                                    llm_qs,          # LLM queries as search terms, not entity names
                                    agent_state,
                                    contact_email=contact_email,
                                    on_event=_event,
                                    use_api_sources=_use_apis,
                                    use_ddg=True,
                                )
                        except Exception as _e:
                            _seed_error.append(_e)
                            import traceback as _tb
                            _error_log.write_text(_tb.format_exc(), encoding="utf-8")
                        finally:
                            _seed_done.set()

                    _t = _threading.Thread(target=_run_seed, daemon=True)
                    _t.start()

                    # Keep UI alive while seeding runs in background
                    _spin_i = 0
                    while not _seed_done.wait(timeout=0.12):
                        dash.current_action = f"{_spinners[_spin_i % len(_spinners)]} Seeding frontier…"
                        live.update(make_renderable(dash))
                        _spin_i += 1

                    if _seed_error:
                        dash.push("error", f"Seed failed: {str(_seed_error[0])[:60]}")
                    dash.frontier_size = len(agent_state.frontier)
                    dash.push("info", f"Frontier seeded: {len(agent_state.frontier)} URLs")
                    live.update(make_renderable(dash))

                _sess_done = _threading.Event()
                _sess_result: list = []   # [docs_in, chunks_in, reasonings] on success
                _sess_error: list[Exception] = []

                def _run_session() -> None:
                    try:
                        result = run_agent_session(
                            state=agent_state,
                            geography=geography,
                            frontier_batch=frontier_batch,
                            max_reasoning=max_reasoning,
                            on_event=_event,
                            on_reasoning=_on_reasoning,
                        )
                        _sess_result.append(result)
                    except Exception as _e:
                        _sess_error.append(_e)
                        import traceback as _tb
                        _error_log.write_text(_tb.format_exc(), encoding="utf-8")
                    finally:
                        _sess_done.set()

                dash.current_action = "Draining frontier…"
                live.update(make_renderable(dash))
                _threading.Thread(target=_run_session, daemon=True).start()

                _spin_i = 0
                while not _sess_done.wait(timeout=0.12):
                    dash.current_action = f"{_spinners[_spin_i % len(_spinners)]} Draining frontier…"
                    live.update(make_renderable(dash))
                    _spin_i += 1

                if _sess_error:
                    dash.push("error", f"Session error (see output/research_errors.log): {str(_sess_error[0])[:60]}")
                    dash.current_action = f"Session {dash.session} failed — retrying next cycle"
                    live.update(make_renderable(dash))
                    time.sleep(5)
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
                    save_agent_state(agent_state)

                    try:
                        _narrate()
                    except Exception:
                        pass

                if interval > 0:
                    dash.current_action = f"Sleeping {interval}s…"
                    live.update(make_renderable(dash))
                    time.sleep(interval)

    finally:
        sys.stderr.close()
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


if __name__ == "__main__":
    app()
