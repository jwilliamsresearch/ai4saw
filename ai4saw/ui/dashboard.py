"""Rich Live terminal dashboard for the research session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


MAX_FEED_LINES = 14
BAR_WIDTH = 16


@dataclass
class DashboardState:
    query: str = ""
    geography: str = ""
    entities: list[str] = field(default_factory=list)

    session: int = 0
    docs_ingested: int = 0
    docs_skipped: int = 0
    docs_errored: int = 0
    chunks_added: int = 0
    frontier_size: int = 0
    novel_entities: int = 0
    queries_queued: int = 0
    queries_executed: int = 0

    start_time: datetime = field(default_factory=datetime.now)
    current_action: str = "Initialising…"

    feed: list[tuple[str, str]] = field(default_factory=list)

    last_doc: str = ""
    last_entities: list[str] = field(default_factory=list)
    last_queries: list[str] = field(default_factory=list)
    last_why: str = ""
    reasoning_count: int = 0

    top_entities: list[tuple[str, int]] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    session_history: list[tuple[int, int, int]] = field(default_factory=list)

    narrator_text: str = ""
    narrator_updated_at: str = ""

    # Recent LLM-generated queries executed
    recent_queries: list[str] = field(default_factory=list)

    # Drift warning (set when consecutive skip rate is too high)
    drift_warning: str = ""

    def push(self, event_type: str, message: str) -> None:
        self.feed.append((event_type, message))
        if len(self.feed) > MAX_FEED_LINES:
            self.feed = self.feed[-MAX_FEED_LINES:]

    def set_reasoning(self, doc: str, entities: list[str], queries: list[str], why: str) -> None:
        self.last_doc = doc
        self.last_entities = entities
        self.last_queries = queries
        self.last_why = why
        self.reasoning_count += 1

    def record_source(self, source: str) -> None:
        self.source_counts[source] = self.source_counts.get(source, 0) + 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed(start: datetime) -> str:
    secs = int((datetime.now() - start).total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(value: int, max_value: int, width: int = BAR_WIDTH) -> str:
    if max_value == 0:
        return "░" * width
    filled = int((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


# ── Panel renderers ───────────────────────────────────────────────────────────

_HEADER_LOGO = (
    "┏━┓╻╻ ╻┏━┓┏━┓╻ ╻\n"
    "┣━┫┃┗━┫┗━┓┣━┫┃╻┃\n"
    "╹ ╹╹  ╹┗━┛╹ ╹┗┻┛"
)


def _header(state: DashboardState) -> Panel:
    total = state.docs_ingested + state.docs_skipped
    pct = f"{int(state.docs_ingested / total * 100)}% relevant" if total else "—"
    entity_str = ", ".join(state.entities[:5]) or "—"

    info = Text()
    info.append(f"{state.query}\n", style="bold white")
    info.append("Entities: ", style="dim")
    info.append(f"{entity_str}   ", style="cyan")
    info.append("Geography: ", style="dim")
    info.append(f"{state.geography}   ", style="white")
    info.append("Session: ", style="dim")
    info.append(f"{state.session}   ", style="white")
    info.append("Runtime: ", style="dim")
    info.append(f"{_elapsed(state.start_time)}   ", style="white")
    info.append("Relevance: ", style="dim")
    info.append(f"{pct}\n", style="white")
    info.append("▸ ", style="dim")
    info.append(state.current_action, style="dim")

    import time as _time
    from rich.align import Align as _Align
    _GRAD = [
        "#f64f59", "#ee536e", "#e75883", "#e05d98", "#d962ad",
        "#d267c2", "#cb6cd7", "#c471ed", "#aa7cec", "#9188eb",
        "#7793eb", "#5e9fea", "#44aaea", "#2bb6e9", "#12c2e9",
    ]
    _RAINBOW = _GRAD + _GRAD[-2:0:-1]  # 15 forward + 13 reversed = 28 total, loops seamlessly
    _offset = int(_time.time() * 6)

    logo = Text(no_wrap=True)
    for line in _HEADER_LOGO.splitlines():
        for i, ch in enumerate(line):
            logo.append(ch, style=_RAINBOW[(i + _offset) % len(_RAINBOW)])
        logo.append("\n")

    row = Table.grid(expand=True, padding=0)
    row.add_column(ratio=1)
    row.add_column(width=26)
    row.add_row(info, _Align(logo, align="right", vertical="top"))

    return Panel(
        row,
        title="[bold blue]AI4SAW  ·  Research Mode[/bold blue]",
        subtitle="[dim]James Williams[/dim]",
        border_style="blue",
        padding=(0, 1),
    )


def _narrator(state: DashboardState) -> Panel:
    if state.drift_warning:
        body = f"[bold red]⚠ {state.drift_warning}[/bold red]"
        if state.narrator_text:
            body += f"\n[dim]─────[/dim]\n[white]{state.narrator_text}[/white]"
    elif state.narrator_text:
        ts = f"  [dim]— {state.narrator_updated_at}[/dim]" if state.narrator_updated_at else ""
        body = f"[white]{state.narrator_text}[/white]{ts}"
    else:
        body = "[dim]Research summary will appear here every few sessions…[/dim]"

    try:
        from ai4saw.core.config import settings as _s
        model_label = f"[dim]model: {_s.default_model}[/dim]"
    except Exception:
        model_label = ""

    return Panel(
        body,
        title="[bold yellow]Research Summary[/bold yellow]",
        subtitle=model_label,
        border_style="yellow",
        padding=(0, 1),
    )


def _feed(state: DashboardState) -> Panel:
    ICONS = {
        "ingest": ("[green]✓[/green]", "green"),
        "skip":   ("[red]✗[/red]",    "dim"),
        "reason": ("[cyan]⟳[/cyan]",  "cyan"),
        "query":  ("[yellow]✦[/yellow]", "yellow"),
        "error":  ("[red]![/red]",    "red"),
        "info":   ("[dim]·[/dim]",    "dim"),
    }
    lines: list[str] = []
    for etype, msg in state.feed:
        icon, colour = ICONS.get(etype, ("[dim]·[/dim]", "dim"))
        lines.append(f"{icon} [{colour}]{msg}[/{colour}]")

    # Pad to MAX_FEED_LINES so panel height stays fixed
    while len(lines) < MAX_FEED_LINES:
        lines.append("")

    return Panel(
        "\n".join(lines),
        title=f"[bold]Live Feed[/bold] [dim]({state.docs_ingested + state.docs_skipped} processed)[/dim]",
        border_style="green",
        padding=(0, 1),
    )


def _reasoning(state: DashboardState) -> Panel:
    if state.last_doc:
        e_lines = "\n".join(f"  [cyan]•[/cyan] {e}" for e in state.last_entities[:5]) or "  [dim]none[/dim]"
        q_lines = "\n".join(f"  [yellow]→[/yellow] {q}" for q in state.last_queries[:3]) or "  [dim]none[/dim]"
        body = (
            f"[bold]{state.last_doc[:55]}[/bold]\n\n"
            f"[dim]Novel entities:[/dim]\n{e_lines}\n\n"
            f"[dim]Generated queries:[/dim]\n{q_lines}"
            + (f"\n\n[dim italic]{state.last_why[:120]}[/dim italic]" if state.last_why else "")
        )
    else:
        body = "\n" * 3 + "  [dim]Waiting for first reasoning cycle…[/dim]"

    return Panel(
        body,
        title=f"[bold]LLM Reasoning[/bold] [dim](×{state.reasoning_count})[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )


def _entity_chart(state: DashboardState) -> Panel:
    if not state.top_entities:
        body = "\n" * 3 + "  [dim]No entities discovered yet[/dim]"
    else:
        max_c = max(n for _, n in state.top_entities) or 1
        lines = []
        for entity, count in state.top_entities[:10]:
            bar = _bar(count, max_c)
            label = entity[:16].ljust(16)
            lines.append(f"[cyan]{label}[/cyan] [magenta]{bar}[/magenta] [dim]{count}[/dim]")
        # Pad so height is consistent
        while len(lines) < 10:
            lines.append("")
        body = "\n".join(lines)

    return Panel(body, title="[bold]Entity Network[/bold]", border_style="magenta", padding=(0, 1))


def _sources(state: DashboardState) -> Panel:
    lines: list[str] = []

    if state.source_counts:
        max_c = max(state.source_counts.values()) or 1
        for src, count in sorted(state.source_counts.items(), key=lambda x: -x[1])[:7]:
            bar = _bar(count, max_c, 10)
            label = src[:14].ljust(14)
            lines.append(f"[dim]{label}[/dim] [blue]{bar}[/blue] [white]{count}[/white]")
    else:
        lines.append("  [dim]Accumulating…[/dim]")

    # Session sparkline
    if state.session_history:
        heights = "▁▂▃▄▅▆▇█"
        max_in = max((d for _, d, _ in state.session_history), default=1) or 1
        spark = "".join(heights[min(7, int((d / max_in) * 7))] for _, d, _ in state.session_history[-24:])
        lines.append("")
        lines.append(f"[dim]Sessions[/dim] [white]{spark}[/white]")

    # Pad
    while len(lines) < 10:
        lines.append("")

    return Panel("\n".join(lines), title="[bold]Sources[/bold]", border_style="blue", padding=(0, 1))


def _query_pipeline(state: DashboardState) -> Panel:
    """Shows the last N LLM-generated queries executed."""
    lines: list[str] = []
    queries = state.recent_queries[-12:] if state.recent_queries else []
    for q in queries:
        lines.append(f"[yellow]→[/yellow] [dim]{q[:52]}[/dim]")
    while len(lines) < 12:
        lines.append("")
    stats_line = (
        f"[cyan]⟳ {state.queries_queued} queued[/cyan]   "
        f"[white]✉ {state.queries_executed} executed[/white]"
    )
    lines.append(stats_line)
    return Panel(
        "\n".join(lines),
        title="[bold]Query Pipeline[/bold]",
        border_style="yellow",
        padding=(0, 1),
    )


def _relevance_trend(state: DashboardState) -> Panel:
    """Per-session ingested vs skipped bar chart."""
    lines: list[str] = []
    if not state.session_history:
        lines = ["", "", "  [dim]No sessions yet[/dim]"]
    else:
        for sess, ingested, skipped in state.session_history[-10:]:
            total = ingested + skipped or 1
            pct = int((ingested / total) * 12)
            bar_in  = f"[green]{'█' * pct}[/green]"
            bar_out = f"[red]{'░' * (12 - pct)}[/red]"
            lines.append(
                f"[dim]S{sess:<2}[/dim] {bar_in}{bar_out} "
                f"[green]{ingested}✓[/green] [red]{skipped}✗[/red]"
            )
    while len(lines) < 12:
        lines.append("")
    total_all = state.docs_ingested + state.docs_skipped or 1
    pct_all = int(state.docs_ingested / total_all * 100)
    lines.append(f"[dim]Overall relevance:[/dim] [bold white]{pct_all}%[/bold white]")
    return Panel(
        "\n".join(lines),
        title="[bold]Relevance Trend[/bold]",
        border_style="green",
        padding=(0, 1),
    )


def _stats(state: DashboardState) -> Panel:
    return Panel(
        f"[green]✓ {state.docs_ingested} ingested[/green]   "
        f"[red]✗ {state.docs_skipped} skipped[/red]   "
        f"[yellow]⬡ {state.chunks_added} chunks[/yellow]   "
        f"[blue]⋯ {state.frontier_size} frontier[/blue]   "
        f"[magenta]✦ {state.novel_entities} entities[/magenta]   "
        f"[cyan]⟳ {state.queries_queued} queued[/cyan]   "
        f"[white]✉ {state.queries_executed} executed[/white]",
        border_style="dim",
        padding=(0, 0),
    )


# ── Layout builder ────────────────────────────────────────────────────────────

def make_renderable(state: DashboardState) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header",   size=6),
        Layout(name="narrator", size=6),
        Layout(name="middle",   ratio=1),
        Layout(name="bottom",   ratio=1),
        Layout(name="stats",    size=3),
    )

    layout["middle"].split_row(
        Layout(name="feed",      ratio=1),
        Layout(name="reasoning", ratio=1),
    )

    layout["bottom"].split_row(
        Layout(name="chart",    ratio=1),
        Layout(name="sources",  ratio=1),
        Layout(name="queries",  ratio=1),
        Layout(name="relevance",ratio=1),
    )

    layout["header"].update(_header(state))
    layout["narrator"].update(_narrator(state))
    layout["feed"].update(_feed(state))
    layout["reasoning"].update(_reasoning(state))
    layout["chart"].update(_entity_chart(state))
    layout["sources"].update(_sources(state))
    layout["queries"].update(_query_pipeline(state))
    layout["relevance"].update(_relevance_trend(state))
    layout["stats"].update(_stats(state))

    return layout
