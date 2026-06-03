"""Project management — per-research-query namespacing of corpus, state, and graph.

Each project isolates:
  - ChromaDB collection  (ai4saw_{slug})
  - Agent state file     (data/projects/{slug}/agent_state.json)
  - sources.csv          (data/projects/{slug}/sources.csv)
  - corpus/ directory    (data/projects/{slug}/corpus/)
  - search_graph.json    (data/projects/{slug}/search_graph.json)
  - output/              (data/projects/{slug}/output/)

The active project is tracked in data/active_project (plain text slug).
When no project is active, the system uses legacy defaults for backward compat.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

PROJECTS_ROOT = Path("data/projects")
ACTIVE_PROJECT_FILE = Path("data/active_project")


def _slug(name: str) -> str:
    """Convert a project name to a filesystem-safe slug (max 48 chars)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48]


class ProjectMeta(BaseModel):
    slug: str
    name: str
    research_query: str
    geography: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    description: str = ""


def project_dir(slug: str) -> Path:
    return PROJECTS_ROOT / slug


def create_project(
    name: str, research_query: str, geography: str = ""
) -> ProjectMeta:
    """Create a new project directory structure and write project.json."""
    slug = _slug(name)
    pdir = project_dir(slug)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "corpus").mkdir(exist_ok=True)
    (pdir / "output").mkdir(exist_ok=True)
    (pdir / "data").mkdir(exist_ok=True)

    meta = ProjectMeta(
        slug=slug,
        name=name,
        research_query=research_query,
        geography=geography,
    )
    _meta_path(slug).write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return meta


def _meta_path(slug: str) -> Path:
    return project_dir(slug) / "project.json"


def load_project(slug: str) -> ProjectMeta:
    p = _meta_path(slug)
    if not p.exists():
        raise FileNotFoundError(f"Project not found: {slug!r}")
    return ProjectMeta.model_validate_json(p.read_text(encoding="utf-8"))


def list_projects() -> list[ProjectMeta]:
    if not PROJECTS_ROOT.exists():
        return []
    projects = []
    for d in sorted(PROJECTS_ROOT.iterdir()):
        pf = d / "project.json"
        if pf.exists():
            try:
                projects.append(
                    ProjectMeta.model_validate_json(pf.read_text(encoding="utf-8"))
                )
            except Exception:
                pass
    return projects


def set_active_project(slug: str) -> None:
    ACTIVE_PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROJECT_FILE.write_text(slug, encoding="utf-8")


def get_active_project() -> Optional[str]:
    if ACTIVE_PROJECT_FILE.exists():
        slug = ACTIVE_PROJECT_FILE.read_text(encoding="utf-8").strip()
        return slug or None
    return None


def clear_active_project() -> None:
    if ACTIVE_PROJECT_FILE.exists():
        ACTIVE_PROJECT_FILE.unlink()


def get_project_paths(slug: str) -> dict[str, Path]:
    """Return all filesystem paths for a project."""
    pdir = project_dir(slug)
    return {
        "dir": pdir,
        "corpus": pdir / "corpus",
        "sources_csv": pdir / "sources.csv",
        "agent_state": pdir / "agent_state.json",
        "agent_log": pdir / "agent_log.jsonl",
        "search_graph": pdir / "search_graph.json",
        "output": pdir / "output",
        "data": pdir / "data",
        "chroma_collection": f"ai4saw_{slug}",
    }


def resolve_project(slug: Optional[str] = None) -> Optional[dict]:
    """Return project paths dict for the given slug, active project, or None (legacy mode).

    Call this at the start of any command that should be project-aware.
    Returns None when no project is active (backward-compat fallback).
    """
    effective = slug or get_active_project()
    if not effective:
        return None
    try:
        load_project(effective)  # validates it exists
    except FileNotFoundError:
        return None
    return get_project_paths(effective)


# ── Active-paths singleton (set at runtime, read by fetch_agent etc.) ──────────
# Any module that needs to know the current corpus dir / sources.csv reads here.
# When no project is active, returns legacy defaults for backward compat.

_active_paths: Optional[dict] = None


def set_active_paths(paths: Optional[dict]) -> None:
    global _active_paths
    _active_paths = paths


def get_corpus_dir() -> Path:
    if _active_paths and "corpus" in _active_paths:
        return _active_paths["corpus"]
    return Path("corpus")


def get_sources_csv() -> Path:
    if _active_paths and "sources_csv" in _active_paths:
        return _active_paths["sources_csv"]
    return Path("corpus/sources.csv")


def get_output_dir() -> Path:
    """Return the active project's output dir, or legacy output/."""
    if _active_paths and "output" in _active_paths:
        return _active_paths["output"]
    return Path("output")


def get_data_dir() -> Path:
    """Return the active project's data dir, or legacy data/."""
    if _active_paths and "data" in _active_paths:
        return _active_paths["data"]
    return Path("data")


def load_active_project_context() -> Optional[dict]:
    """Load the active project and set all module-level context (paths + ChromaDB).

    Called automatically by the Typer app.callback() before every command.
    Safe to call multiple times — idempotent.
    Returns the project paths dict, or None if no project is active.
    """
    paths = resolve_project()
    if paths:
        set_active_paths(paths)
        try:
            from ai4saw.core.config import settings
            settings.chroma_collection = paths["chroma_collection"]
        except Exception:
            pass
        try:
            from ai4saw.core.search_graph import SearchGraph, set_active_graph
            _graph_path = paths["search_graph"]
            set_active_graph(SearchGraph(_graph_path))
        except Exception:
            pass
    return paths
