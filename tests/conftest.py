"""Shared pytest fixtures for the AI4SAW test suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


# ── Document fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_doc() -> Document:
    return Document(
        page_content=(
            "The Armed Group Alpha (Armed-Group-Alpha) under Commander Alpha carried out "
            "systematic attacks on the civilian population of Location Alpha in a specific date. "
            "Peacekeeping Unit (Peacekeeping Unit) troops were present but unable to prevent the massacre."
        ),
        metadata={
            "source_filename": "srebrenica_report.pdf",
            "source_url": "https://example.com/srebrenica.pdf",
            "doc_type": "report",
            "language": "en",
            "geography": "conflict-region",
            "chunk_index": 0,
        },
    )


@pytest.fixture
def sample_docs(sample_doc) -> list[Document]:
    docs = []
    for i in range(3):
        doc = Document(
            page_content=f"Paragraph {i}: " + "Evidence text " * 100,
            metadata={**sample_doc.metadata, "chunk_index": i},
        )
        docs.append(doc)
    return docs


# ── LLM mock ──────────────────────────────────────────────────────────────────

def make_mock_llm(content: str) -> MagicMock:
    """Return a mock LLM whose .invoke() returns a response with given content."""
    mock_response = MagicMock()
    mock_response.content = content
    llm = MagicMock()
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def mock_llm_ner():
    return make_mock_llm(json.dumps({
        "entities": [
            {"text": "Commander Alpha", "label": "PERSON", "confidence": 0.95},
            {"text": "Armed-Group-Alpha", "label": "ORG", "confidence": 0.90},
            {"text": "Location Alpha", "label": "LOCATION", "confidence": 0.98},
        ]
    }))


@pytest.fixture
def mock_llm_reasoning():
    return make_mock_llm(json.dumps({
        "novel_entities": ["Peacekeeping Unit", "Witness Alpha"],
        "queries": [
            "Peacekeeping Unit UNPROFOR Location Alpha report 1995",
            "Witness Alpha plea agreement International Tribunal 1996",
        ],
        "reasoning": "Witness Alpha is primary eyewitness; Peacekeeping Unit report is primary source.",
    }))


# ── File system fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sources_csv(tmp_path) -> Path:
    """A populated sources.csv in a temp directory."""
    path = tmp_path / "sources.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "source_url", "date_accessed", "licence", "geography", "notes"],
        )
        writer.writeheader()
        writer.writerow({
            "filename": "report.pdf",
            "source_url": "https://hrw.org/report1.pdf",
            "date_accessed": "2026-06-01",
            "licence": "open-access",
            "geography": "conflict-region",
            "notes": "test",
        })
        writer.writerow({
            "filename": "news.html",
            "source_url": "https://gdelt.org/article1",
            "date_accessed": "2026-06-01",
            "licence": "news",
            "geography": "conflict-region",
            "notes": "test",
        })
    return path


@pytest.fixture
def empty_sources_csv(tmp_path) -> Path:
    path = tmp_path / "sources.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "source_url", "date_accessed", "licence", "geography", "notes"],
        )
        writer.writeheader()
    return path


# ── Discovery fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_discovered_doc():
    from ai4saw.core.models import DiscoveredDocument
    return DiscoveredDocument(
        title="Location Alpha: A Safe Area",
        url="https://archive.org/details/srebrenica-report",
        source="internetarchive",
        date="1995-11-01",
        relevance_score=0.80,
        trigger_entity="Location Alpha",
        snippet="International Tribunal investigation into the fall of Location Alpha.",
    )


# ── Web agent state fixture ────────────────────────────────────────────────────

@pytest.fixture
def fresh_web_state():
    from ai4saw.agents.web_agent import WebAgentState
    return WebAgentState()


@pytest.fixture
def fresh_agent_state():
    from ai4saw.agents.agent_discover import AgentDiscoverState
    return AgentDiscoverState(initial_entities=["Location Alpha", "Commander Alpha"])
