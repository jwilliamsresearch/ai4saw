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
            "The Bosnian Serb Army (VRS) under General Ratko Mladić carried out "
            "systematic attacks on the civilian population of Srebrenica in July 1995. "
            "Dutch Battalion (Dutchbat) troops were present but unable to prevent the massacre."
        ),
        metadata={
            "source_filename": "srebrenica_report.pdf",
            "source_url": "https://example.com/srebrenica.pdf",
            "doc_type": "report",
            "language": "en",
            "geography": "Bosnia",
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
            {"text": "Ratko Mladić", "label": "PERSON", "confidence": 0.95},
            {"text": "VRS", "label": "ORG", "confidence": 0.90},
            {"text": "Srebrenica", "label": "LOCATION", "confidence": 0.98},
        ]
    }))


@pytest.fixture
def mock_llm_reasoning():
    return make_mock_llm(json.dumps({
        "novel_entities": ["Dutch Battalion", "Dražen Erdemović"],
        "queries": [
            "Dutch Battalion UNPROFOR Srebrenica report 1995",
            "Dražen Erdemović plea agreement ICTY 1996",
        ],
        "reasoning": "Erdemović is primary eyewitness; Dutchbat report is primary source.",
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
            "geography": "Bosnia",
            "notes": "test",
        })
        writer.writerow({
            "filename": "news.html",
            "source_url": "https://gdelt.org/article1",
            "date_accessed": "2026-06-01",
            "licence": "news",
            "geography": "Sudan",
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
        title="Srebrenica: A Safe Area",
        url="https://archive.org/details/srebrenica-report",
        source="internetarchive",
        date="1995-11-01",
        relevance_score=0.80,
        trigger_entity="Srebrenica",
        snippet="ICTY investigation into the fall of Srebrenica.",
    )


# ── Web agent state fixture ────────────────────────────────────────────────────

@pytest.fixture
def fresh_web_state():
    from ai4saw.agents.web_agent import WebAgentState
    return WebAgentState()


@pytest.fixture
def fresh_agent_state():
    from ai4saw.agents.agent_discover import AgentDiscoverState
    return AgentDiscoverState(initial_entities=["Srebrenica", "Mladic"])
