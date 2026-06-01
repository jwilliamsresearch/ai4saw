"""Tests for the agentic discovery loop.

The LLM reasoning step is the critical path — if it fails silently,
the corpus stops expanding. These tests verify:
- Reasoning parses correctly from LLM output (including markdown fences)
- Invalid JSON from LLM returns None (graceful degradation)
- Novel entities accumulate in state across sessions
- Query queue grows and drains correctly
- State persistence round-trips
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4saw.agents.agent_discover import (
    AgentDiscoverState,
    DiscoveryReasoning,
    _llm_reason,
    _log_reasoning,
    get_agent_summary,
    load_agent_state,
    save_agent_state,
)
from ai4saw.agents.web_agent import FrontierItem
from ai4saw.core.models import DiscoveredDocument


def _make_doc(url: str = "https://icty.org/judgment.pdf", entity: str = "Srebrenica") -> DiscoveredDocument:
    return DiscoveredDocument(
        title="ICTY Krstić Trial Judgment",
        url=url,
        source="duckduckgo",
        date="2001-08-02",
        relevance_score=0.85,
        trigger_entity=entity,
    )


# ── DiscoveryReasoning model ───────────────────────────────────────────────────

class TestDiscoveryReasoning:
    def test_creation(self):
        r = DiscoveryReasoning(
            source_url="https://icty.org/krstic.pdf",
            source_title="Krstić Trial Judgment",
            trigger_entity="Srebrenica",
            novel_entities=["Dražen Erdemović", "10th Sabotage Detachment"],
            generated_queries=["Erdemović plea ICTY 1996", "10th Sabotage Detachment Branjevo"],
            reasoning="Erdemović is the primary execution witness.",
        )
        assert len(r.novel_entities) == 2
        assert len(r.generated_queries) == 2
        assert r.timestamp  # auto-set

    def test_serialisable_to_json(self):
        r = DiscoveryReasoning(
            source_url="https://x.com",
            source_title="Test",
            trigger_entity="Test",
            novel_entities=["Entity A"],
            generated_queries=["query one"],
            reasoning="test",
        )
        serialised = r.model_dump_json()
        data = json.loads(serialised)
        assert data["novel_entities"] == ["Entity A"]


# ── LLM reasoning ─────────────────────────────────────────────────────────────

class TestLlmReason:
    def _valid_payload(self) -> str:
        return json.dumps({
            "novel_entities": ["Dražen Erdemović", "10th Sabotage Detachment"],
            "queries": [
                "Dražen Erdemović plea agreement ICTY 1996",
                "10th Sabotage Detachment Branjevo Farm executions",
            ],
            "reasoning": "Erdemović is the primary eyewitness to the Branjevo executions.",
        })

    def test_parses_clean_json(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=self._valid_payload())

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason(
                text="Mladic ordered executions. Erdemović carried them out.",
                doc=_make_doc(),
                initial_entities=["Srebrenica"],
                geography="Bosnia",
                known_entities=[],
            )

        assert result is not None
        assert "Dražen Erdemović" in result.novel_entities
        assert len(result.generated_queries) == 2
        assert result.reasoning

    def test_parses_json_in_markdown_fence(self):
        fenced = "```json\n" + self._valid_payload() + "\n```"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=fenced)

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("text", _make_doc(), ["Srebrenica"], "Bosnia", [])

        assert result is not None
        assert len(result.novel_entities) == 2

    def test_parses_json_in_backtick_fence(self):
        fenced = "```\n" + self._valid_payload() + "\n```"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=fenced)

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("text", _make_doc(), ["Srebrenica"], "Bosnia", [])

        assert result is not None

    def test_invalid_json_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Not JSON at all, sorry.")

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("text", _make_doc(), ["Srebrenica"], "Bosnia", [])

        assert result is None

    def test_empty_text_returns_none(self):
        mock_llm = MagicMock()

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("", _make_doc(), ["Srebrenica"], "Bosnia", [])

        assert result is None
        mock_llm.invoke.assert_not_called()

    def test_llm_exception_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("some text", _make_doc(), ["Srebrenica"], "Bosnia", [])

        assert result is None

    def test_novel_entities_capped_at_ten(self):
        payload = json.dumps({
            "novel_entities": [f"Entity_{i}" for i in range(20)],
            "queries": ["q1"],
            "reasoning": "many entities",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=payload)

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("text", _make_doc(), ["X"], "Bosnia", [])

        assert result is not None
        assert len(result.novel_entities) <= 10

    def test_queries_capped_at_three(self):
        payload = json.dumps({
            "novel_entities": [],
            "queries": ["q1", "q2", "q3", "q4", "q5"],
            "reasoning": "many queries",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=payload)

        with patch("ai4saw.core.providers.get_llm", return_value=mock_llm):
            result = _llm_reason("text", _make_doc(), ["X"], "Bosnia", [])

        assert result is not None
        assert len(result.generated_queries) <= 3


# ── State management ───────────────────────────────────────────────────────────

class TestAgentDiscoverState:
    def test_initial_entities_stored(self, fresh_agent_state):
        assert "Srebrenica" in fresh_agent_state.initial_entities
        assert "Mladic" in fresh_agent_state.initial_entities

    def test_query_queue_drains_correctly(self, fresh_agent_state):
        fresh_agent_state.query_queue = ["q1", "q2", "q3", "q4", "q5", "q6"]
        batch = fresh_agent_state.query_queue[:5]
        fresh_agent_state.query_queue = fresh_agent_state.query_queue[5:]
        assert len(batch) == 5
        assert fresh_agent_state.query_queue == ["q6"]

    def test_novel_entities_accumulated(self, fresh_agent_state):
        fresh_agent_state.discovered_entities["Dražen Erdemović"] = {
            "from_url": "https://icty.org/krstic",
            "timestamp": "2026-06-01T12:00:00Z",
        }
        assert "Dražen Erdemović" in fresh_agent_state.discovered_entities

    def test_executed_queries_tracked(self, fresh_agent_state):
        fresh_agent_state.executed_queries.append("Erdemović plea 1996")
        assert "Erdemović plea 1996" in fresh_agent_state.executed_queries


# ── State persistence ──────────────────────────────────────────────────────────

class TestAgentStatePersistence:
    def test_load_missing_file_returns_fresh(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        monkeypatch.setattr(agent_discover, "AGENT_STATE_FILE", tmp_path / "missing.json")
        state = load_agent_state()
        assert isinstance(state, AgentDiscoverState)

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        path = tmp_path / "agent_state.json"
        monkeypatch.setattr(agent_discover, "AGENT_STATE_FILE", path)

        state = AgentDiscoverState(initial_entities=["Srebrenica"])
        state.discovered_entities["Erdemović"] = {"from_url": "https://icty.org", "timestamp": "2026-06-01"}
        state.query_queue = ["query 1", "query 2"]
        state.session_count = 7

        save_agent_state(state)
        loaded = load_agent_state()

        assert loaded.session_count == 7
        assert "Erdemović" in loaded.discovered_entities
        assert loaded.query_queue == ["query 1", "query 2"]

    def test_corrupt_file_returns_fresh(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        path = tmp_path / "corrupt.json"
        path.write_text("!!!not json!!!", encoding="utf-8")
        monkeypatch.setattr(agent_discover, "AGENT_STATE_FILE", path)

        state = load_agent_state()
        assert state.session_count == 0


# ── Reasoning log ──────────────────────────────────────────────────────────────

class TestReasoningLog:
    def test_log_appends_jsonl(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        log_path = tmp_path / "log.jsonl"
        monkeypatch.setattr(agent_discover, "AGENT_LOG_FILE", log_path)

        r = DiscoveryReasoning(
            source_url="https://x.com",
            source_title="Test Doc",
            trigger_entity="Srebrenica",
            novel_entities=["Entity A"],
            generated_queries=["query 1"],
            reasoning="test reasoning",
        )
        _log_reasoning(r)
        _log_reasoning(r)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["source_url"] == "https://x.com"
        assert entry["novel_entities"] == ["Entity A"]


# ── get_agent_summary ──────────────────────────────────────────────────────────

class TestGetAgentSummary:
    def test_summary_fields_present(self, fresh_agent_state):
        fresh_agent_state.session_count = 3
        fresh_agent_state.total_docs_ingested = 42
        fresh_agent_state.discovered_entities = {"Entity A": {}, "Entity B": {}}

        summary = get_agent_summary(fresh_agent_state)
        assert summary["sessions"] == 3
        assert summary["docs_ingested"] == 42
        assert summary["novel_entities"] == 2
        assert "top_novel_entities" in summary

    def test_top_novel_entities_capped(self, fresh_agent_state):
        for i in range(20):
            fresh_agent_state.discovered_entities[f"Entity_{i}"] = {}

        summary = get_agent_summary(fresh_agent_state)
        assert len(summary["top_novel_entities"]) <= 10
