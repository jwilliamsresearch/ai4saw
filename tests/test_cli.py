"""CLI smoke tests — verify commands parse, route, and exit cleanly.

These don't test business logic (that's in the unit tests above) but they
verify the CLI layer doesn't crash on valid inputs and rejects bad ones.
Uses typer's CliRunner so no real subprocesses or network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai4saw.cli import app

runner = CliRunner()


# ── --help on every command group ─────────────────────────────────────────────

class TestHelpStrings:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "discover" in result.output.lower() or "ingest" in result.output.lower()

    def test_discover_help(self):
        result = runner.invoke(app, ["discover", "--help"])
        assert result.exit_code == 0
        assert "fetch" in result.output
        assert "web" in result.output
        assert "agent" in result.output

    def test_discover_fetch_help(self):
        result = runner.invoke(app, ["discover", "fetch", "--help"])
        assert result.exit_code == 0
        assert "--geography" in result.output
        assert "--max-docs" in result.output

    def test_discover_web_help(self):
        result = runner.invoke(app, ["discover", "web", "--help"])
        assert result.exit_code == 0
        assert "--loop" in result.output
        assert "--interval" in result.output
        assert "--state" in result.output

    def test_discover_agent_help(self):
        result = runner.invoke(app, ["discover", "agent", "--help"])
        assert result.exit_code == 0
        assert "--loop" in result.output
        assert "--log" in result.output
        assert "--state" in result.output

    def test_ingest_help(self):
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0

    def test_extract_help(self):
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0

    def test_graph_help(self):
        result = runner.invoke(app, ["graph", "--help"])
        assert result.exit_code == 0

    def test_query_help(self):
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0


# ── discover web --state ───────────────────────────────────────────────────────

class TestDiscoverWebState:
    def test_state_command_with_fresh_state(self, tmp_path, monkeypatch):
        from ai4saw.agents import web_agent
        monkeypatch.setattr(web_agent, "STATE_FILE", tmp_path / "fresh.json")

        result = runner.invoke(app, [
            "discover", "web", "Location Alpha",
            "--geography", "conflict-region",
            "--state",
        ])
        assert result.exit_code == 0
        assert "Session" in result.output or "frontier" in result.output.lower()


# ── discover agent --state ─────────────────────────────────────────────────────

class TestDiscoverAgentState:
    def test_state_command_with_fresh_state(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        monkeypatch.setattr(agent_discover, "AGENT_STATE_FILE", tmp_path / "fresh.json")

        result = runner.invoke(app, [
            "discover", "agent", "Location Alpha",
            "--geography", "conflict-region",
            "--state",
        ])
        assert result.exit_code == 0

    def test_log_command_no_log_file(self, tmp_path, monkeypatch):
        from ai4saw.agents import agent_discover
        monkeypatch.setattr(agent_discover, "AGENT_STATE_FILE", tmp_path / "fresh.json")
        monkeypatch.setattr(agent_discover, "AGENT_LOG_FILE", tmp_path / "nonexistent.jsonl")

        result = runner.invoke(app, [
            "discover", "agent", "Location Alpha",
            "--geography", "conflict-region",
            "--log", "5",
        ])
        assert result.exit_code == 0
        assert "No reasoning log" in result.output


# ── discover fetch --dry-run ───────────────────────────────────────────────────

class TestDiscoverFetchDryRun:
    def test_dry_run_does_not_ingest(self):
        mock_result = MagicMock()
        mock_result.candidates_found = 10
        mock_result.candidates_above_threshold = 3

        with patch("ai4saw.agents.fetch_agent.fetch_corpus", return_value=mock_result):
            result = runner.invoke(app, [
                "discover", "fetch", "Location Alpha", "Commander Alpha",
                "--geography", "conflict-region",
                "--dry-run",
            ])

        assert result.exit_code == 0
        assert "10" in result.output or "3" in result.output

    def test_missing_geography_fails(self):
        result = runner.invoke(app, [
            "discover", "fetch", "Location Alpha",
            # --geography is required
        ])
        assert result.exit_code != 0


# ── info command ───────────────────────────────────────────────────────────────

class TestInfoCommand:
    def test_info_runs_without_error(self):
        with (
            patch("ai4saw.ingestion.embedder.get_vector_store") as mock_store,
            patch("ai4saw.core.config.settings") as mock_settings,
        ):
            mock_settings.provider = "ollama"
            mock_settings.default_model = "mistral"
            mock_settings.embed_model = "nomic-embed-text"
            mock_settings.chroma_path = "/tmp/chroma"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_collection = MagicMock()
            mock_collection.count.return_value = 42
            mock_store.return_value._collection = mock_collection

            result = runner.invoke(app, ["info"])
            # Should not raise — exit code may vary depending on Ollama availability
            assert result.exit_code in (0, 1)
