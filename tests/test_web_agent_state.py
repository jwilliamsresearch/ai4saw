"""Tests for the web agent's persistent state management.

The frontier, domain scoring, and state persistence are the core innovations
of the stateful crawler. These tests verify the invariants the loop depends on:
- visited URLs are never re-added to the frontier
- frontier stays sorted by priority
- domain scores update correctly
- state round-trips through JSON without data loss
"""

from __future__ import annotations

import json

import pytest

from ai4saw.agents.web_agent import (
    DomainStats,
    FrontierItem,
    QueryTemplateStats,
    WebAgentState,
    _add_to_frontier,
    _domain,
    _frontier_priority,
    _is_pdf,
    _record_visit,
    _relevance,
    _sort_frontier,
    _trusted,
    load_state,
    save_state,
)


# ── Helper predicates ──────────────────────────────────────────────────────────

class TestDomainHelper:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.hrw.org/report", "hrw.org"),
        ("https://hrw.org/report", "hrw.org"),
        ("https://sub.un.org/page", "sub.un.org"),
        ("not-a-url", ""),
    ])
    def test_domain_extraction(self, url, expected):
        assert _domain(url) == expected


class TestTrustedDomain:
    @pytest.mark.parametrize("url", [
        "https://hrw.org/report.pdf",
        "https://www.amnesty.org/article",
        "https://tribunal.int/judgment",
        "https://archive.org/details/doc",
    ])
    def test_trusted_urls(self, url):
        assert _trusted(url) is True

    @pytest.mark.parametrize("url", [
        "https://twitter.com/post",
        "https://randomnews.com/article",
        "https://example.com/page",
    ])
    def test_untrusted_urls(self, url):
        assert _trusted(url) is False


class TestIsPdf:
    @pytest.mark.parametrize("url", [
        "https://example.com/report.pdf",
        "https://example.com/doc.PDF",
        "https://example.com/file.pdf?version=2",
    ])
    def test_pdf_urls(self, url):
        assert _is_pdf(url) is True

    @pytest.mark.parametrize("url", [
        "https://example.com/page.html",
        "https://example.com/report",
        "https://example.com/pdf-viewer",  # 'pdf' in path but not extension
    ])
    def test_non_pdf_urls(self, url):
        assert _is_pdf(url) is False


# ── DomainStats ────────────────────────────────────────────────────────────────

class TestDomainStats:
    def test_score_no_attempts_returns_neutral(self):
        stats = DomainStats()
        assert stats.score == pytest.approx(0.5)

    def test_score_all_hits(self):
        stats = DomainStats(hits=10, attempts=10)
        assert stats.score == pytest.approx(1.0)

    def test_score_no_hits(self):
        stats = DomainStats(hits=0, attempts=5)
        assert stats.score == pytest.approx(0.0)

    def test_score_partial(self):
        stats = DomainStats(hits=3, attempts=10)
        assert stats.score == pytest.approx(0.3)

    def test_score_rounded_to_3dp(self):
        stats = DomainStats(hits=1, attempts=3)
        assert stats.score == round(stats.score, 3)


# ── QueryTemplateStats ─────────────────────────────────────────────────────────

class TestQueryTemplateStats:
    def test_yield_rate_no_runs_is_optimistic(self):
        stats = QueryTemplateStats()
        assert stats.yield_rate == pytest.approx(1.0)

    def test_yield_rate_calculated(self):
        stats = QueryTemplateStats(runs=4, new_docs=8)
        assert stats.yield_rate == pytest.approx(2.0)

    def test_yield_rate_zero_docs(self):
        stats = QueryTemplateStats(runs=5, new_docs=0)
        assert stats.yield_rate == pytest.approx(0.0)


# ── Frontier management ────────────────────────────────────────────────────────

class TestAddToFrontier:
    def test_adds_new_url(self, fresh_web_state):
        added = _add_to_frontier(
            fresh_web_state, "https://hrw.org/new", 0.8, "Location Alpha", "duckduckgo"
        )
        assert added is True
        assert len(fresh_web_state.frontier) == 1

    def test_does_not_add_visited_url(self, fresh_web_state):
        fresh_web_state.visited_urls["https://hrw.org/visited"] = {"chunks": 5}
        added = _add_to_frontier(
            fresh_web_state, "https://hrw.org/visited", 0.8, "Location Alpha", "duckduckgo"
        )
        assert added is False
        assert len(fresh_web_state.frontier) == 0

    def test_does_not_add_duplicate_frontier_url(self, fresh_web_state):
        _add_to_frontier(fresh_web_state, "https://hrw.org/doc", 0.8, "Test", "ddg")
        added = _add_to_frontier(fresh_web_state, "https://hrw.org/doc", 0.9, "Test", "ddg")
        assert added is False
        assert len(fresh_web_state.frontier) == 1

    def test_frontier_item_fields_set_correctly(self, fresh_web_state):
        _add_to_frontier(
            fresh_web_state, "https://tribunal.int/judgment.pdf",
            0.75, "Commander Alpha", "wikipedia", depth=1
        )
        item = fresh_web_state.frontier[0]
        assert item.url == "https://tribunal.int/judgment.pdf"
        assert item.priority == pytest.approx(0.75)
        assert item.trigger_entity == "Commander Alpha"
        assert item.source == "wikipedia"
        assert item.depth == 1


class TestSortFrontier:
    def test_sorted_descending_by_priority(self, fresh_web_state):
        _add_to_frontier(fresh_web_state, "https://a.com", 0.4, "X", "ddg")
        _add_to_frontier(fresh_web_state, "https://b.com", 0.9, "X", "ddg")
        _add_to_frontier(fresh_web_state, "https://c.com", 0.6, "X", "ddg")
        _sort_frontier(fresh_web_state)
        priorities = [item.priority for item in fresh_web_state.frontier]
        assert priorities == sorted(priorities, reverse=True)

    def test_frontier_capped_at_max(self, fresh_web_state):
        from ai4saw.agents.web_agent import FRONTIER_MAX
        for i in range(FRONTIER_MAX + 100):
            fresh_web_state.frontier.append(
                FrontierItem(url=f"https://example.com/{i}", priority=0.5,
                             trigger_entity="X", source="ddg")
            )
        _sort_frontier(fresh_web_state)
        assert len(fresh_web_state.frontier) == FRONTIER_MAX


class TestFrontierPriority:
    def test_pdf_gets_boost(self, fresh_web_state):
        pdf_priority = _frontier_priority(0.5, "https://example.com/doc.pdf", fresh_web_state)
        html_priority = _frontier_priority(0.5, "https://example.com/page", fresh_web_state)
        assert pdf_priority > html_priority

    def test_trusted_domain_gets_boost(self, fresh_web_state):
        trusted = _frontier_priority(0.5, "https://hrw.org/report", fresh_web_state)
        untrusted = _frontier_priority(0.5, "https://random.com/report", fresh_web_state)
        assert trusted > untrusted

    def test_depth_reduces_priority(self, fresh_web_state):
        shallow = _frontier_priority(0.7, "https://x.com/p", fresh_web_state, depth=0)
        deep = _frontier_priority(0.7, "https://x.com/p", fresh_web_state, depth=3)
        assert shallow > deep

    def test_priority_capped_at_one(self, fresh_web_state):
        # Trusted domain + PDF + high relevance should not exceed 1.0
        p = _frontier_priority(1.0, "https://hrw.org/report.pdf", fresh_web_state)
        assert p <= 1.0


# ── Record visit ───────────────────────────────────────────────────────────────

class TestRecordVisit:
    def test_adds_to_visited_urls(self, fresh_web_state):
        _record_visit(fresh_web_state, "https://hrw.org/doc", "Location Alpha", 5)
        assert "https://hrw.org/doc" in fresh_web_state.visited_urls

    def test_records_chunk_count(self, fresh_web_state):
        _record_visit(fresh_web_state, "https://hrw.org/doc", "Location Alpha", 7)
        assert fresh_web_state.visited_urls["https://hrw.org/doc"]["chunks"] == 7

    def test_updates_domain_stats_on_hit(self, fresh_web_state):
        _record_visit(fresh_web_state, "https://hrw.org/doc", "Location Alpha", 3)
        assert fresh_web_state.domain_scores["hrw.org"].hits == 1
        assert fresh_web_state.domain_scores["hrw.org"].attempts == 1

    def test_updates_domain_stats_on_miss(self, fresh_web_state):
        _record_visit(fresh_web_state, "https://hrw.org/doc", "Location Alpha", 0)
        assert fresh_web_state.domain_scores["hrw.org"].hits == 0
        assert fresh_web_state.domain_scores["hrw.org"].attempts == 1

    def test_multiple_visits_accumulate(self, fresh_web_state):
        _record_visit(fresh_web_state, "https://hrw.org/doc1", "X", 3)
        _record_visit(fresh_web_state, "https://hrw.org/doc2", "X", 0)
        assert fresh_web_state.domain_scores["hrw.org"].hits == 1
        assert fresh_web_state.domain_scores["hrw.org"].attempts == 2


# ── State persistence ──────────────────────────────────────────────────────────

class TestStatePersistence:
    def test_load_state_missing_file_returns_fresh(self, tmp_path, monkeypatch):
        from ai4saw.agents import web_agent
        monkeypatch.setattr(web_agent, "STATE_FILE", tmp_path / "nonexistent.json")
        state = load_state()
        assert isinstance(state, WebAgentState)
        assert state.visited_urls == {}
        assert state.frontier == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from ai4saw.agents import web_agent
        state_path = tmp_path / "state.json"
        monkeypatch.setattr(web_agent, "STATE_FILE", state_path)

        state = WebAgentState()
        state.visited_urls["https://hrw.org/doc"] = {"chunks": 5, "entity": "Location Alpha"}
        state.frontier.append(
            FrontierItem(url="https://amnesty.org/p", priority=0.7,
                         trigger_entity="Commander Alpha", source="ddg")
        )
        state.session_count = 3

        save_state(state)
        assert state_path.exists()

        loaded = load_state()
        assert loaded.session_count == 3
        assert "https://hrw.org/doc" in loaded.visited_urls
        assert loaded.frontier[0].url == "https://amnesty.org/p"

    def test_corrupt_state_file_returns_fresh(self, tmp_path, monkeypatch):
        from ai4saw.agents import web_agent
        state_path = tmp_path / "bad_state.json"
        state_path.write_text("{ invalid json }", encoding="utf-8")
        monkeypatch.setattr(web_agent, "STATE_FILE", state_path)

        state = load_state()
        assert isinstance(state, WebAgentState)
        assert state.visited_urls == {}

    def test_save_creates_parent_directory(self, tmp_path, monkeypatch):
        from ai4saw.agents import web_agent
        nested = tmp_path / "deep" / "nested" / "state.json"
        monkeypatch.setattr(web_agent, "STATE_FILE", nested)

        save_state(WebAgentState())
        assert nested.exists()
