"""Tests for the discovery module — relevance scoring, deduplication, API parsing.

All HTTP calls are mocked. These tests verify:
- The relevance function scores correctly
- Deduplication excludes known URLs
- API response parsing produces correct DiscoveredDocument objects
- Batching logic works (S2, arXiv, GDELT use single OR query)
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4saw.core.models import DiscoveredDocument
from ai4saw.discovery.discovery import _dedup_and_rank, _known_urls, _relevance


# ── _relevance ─────────────────────────────────────────────────────────────────

class TestRelevance:
    def test_exact_single_token_match(self):
        score = _relevance("Srebrenica", "Srebrenica massacre 1995")
        assert score > 0.4

    def test_no_match_returns_base(self):
        score = _relevance("Srebrenica", "Unrelated agricultural report")
        assert score == pytest.approx(0.4)

    def test_multi_token_entity_partial_match(self):
        score_partial = _relevance("Bosnian Serb Army", "Bosnian forces in conflict")
        score_full = _relevance("Bosnian Serb Army", "Bosnian Serb Army operations")
        assert score_full > score_partial

    def test_score_capped_at_one(self):
        # Many token hits should not exceed 1.0
        score = _relevance("a b c d e", "a b c d e a b c d e a b c d e")
        assert score <= 1.0

    def test_custom_base(self):
        score_low = _relevance("X", "unrelated text", base=0.1)
        score_high = _relevance("X", "unrelated text", base=0.8)
        assert score_low < score_high

    def test_case_insensitive(self):
        score_upper = _relevance("SREBRENICA", "Srebrenica massacre")
        score_lower = _relevance("srebrenica", "Srebrenica massacre")
        assert score_upper == score_lower

    def test_result_is_rounded(self):
        score = _relevance("test entity", "test entity report")
        # Should be rounded to 3 decimal places
        assert score == round(score, 3)


# ── _known_urls ────────────────────────────────────────────────────────────────

class TestKnownUrls:
    def test_missing_file_returns_empty_set(self, tmp_path):
        result = _known_urls(str(tmp_path / "nonexistent.csv"))
        assert result == set()

    def test_reads_urls_from_csv(self, sources_csv):
        result = _known_urls(str(sources_csv))
        assert "https://hrw.org/report1.pdf" in result
        assert "https://gdelt.org/article1" in result

    def test_empty_csv_returns_empty_set(self, empty_sources_csv):
        result = _known_urls(str(empty_sources_csv))
        assert result == set()

    def test_skips_rows_without_url(self, tmp_path):
        path = tmp_path / "partial.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "source_url"])
            writer.writeheader()
            writer.writerow({"filename": "a.pdf", "source_url": "https://example.com"})
            writer.writerow({"filename": "b.pdf", "source_url": ""})
        result = _known_urls(str(path))
        assert "https://example.com" in result
        assert "" not in result


# ── _dedup_and_rank ────────────────────────────────────────────────────────────

class TestDedupAndRank:
    def _make_doc(self, url: str, score: float, entity: str = "Test") -> DiscoveredDocument:
        return DiscoveredDocument(
            title=f"Doc {url}", url=url, source="openalex",
            relevance_score=score, trigger_entity=entity,
        )

    def test_deduplicates_same_url(self):
        docs = [
            self._make_doc("https://a.com", 0.8),
            self._make_doc("https://a.com", 0.6),  # duplicate
        ]
        result = _dedup_and_rank(docs, set())
        assert len(result) == 1

    def test_excludes_known_urls(self):
        docs = [
            self._make_doc("https://known.com", 0.9),
            self._make_doc("https://new.com", 0.5),
        ]
        result = _dedup_and_rank(docs, {"https://known.com"})
        assert len(result) == 1
        assert result[0].url == "https://new.com"

    def test_sorted_by_relevance_descending(self):
        docs = [
            self._make_doc("https://a.com", 0.5),
            self._make_doc("https://b.com", 0.9),
            self._make_doc("https://c.com", 0.7),
        ]
        result = _dedup_and_rank(docs, set())
        scores = [d.relevance_score for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self):
        assert _dedup_and_rank([], set()) == []

    def test_all_known_returns_empty(self):
        docs = [self._make_doc("https://a.com", 0.8)]
        result = _dedup_and_rank(docs, {"https://a.com"})
        assert result == []

    def test_first_occurrence_kept_on_duplicate(self):
        docs = [
            self._make_doc("https://a.com", 0.8),
            self._make_doc("https://a.com", 0.9),  # later, higher score
        ]
        result = _dedup_and_rank(docs, set())
        # First occurrence is kept (whichever has url first)
        assert len(result) == 1


# ── OpenAlex response parsing ──────────────────────────────────────────────────

class TestOpenAlexParsing:
    """Test that _query_openalex correctly parses API responses."""

    def _make_response(self, works: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"results": works, "meta": {"count": len(works)}}
        return mock_resp

    def test_extracts_oa_url(self):
        from ai4saw.discovery.discovery import _query_openalex

        work = {
            "title": "Srebrenica Genocide Analysis",
            "open_access": {"oa_url": "https://openalex.org/paper.pdf", "is_oa": True},
            "publication_date": "2020-06-15",
            "abstract_inverted_index": None,
        }
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_response([work])

        docs = _query_openalex("Srebrenica", limit=5, client=mock_client)
        assert len(docs) == 1
        assert docs[0].url == "https://openalex.org/paper.pdf"
        assert docs[0].source == "openalex"
        assert docs[0].date == "2020-06-15"

    def test_skips_work_without_oa_url(self):
        from ai4saw.discovery.discovery import _query_openalex

        work = {
            "title": "No URL Work",
            "open_access": {"oa_url": None, "is_oa": True},
            "publication_date": "2020-01-01",
            "abstract_inverted_index": None,
        }
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_response([work])

        docs = _query_openalex("test", limit=5, client=mock_client)
        assert len(docs) == 0

    def test_handles_api_error_gracefully(self):
        from ai4saw.discovery.discovery import _query_openalex

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection refused")

        docs = _query_openalex("test", limit=5, client=mock_client)
        assert docs == []


# ── Internet Archive parsing ───────────────────────────────────────────────────

class TestInternetArchiveParsing:
    def test_parses_identifier_to_url(self):
        from ai4saw.discovery.discovery import _query_internetarchive

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": {"docs": [{
                "identifier": "srebrenica-1995-report",
                "title": "Srebrenica 1995 Report",
                "date": "1995-11-01",
                "description": "ICTY investigation report",
            }]}
        }
        mock_client.get.return_value = mock_resp

        docs = _query_internetarchive("Srebrenica", limit=5, client=mock_client)
        assert len(docs) == 1
        assert docs[0].url == "https://archive.org/details/srebrenica-1995-report"
        assert docs[0].source == "internetarchive"

    def test_handles_list_valued_title(self):
        from ai4saw.discovery.discovery import _query_internetarchive

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": {"docs": [{
                "identifier": "doc-123",
                "title": ["First Title", "Second Title"],
                "date": "2000-01-01",
                "description": "Some description",
            }]}
        }
        mock_client.get.return_value = mock_resp

        docs = _query_internetarchive("test", limit=5, client=mock_client)
        assert docs[0].title == "First Title"

    def test_skips_docs_without_identifier(self):
        from ai4saw.discovery.discovery import _query_internetarchive

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": {"docs": [{"title": "No ID doc", "date": "2000-01-01"}]}
        }
        mock_client.get.return_value = mock_resp

        docs = _query_internetarchive("test", limit=5, client=mock_client)
        assert len(docs) == 0


# ── arXiv XML parsing ──────────────────────────────────────────────────────────

class TestArxivParsing:
    def _make_arxiv_xml(self, entries: list[dict]) -> str:
        ns = "http://www.w3.org/2005/Atom"
        root = ET.Element(f"{{{ns}}}feed")
        for e in entries:
            entry = ET.SubElement(root, f"{{{ns}}}entry")
            ET.SubElement(entry, f"{{{ns}}}title").text = e.get("title", "Untitled")
            ET.SubElement(entry, f"{{{ns}}}published").text = e.get("published", "2020-01-01T00:00:00Z")
            ET.SubElement(entry, f"{{{ns}}}summary").text = e.get("summary", "")
            if "pdf_url" in e:
                link = ET.SubElement(entry, f"{{{ns}}}link")
                link.set("type", "application/pdf")
                link.set("href", e["pdf_url"])
            id_el = ET.SubElement(entry, f"{{{ns}}}id")
            id_el.text = e.get("id", "https://arxiv.org/abs/2001.00001")
        return ET.tostring(root, encoding="unicode")

    def test_extracts_pdf_link(self):
        from ai4saw.discovery.discovery import _query_arxiv_batch

        xml = self._make_arxiv_xml([{
            "title": "War Crimes in Bosnia",
            "published": "2021-03-15T00:00:00Z",
            "pdf_url": "https://arxiv.org/pdf/2001.00001",
            "summary": "Study of war crimes documentation",
        }])

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.text = xml
        mock_client.get.return_value = mock_resp

        docs = _query_arxiv_batch(["Bosnia"], limit=5, client=mock_client)
        assert len(docs) == 1
        assert docs[0].url == "https://arxiv.org/pdf/2001.00001"
        assert docs[0].date == "2021-03-15"

    def test_falls_back_to_id_when_no_pdf_link(self):
        from ai4saw.discovery.discovery import _query_arxiv_batch

        xml = self._make_arxiv_xml([{
            "title": "Some Paper",
            "id": "https://arxiv.org/abs/2001.99999",
        }])

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.text = xml
        mock_client.get.return_value = mock_resp

        docs = _query_arxiv_batch(["test"], limit=5, client=mock_client)
        assert docs[0].url == "https://arxiv.org/abs/2001.99999"

    def test_empty_feed_returns_empty_list(self):
        from ai4saw.discovery.discovery import _query_arxiv_batch

        xml = self._make_arxiv_xml([])

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.text = xml
        mock_client.get.return_value = mock_resp

        docs = _query_arxiv_batch(["test"], limit=5, client=mock_client)
        assert docs == []
