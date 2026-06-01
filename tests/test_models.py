"""Tests for all Pydantic v2 schemas in core/models.py.

These are the contracts the rest of the pipeline depends on.
A schema change that breaks validation here breaks everything downstream.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ai4saw.core.models import (
    AgentResponse,
    AgentStep,
    ChunkMetadata,
    ContradictionPair,
    ContradictionType,
    DiscoveredDocument,
    DiscoveryResult,
    Entity,
    EventResult,
    EventType,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    NERResult,
    QAResponse,
    Relation,
    RelationResult,
    ResolvedEntity,
    SilenceCandidate,
)


# ── ChunkMetadata ──────────────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_valid(self):
        m = ChunkMetadata(
            source_filename="report.pdf",
            doc_type="report",
            language="en",
            chunk_index=0,
        )
        assert m.source_filename == "report.pdf"
        assert m.source_url is None
        assert m.geography is None

    def test_all_fields(self):
        m = ChunkMetadata(
            source_filename="report.pdf",
            source_url="https://hrw.org/report.pdf",
            doc_type="legal",
            language="bs",
            date_published=date(1995, 7, 11),
            geography="conflict-region",
            chunk_index=5,
        )
        assert m.date_published == date(1995, 7, 11)
        assert m.chunk_index == 5

    @pytest.mark.parametrize("doc_type", ["report", "news", "legal", "grey_literature"])
    def test_valid_doc_types(self, doc_type):
        m = ChunkMetadata(source_filename="f.pdf", doc_type=doc_type, language="en", chunk_index=0)
        assert m.doc_type == doc_type

    def test_invalid_doc_type(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(source_filename="f.pdf", doc_type="blog_post", language="en", chunk_index=0)

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(doc_type="report", language="en", chunk_index=0)  # missing source_filename


# ── Entity ─────────────────────────────────────────────────────────────────────

class TestEntity:
    @pytest.mark.parametrize("label", [
        "PERSON", "ORG", "LOCATION", "FACILITY", "EVENT", "GROUP", "LEGAL_INSTRUMENT"
    ])
    def test_all_valid_labels(self, label):
        e = Entity(text="Test", label=label, confidence=0.9)
        assert e.label == label

    def test_confidence_bounds(self):
        Entity(text="X", label="PERSON", confidence=0.0)
        Entity(text="X", label="PERSON", confidence=1.0)

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Entity(text="X", label="PERSON", confidence=1.1)

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            Entity(text="X", label="PERSON", confidence=-0.1)

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            Entity(text="X", label="UNKNOWN_TYPE", confidence=0.5)

    def test_optional_spans(self):
        e = Entity(text="Commander Alpha", label="PERSON", confidence=0.95, span_start=10, span_end=16)
        assert e.span_start == 10
        assert e.span_end == 16


# ── NERResult ──────────────────────────────────────────────────────────────────

class TestNERResult:
    def test_empty_entities(self):
        r = NERResult(entities=[], source_chunk_id="chunk_001")
        assert r.entities == []

    def test_populated(self):
        r = NERResult(
            entities=[Entity(text="Commander Alpha", label="PERSON", confidence=0.95)],
            source_chunk_id="chunk_001",
        )
        assert len(r.entities) == 1
        assert r.entities[0].text == "Commander Alpha"


# ── Relation ───────────────────────────────────────────────────────────────────

class TestRelation:
    def test_valid(self):
        r = Relation(
            subject="Commander Alpha",
            predicate="commanded",
            object="Armed-Group-Alpha",
            confidence=0.9,
            evidence="Commander Alpha commanded Armed-Group-Alpha forces",
        )
        assert r.predicate == "commanded"
        assert r.location is None
        assert r.date is None

    def test_with_optional_fields(self):
        r = Relation(
            subject="Armed-Group-Beta",
            predicate="attacked",
            object="Location Beta",
            location="Province Beta",
            date="2023-04-15",
            confidence=0.85,
            evidence="Armed-Group-Beta forces attacked Location Beta on 15 April",
        )
        assert r.date == "2023-04-15"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Relation(subject="A", predicate="B", object="C", confidence=1.5, evidence="X")


# ── EventResult ────────────────────────────────────────────────────────────────

class TestEventResult:
    @pytest.mark.parametrize("event_type", [e.value for e in EventType])
    def test_all_event_types(self, event_type):
        e = EventResult(
            event_type=event_type,
            confidence=0.8,
            source_chunk_id="chunk_001",
        )
        assert e.event_type.value == event_type

    def test_no_event(self):
        e = EventResult(event_type="no_event", confidence=0.1, source_chunk_id="c1")
        assert e.event_type == EventType.NO_EVENT


# ── DiscoveredDocument ─────────────────────────────────────────────────────────

class TestDiscoveredDocument:
    @pytest.mark.parametrize("source", [
        "openalex", "gdelt", "semanticscholar", "arxiv",
        "internetarchive", "duckduckgo", "wikipedia", "crossref", "manual",
    ])
    def test_all_valid_sources(self, source):
        doc = DiscoveredDocument(
            title="Test", url="https://example.com",
            source=source, relevance_score=0.5, trigger_entity="Location Alpha",
        )
        assert doc.source == source

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            DiscoveredDocument(
                title="Test", url="https://x.com",
                source="twitter", relevance_score=0.5, trigger_entity="X",
            )

    def test_relevance_score_bounds(self):
        with pytest.raises(ValidationError):
            DiscoveredDocument(
                title="T", url="https://x.com",
                source="manual", relevance_score=1.5, trigger_entity="X",
            )

    def test_optional_fields(self):
        doc = DiscoveredDocument(
            title="T", url="https://x.com",
            source="openalex", relevance_score=0.7, trigger_entity="X",
        )
        assert doc.date is None
        assert doc.snippet is None


# ── DiscoveryResult ────────────────────────────────────────────────────────────

class TestDiscoveryResult:
    def test_valid(self, sample_discovered_doc):
        r = DiscoveryResult(
            trigger_entities=["Location Alpha"],
            documents=[sample_discovered_doc],
            query_count=5,
            new_documents=1,
        )
        assert r.new_documents == 1
        assert len(r.documents) == 1


# ── KnowledgeGraph ─────────────────────────────────────────────────────────────

class TestKnowledgeGraphEdge:
    def test_temporal_fields(self):
        edge = KnowledgeGraphEdge(
            source_id="n1", target_id="n2",
            predicate="commanded", confidence=0.9,
            evidence="Commander Alpha commanded Armed-Group-Alpha",
            source_chunk_id="c1",
            valid_from="1992-04-01",
            valid_to="1996-12-14",
        )
        assert edge.valid_from == "1992-04-01"
        assert edge.valid_to == "1996-12-14"

    def test_open_ended_temporal(self):
        edge = KnowledgeGraphEdge(
            source_id="n1", target_id="n2",
            predicate="led", confidence=0.8,
            evidence="X", source_chunk_id="c1",
        )
        assert edge.valid_to is None


# ── ContradictionPair ──────────────────────────────────────────────────────────

class TestContradictionPair:
    @pytest.mark.parametrize("ctype", [e.value for e in ContradictionType])
    def test_all_contradiction_types(self, ctype):
        p = ContradictionPair(
            chunk_id_a="c1", chunk_id_b="c2",
            source_a="report_a.pdf", source_b="report_b.pdf",
            claim_a="8,000 were killed", claim_b="6,000 were killed",
            contradiction_type=ctype, confidence=0.8,
            explanation="Different casualty figures",
        )
        assert p.contradiction_type.value == ctype


# ── SilenceCandidate ───────────────────────────────────────────────────────────

class TestSilenceCandidate:
    def test_silence_score_semantics(self):
        s = SilenceCandidate(
            event_id="conflict-events-dataset_001",
            location="Location C",
            date="1992-04-01",
            conflict_intensity=0.9,
            retrieval_confidence=0.1,
            silence_score=0.8,
            candidate_reason="High conflict, low documentation",
        )
        assert s.silence_score == pytest.approx(0.8)


# ── ResolvedEntity ─────────────────────────────────────────────────────────────

class TestResolvedEntity:
    def test_aliases(self):
        e = ResolvedEntity(
            canonical_id="abc123",
            canonical_text="Rapid Support Forces",
            label="ORG",
            aliases=["Armed-Group-Beta", "the paramilitaries", "Janjaweed"],
            occurrence_count=47,
            source_chunks=["c1", "c2"],
            mean_confidence=0.88,
        )
        assert "Armed-Group-Beta" in e.aliases
        assert e.occurrence_count == 47
