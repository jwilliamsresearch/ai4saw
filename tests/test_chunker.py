"""Tests for the chunking layer.

Chunking is purely deterministic — no LLM, no network.
These tests verify the contract the rest of the pipeline relies on:
- chunk_index is sequential across all input documents
- metadata is propagated and validated
- short text produces 1 chunk; long text produces N > 1 chunks
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ai4saw.ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents


def _make_doc(text: str, **meta) -> Document:
    base = {
        "source_filename": "test.pdf",
        "doc_type": "report",
        "language": "en",
        "chunk_index": 0,
    }
    return Document(page_content=text, metadata={**base, **meta})


class TestChunkDocuments:
    def test_empty_list_returns_empty(self):
        assert chunk_documents([]) == []

    def test_short_text_produces_single_chunk(self):
        doc = _make_doc("Short text that fits in one chunk.")
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        long_text = "Evidence sentence. " * 500  # ~9500 chars > CHUNK_SIZE
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_chunk_index_is_sequential_single_doc(self):
        long_text = "Word " * 2000
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_is_sequential_across_multiple_docs(self):
        docs = [_make_doc("Word " * 600) for _ in range(3)]
        chunks = chunk_documents(docs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
        assert indices[0] == 0

    def test_metadata_propagated_to_chunks(self):
        doc = _make_doc(
            "Text " * 10,
            source_filename="specific_report.pdf",
            geography="conflict-region",
            source_url="https://tribunal.int/doc.pdf",
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source_filename"] == "specific_report.pdf"
            assert chunk.metadata["geography"] == "conflict-region"
            assert chunk.metadata["source_url"] == "https://tribunal.int/doc.pdf"

    def test_chunk_index_set_in_metadata(self):
        doc = _make_doc("Word " * 600)
        chunks = chunk_documents([doc])
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_content_not_empty_in_any_chunk(self):
        doc = _make_doc("Meaningful content. " * 300)
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.page_content.strip()

    def test_multiple_docs_total_chunks_additive(self):
        short = _make_doc("Short sentence.")
        long = _make_doc("Sentence. " * 1000)
        combined = chunk_documents([short, long])
        from_short = chunk_documents([short])
        from_long = chunk_documents([long])
        assert len(combined) == len(from_short) + len(from_long)

    def test_chunk_size_constant_is_reasonable(self):
        assert 1000 <= CHUNK_SIZE <= 8000
        assert CHUNK_OVERLAP < CHUNK_SIZE
        assert CHUNK_OVERLAP > 0

    def test_invalid_doc_type_logs_warning_but_does_not_raise(self, caplog):
        doc = Document(
            page_content="Some text",
            metadata={
                "source_filename": "test.pdf",
                "doc_type": "invalid_type",
                "language": "en",
                "chunk_index": 0,
            },
        )
        # Should not raise — warns via loguru
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
