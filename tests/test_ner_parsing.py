"""Tests for NER response parsing — no LLM, no network.

_parse_ner_response is the most failure-prone function in the pipeline:
the LLM output format varies, markdown fences appear unexpectedly,
and a parse failure silently loses all entities for a chunk.
These tests lock down every variant we've seen in production.
"""

from __future__ import annotations

import json

import pytest

from ai4saw.extraction.ner import _build_few_shot_block, _parse_ner_response
from ai4saw.core.models import NERResult


CHUNK_ID = "test_chunk_001"


class TestParseNerResponse:
    def test_clean_json(self):
        payload = json.dumps({
            "entities": [
                {"text": "Ratko Mladić", "label": "PERSON", "confidence": 0.95},
                {"text": "VRS", "label": "ORG", "confidence": 0.90},
            ]
        })
        result = _parse_ner_response(payload, CHUNK_ID)
        assert isinstance(result, NERResult)
        assert len(result.entities) == 2
        assert result.source_chunk_id == CHUNK_ID
        assert result.entities[0].text == "Ratko Mladić"

    def test_json_in_backtick_fence(self):
        payload = "```\n" + json.dumps({"entities": [
            {"text": "Srebrenica", "label": "LOCATION", "confidence": 0.98}
        ]}) + "\n```"
        result = _parse_ner_response(payload, CHUNK_ID)
        assert len(result.entities) == 1
        assert result.entities[0].label == "LOCATION"

    def test_json_in_json_fence(self):
        payload = "```json\n" + json.dumps({"entities": [
            {"text": "ICTY", "label": "ORG", "confidence": 0.85}
        ]}) + "\n```"
        result = _parse_ner_response(payload, CHUNK_ID)
        assert result.entities[0].text == "ICTY"

    def test_empty_entities_list(self):
        payload = json.dumps({"entities": []})
        result = _parse_ner_response(payload, CHUNK_ID)
        assert result.entities == []
        assert result.source_chunk_id == CHUNK_ID

    def test_missing_entities_key_returns_empty(self):
        payload = json.dumps({"result": "no entities found"})
        result = _parse_ner_response(payload, CHUNK_ID)
        assert result.entities == []

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_ner_response("not valid json at all", CHUNK_ID)

    def test_whitespace_stripped_before_parse(self):
        payload = "   \n" + json.dumps({"entities": []}) + "\n   "
        result = _parse_ner_response(payload, CHUNK_ID)
        assert result.entities == []

    def test_all_entity_labels_accepted(self):
        labels = ["PERSON", "ORG", "LOCATION", "FACILITY", "EVENT", "GROUP", "LEGAL_INSTRUMENT"]
        entities = [{"text": f"Entity_{l}", "label": l, "confidence": 0.8} for l in labels]
        payload = json.dumps({"entities": entities})
        result = _parse_ner_response(payload, CHUNK_ID)
        assert len(result.entities) == len(labels)

    def test_chunk_id_preserved(self):
        specific_id = "chunk_xyz_789"
        payload = json.dumps({"entities": []})
        result = _parse_ner_response(payload, specific_id)
        assert result.source_chunk_id == specific_id

    def test_entity_with_span_positions(self):
        payload = json.dumps({"entities": [
            {"text": "Mladic", "label": "PERSON", "confidence": 0.9,
             "span_start": 4, "span_end": 10}
        ]})
        result = _parse_ner_response(payload, CHUNK_ID)
        assert result.entities[0].span_start == 4
        assert result.entities[0].span_end == 10


class TestBuildFewShotBlock:
    def test_empty_examples(self):
        block = _build_few_shot_block([])
        assert block == ""

    def test_single_example_format(self):
        examples = [{"input": "Some text", "output": {"entities": []}}]
        block = _build_few_shot_block(examples)
        assert "Input: Some text" in block
        assert "Output:" in block
        assert '"entities"' in block

    def test_multiple_examples_separated(self):
        examples = [
            {"input": "Text A", "output": {"entities": []}},
            {"input": "Text B", "output": {"entities": []}},
        ]
        block = _build_few_shot_block(examples)
        assert "Text A" in block
        assert "Text B" in block

    def test_unicode_preserved_in_output(self):
        examples = [{"input": "Mladić", "output": {"entities": [
            {"text": "Mladić", "label": "PERSON", "confidence": 0.9}
        ]}}]
        block = _build_few_shot_block(examples)
        assert "Mladić" in block
