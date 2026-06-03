"""Relation extraction via chain-of-thought prompting.

CoT prompting asks the model to reason step-by-step before committing to
a triple, which measurably reduces hallucinated relations compared to
direct extraction. See spec §5.3.
"""

from __future__ import annotations

import json
import time

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from ai4saw.core.config import settings
from ai4saw.core.models import Relation, RelationResult
from ai4saw.core.providers import get_llm


def _load_prompt() -> dict:
    with open(settings.prompts_dir / "relations_cot.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json_block(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output regardless of wrapping."""
    # Strip code fences
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                pass
    # Try whole string
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    # Find the FIRST complete { ... } block using bracket counting
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    start = -1  # try next block
    raise json.JSONDecodeError("No valid JSON object found", raw, 0)


def _parse_relation_response(raw: str, chunk_id: str) -> RelationResult:
    raw = raw.strip()
    data = _extract_json_block(raw)
    relations = [Relation(**r) for r in data.get("relations", [])]
    return RelationResult(relations=relations, source_chunk_id=chunk_id)


@retry(stop=stop_after_attempt(settings.extraction_retry_limit + 1), wait=wait_fixed(1))
def _call_llm(messages: list, chunk_id: str) -> RelationResult:
    llm = get_llm()
    response = llm.invoke(messages)
    return _parse_relation_response(response.content, chunk_id)


def extract_relations(chunk_text: str, chunk_id: str) -> RelationResult:
    """Extract subject–predicate–object triples from a chunk using CoT prompting."""
    prompt = _load_prompt()
    template: str = prompt["template"]
    user_content = template.replace("{chunk_text}", chunk_text)

    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]

    try:
        result = _call_llm(messages, chunk_id)
        logger.debug(
            f"Relations chunk={chunk_id!r}: {len(result.relations)} triples extracted"
        )
        return result
    except Exception as exc:
        logger.error(f"Relation extraction failed for chunk={chunk_id!r}: {exc}")
        raise


def extract_relations_batch(
    chunks: list[tuple[str, str]],
    delay_between: float = 0.25,
) -> list[RelationResult]:
    results = []
    for text, cid in chunks:
        try:
            results.append(extract_relations(text, cid))
        except Exception:
            logger.warning(f"Skipping chunk {cid!r} after relation extraction failure.")
        if delay_between:
            time.sleep(delay_between)
    return results
