"""Named Entity Recognition via few-shot prompting.

Loads the versioned few-shot prompt from prompts/ner_few_shot.yaml,
calls the active LLM, parses JSON output into NERResult.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from ai4saw.core.config import settings
from ai4saw.core.models import Entity, NERResult
from ai4saw.core.providers import get_llm


def _load_prompt() -> dict:
    prompt_path = settings.prompts_dir / "ner_few_shot.yaml"
    with open(prompt_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_few_shot_block(examples: list[dict]) -> str:
    lines = []
    for ex in examples:
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {json.dumps(ex['output'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def _parse_ner_response(raw: str, chunk_id: str) -> NERResult:
    raw = raw.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    data = json.loads(raw)
    entities = [Entity(**e) for e in data.get("entities", [])]
    return NERResult(entities=entities, source_chunk_id=chunk_id)


@retry(stop=stop_after_attempt(settings.extraction_retry_limit + 1), wait=wait_fixed(1))
def _call_llm_with_retry(messages: list, chunk_id: str, attempt: int = 0) -> NERResult:
    llm = get_llm()
    if attempt > 0:
        # Append an explicit format reminder on retry
        messages = messages + [
            HumanMessage(
                content=(
                    "Your previous response was not valid JSON. "
                    "Return ONLY a JSON object with an 'entities' key. No prose."
                )
            )
        ]
    response = llm.invoke(messages)
    return _parse_ner_response(response.content, chunk_id)


def extract_entities(chunk_text: str, chunk_id: str) -> NERResult:
    """Extract named entities from a single text chunk.

    Uses few-shot prompting from prompts/ner_few_shot.yaml.
    Retries once with an explicit format reminder on JSON parse failure.
    Failures after retry are logged and re-raised — callers decide whether
    to skip or abort.
    """
    prompt = _load_prompt()
    few_shot_block = _build_few_shot_block(prompt.get("examples", []))

    user_content = (
        f"{few_shot_block}\n"
        f"Input: {chunk_text}\n"
        f"Output:"
    )

    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]

    try:
        result = _call_llm_with_retry(messages, chunk_id)
        logger.debug(
            f"NER chunk={chunk_id!r}: {len(result.entities)} entities extracted"
        )
        return result
    except json.JSONDecodeError as exc:
        logger.error(
            f"NER JSON parse failed permanently for chunk={chunk_id!r}: {exc}. "
            f"Check logs/extraction.log for the raw response."
        )
        raise
    except Exception as exc:
        logger.error(f"NER extraction failed for chunk={chunk_id!r}: {exc}")
        raise


def extract_entities_batch(
    chunks: list[tuple[str, str]],   # (text, chunk_id)
    delay_between: float = 0.25,
) -> list[NERResult]:
    """Run NER over a list of (text, chunk_id) pairs.

    delay_between: seconds to sleep between calls — avoids rate-limiting
    on cloud providers. Set to 0 for Ollama.
    """
    results = []
    for text, cid in chunks:
        try:
            results.append(extract_entities(text, cid))
        except Exception:
            logger.warning(f"Skipping chunk {cid!r} after extraction failure.")
        if delay_between:
            time.sleep(delay_between)
    return results
