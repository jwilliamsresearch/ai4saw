"""Event classification — zero-shot first pass, few-shot fallback.

Zero-shot first enables direct comparison of zero-shot vs few-shot
performance on the same corpus, which is a publishable finding per spec §5.4.
"""

from __future__ import annotations

import json
import time

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from ai4saw.core.config import settings
from ai4saw.core.models import EventResult, EventType
from ai4saw.core.providers import get_llm

# Confidence below which we fall back to few-shot
FEW_SHOT_FALLBACK_THRESHOLD = 0.6


def _load_zero_shot_prompt() -> dict:
    with open(settings.prompts_dir / "events_zero_shot.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_few_shot_prompt() -> dict:
    # Few-shot prompt for events lives in its own file created at Phase 2.
    # Fall back gracefully if not yet present.
    path = settings.prompts_dir / "events_few_shot.yaml"
    if not path.exists():
        return _load_zero_shot_prompt()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _robust_json(raw: str) -> dict:
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                pass
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    start = -1
    raise json.JSONDecodeError("No valid JSON found", raw, 0)


def _parse_event_response(raw: str, chunk_id: str) -> EventResult:
    raw = raw.strip()
    data = _robust_json(raw)
    # Normalise event_type to enum
    data["event_type"] = EventType(data.get("event_type", "no_event"))
    data["source_chunk_id"] = chunk_id
    return EventResult(**data)


@retry(stop=stop_after_attempt(settings.extraction_retry_limit + 1), wait=wait_fixed(1))
def _call_llm(messages: list, chunk_id: str) -> EventResult:
    llm = get_llm()
    response = llm.invoke(messages)
    return _parse_event_response(response.content, chunk_id)


def _build_messages(prompt: dict, chunk_text: str) -> list:
    template: str = prompt["template"]
    user_content = template.replace("{chunk_text}", chunk_text)
    return [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]


def classify_event(
    chunk_text: str,
    chunk_id: str,
    record_strategy: bool = True,
) -> EventResult:
    """Classify a chunk's event type.

    Strategy:
    1. Zero-shot pass.
    2. If confidence < FEW_SHOT_FALLBACK_THRESHOLD, retry with few-shot prompt.
    3. Attach the strategy used as a side-channel log so eval scripts can
       reconstruct which result came from which pass.

    record_strategy: if True, logs which pass produced the final result.
    """
    zero_shot_prompt = _load_zero_shot_prompt()
    messages = _build_messages(zero_shot_prompt, chunk_text)

    try:
        result = _call_llm(messages, chunk_id)
    except Exception as exc:
        logger.error(f"Event classification failed for chunk={chunk_id!r}: {exc}")
        raise

    strategy = "zero_shot"

    if result.confidence < FEW_SHOT_FALLBACK_THRESHOLD:
        logger.debug(
            f"Event chunk={chunk_id!r}: zero-shot confidence={result.confidence:.2f} "
            f"< {FEW_SHOT_FALLBACK_THRESHOLD} — falling back to few-shot."
        )
        few_shot_prompt = _load_few_shot_prompt()
        few_shot_messages = _build_messages(few_shot_prompt, chunk_text)
        try:
            result = _call_llm(few_shot_messages, chunk_id)
            strategy = "few_shot"
        except Exception as exc:
            logger.warning(
                f"Few-shot fallback also failed for chunk={chunk_id!r}: {exc}. "
                f"Returning zero-shot result."
            )

    if record_strategy:
        logger.info(
            f"Event chunk={chunk_id!r}: type={result.event_type.value!r} "
            f"confidence={result.confidence:.2f} strategy={strategy}"
        )

    return result


def classify_events_batch(
    chunks: list[tuple[str, str]],
    delay_between: float = 0.25,
) -> list[EventResult]:
    results = []
    for text, cid in chunks:
        try:
            results.append(classify_event(text, cid))
        except Exception:
            logger.warning(f"Skipping chunk {cid!r} after event classification failure.")
        if delay_between:
            time.sleep(delay_between)
    return results
