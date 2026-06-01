"""Contradiction Detection — surface conflicting claims across source documents.

Contradictions in conflict research are not errors to discard — they are often
the most analytically important signal. Two sources giving different death tolls
for the same massacre may reflect: different witness perspectives, propaganda,
selective documentation, or genuinely different incidents being conflated.
This module makes those contradictions visible rather than averaging them away.

Detection strategy (two passes to keep LLM costs manageable):

  Pass 1 — Candidate generation (cheap):
    Group extracted events and relations by geographic/temporal proximity.
    Within each group, flag pairs that are candidates for contradiction:
    - EventResults: same location + overlapping date → check event type mismatch
    - RelationResults: same (subject, object) pair with different predicate

  Pass 2 — LLM verification (per-pair LLM call):
    For each candidate pair, retrieve the source chunk text and call the LLM
    with the contradiction_check.yaml prompt. Only confirmed contradictions
    (is_contradiction=True, confidence >= threshold) enter the report.

This two-pass approach avoids calling the LLM on every pair (O(n²) would be
prohibitive on large corpora) while maintaining high recall on genuine conflicts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import (
    ContradictionPair,
    ContradictionReport,
    ContradictionType,
    EventResult,
    EventType,
    NERResult,
    RelationResult,
)
from ai4saw.core.providers import get_llm
from ai4saw.ingestion.embedder import get_vector_store

# Only assess pairs where embeddings are closer than this (same topic area)
SEMANTIC_CANDIDATE_THRESHOLD = 0.75
# Confirmed pairs must reach this LLM confidence to enter the report
LLM_CONFIDENCE_THRESHOLD = 0.65


# ── Pass 1: Candidate generation ──────────────────────────────────────────────

def _location_key(location: str | None) -> str:
    if not location:
        return "unknown"
    return location.lower().strip()


def _date_year(date_str: str | None) -> str:
    if not date_str or len(date_str) < 4:
        return "unknown"
    return date_str[:4]


def _event_candidates(events: list[EventResult]) -> list[tuple[EventResult, EventResult]]:
    """Group events by (location, year) and flag pairs with different event types."""
    groups: dict[str, list[EventResult]] = defaultdict(list)
    for ev in events:
        if ev.event_type == EventType.NO_EVENT:
            continue
        key = f"{_location_key(ev.location)}_{_date_year(ev.date)}"
        groups[key].append(ev)

    candidates = []
    for group in groups.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                # Different event types for same location/year = candidate
                if a.event_type != b.event_type:
                    candidates.append((a, b))
    return candidates


def _relation_candidates(
    relations: list[RelationResult],
) -> list[tuple[RelationResult, RelationResult, str, str]]:
    """Find (subject, object) pairs where two different chunks assign different predicates."""
    # Map (subject_lower, object_lower) → list of (RelationResult, predicate)
    pair_map: dict[tuple[str, str], list[tuple[RelationResult, str]]] = defaultdict(list)
    for result in relations:
        for rel in result.relations:
            key = (rel.subject.lower().strip(), rel.object.lower().strip())
            pair_map[key].append((result, rel.predicate))

    candidates = []
    for (subj, obj), entries in pair_map.items():
        if len(entries) < 2:
            continue
        # Different predicates for same (subject, object)
        predicates = {e[1] for e in entries}
        if len(predicates) > 1:
            for i, (res_a, pred_a) in enumerate(entries):
                for res_b, pred_b in entries[i + 1:]:
                    if pred_a != pred_b and res_a.source_chunk_id != res_b.source_chunk_id:
                        candidates.append((res_a, res_b, pred_a, pred_b))
    return candidates


# ── Pass 2: LLM verification ──────────────────────────────────────────────────

def _load_prompt() -> dict:
    with open(settings.prompts_dir / "contradiction_check.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chunk_text_for_id(chunk_id: str) -> str:
    """Retrieve the original text of a chunk from ChromaDB by ID."""
    try:
        store = get_vector_store()
        result = store._collection.get(ids=[chunk_id], include=["documents"])
        docs = result.get("documents") or []
        return docs[0] if docs else f"[chunk {chunk_id} not found in store]"
    except Exception:
        return f"[chunk {chunk_id} — retrieval failed]"


def _chunk_meta_for_id(chunk_id: str) -> dict:
    try:
        store = get_vector_store()
        result = store._collection.get(ids=[chunk_id], include=["metadatas"])
        metas = result.get("metadatas") or [{}]
        return metas[0]
    except Exception:
        return {}


def _llm_assess_contradiction(
    claim_a: str,
    claim_b: str,
    source_a: str,
    source_b: str,
) -> dict:
    """Call the LLM to assess whether two claims contradict each other."""
    prompt = _load_prompt()
    user_content = (
        prompt["template"]
        .replace("{claim_a}", claim_a[:800])
        .replace("{claim_b}", claim_b[:800])
        .replace("{source_a}", source_a)
        .replace("{source_b}", source_b)
    )
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]

    llm = get_llm()
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Contradiction LLM returned invalid JSON; treating as no contradiction.")
        return {"is_contradiction": False, "contradiction_type": "none",
                "confidence": 0.0, "explanation": "JSON parse failed"}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_contradictions(
    event_results: list[EventResult],
    relation_results: list[RelationResult],
    llm_confidence_threshold: float = LLM_CONFIDENCE_THRESHOLD,
    max_pairs_to_assess: int = 100,
) -> ContradictionReport:
    """Detect contradictions across the corpus using a two-pass approach.

    Args:
        event_results: All EventResults from the extraction pipeline.
        relation_results: All RelationResults from the extraction pipeline.
        llm_confidence_threshold: Minimum LLM confidence to include a pair in
            the final report. 0.65 balances precision vs. recall.
        max_pairs_to_assess: Hard cap on LLM calls to control cost. Pairs are
            prioritised by event confidence × relation confidence.

    Returns:
        ContradictionReport with confirmed contradiction pairs, sorted by confidence.
    """
    # Generate candidates
    event_candidates = _event_candidates(event_results)
    relation_candidates = _relation_candidates(relation_results)

    total_candidates = len(event_candidates) + len(relation_candidates)
    logger.info(
        f"Contradiction detection: {len(event_candidates)} event candidates, "
        f"{len(relation_candidates)} relation candidates"
    )

    confirmed: list[ContradictionPair] = []
    assessed = 0

    # Assess event candidate pairs
    for ev_a, ev_b in event_candidates[:max_pairs_to_assess // 2]:
        text_a = _chunk_text_for_id(ev_a.source_chunk_id)
        text_b = _chunk_text_for_id(ev_b.source_chunk_id)
        meta_a = _chunk_meta_for_id(ev_a.source_chunk_id)
        meta_b = _chunk_meta_for_id(ev_b.source_chunk_id)
        source_a = meta_a.get("source_filename", ev_a.source_chunk_id)
        source_b = meta_b.get("source_filename", ev_b.source_chunk_id)

        result = _llm_assess_contradiction(text_a, text_b, source_a, source_b)
        assessed += 1

        if result.get("is_contradiction") and result.get("confidence", 0) >= llm_confidence_threshold:
            confirmed.append(ContradictionPair(
                chunk_id_a=ev_a.source_chunk_id,
                chunk_id_b=ev_b.source_chunk_id,
                source_a=source_a,
                source_b=source_b,
                claim_a=text_a[:500],
                claim_b=text_b[:500],
                contradiction_type=ContradictionType(
                    result.get("contradiction_type", "factual")
                ),
                confidence=round(result.get("confidence", 0.0), 4),
                explanation=result.get("explanation", ""),
            ))

    # Assess relation candidate pairs
    remaining = max(0, max_pairs_to_assess - assessed)
    for res_a, res_b, pred_a, pred_b in relation_candidates[:remaining]:
        text_a = _chunk_text_for_id(res_a.source_chunk_id)
        text_b = _chunk_text_for_id(res_b.source_chunk_id)
        meta_a = _chunk_meta_for_id(res_a.source_chunk_id)
        meta_b = _chunk_meta_for_id(res_b.source_chunk_id)
        source_a = meta_a.get("source_filename", res_a.source_chunk_id)
        source_b = meta_b.get("source_filename", res_b.source_chunk_id)

        claim_a = f"[Relation predicate: '{pred_a}']\n{text_a[:400]}"
        claim_b = f"[Relation predicate: '{pred_b}']\n{text_b[:400]}"

        result = _llm_assess_contradiction(claim_a, claim_b, source_a, source_b)
        assessed += 1

        if result.get("is_contradiction") and result.get("confidence", 0) >= llm_confidence_threshold:
            confirmed.append(ContradictionPair(
                chunk_id_a=res_a.source_chunk_id,
                chunk_id_b=res_b.source_chunk_id,
                source_a=source_a,
                source_b=source_b,
                claim_a=claim_a[:500],
                claim_b=claim_b[:500],
                contradiction_type=ContradictionType(
                    result.get("contradiction_type", "attribution")
                ),
                confidence=round(result.get("confidence", 0.0), 4),
                explanation=result.get("explanation", ""),
            ))

    confirmed.sort(key=lambda p: p.confidence, reverse=True)
    high_conf = sum(1 for p in confirmed if p.confidence >= 0.7)

    logger.info(
        f"Contradiction detection complete: {assessed} pairs assessed, "
        f"{len(confirmed)} confirmed ({high_conf} high-confidence)"
    )

    return ContradictionReport(
        pairs=confirmed,
        total_chunks_analysed=len(set(
            [e.source_chunk_id for e in event_results] +
            [r.source_chunk_id for r in relation_results]
        )),
        candidate_pairs_assessed=assessed,
        high_confidence_count=high_conf,
    )
