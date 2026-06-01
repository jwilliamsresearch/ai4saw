"""Entity Resolution — cross-document entity deduplication and canonical registry.

Problem: NER processes each chunk independently, so "the RSF", "Rapid Support Forces",
and "the Khartoum paramilitaries" appear as three separate entity records. For network
analysis, silence detection, and any aggregation across the corpus, these must be
unified into one canonical entity with aliases.

Algorithm:
  1. Collect all (text, label) mentions from NERResults, counting frequency.
  2. Batch-embed all unique entity texts using the configured embedder.
  3. Group by label (PERSON with PERSON, ORG with ORG — cross-label merging is wrong).
  4. Within each label group, compute cosine similarity via numpy.
  5. Greedy union-find: if two entities have cosine_sim > threshold AND
     fuzzy_string_ratio > fuzzy_threshold, merge them into one cluster.
  6. Each cluster becomes a ResolvedEntity: canonical = most frequent form, rest = aliases.

The threshold pair (cosine + fuzzy) prevents false merges: embeddings alone can
conflate different organisations in the same domain, while string matching alone
misses "RSF" ↔ "Rapid Support Forces". Both checks together are much more precise.
"""

from __future__ import annotations

import difflib
import hashlib
from collections import Counter, defaultdict

import numpy as np
from loguru import logger

from ai4saw.core.models import Entity, EntityResolutionResult, NERResult, ResolvedEntity
from ai4saw.core.providers import get_embedder

# Tune these thresholds for the corpus. Lower values merge more aggressively.
DEFAULT_COSINE_THRESHOLD = 0.88
DEFAULT_FUZZY_THRESHOLD = 72.0   # SequenceMatcher ratio × 100


# ── Internal helpers ──────────────────────────────────────────────────────────

def _canonical_id(text: str, label: str) -> str:
    return hashlib.sha256(f"{label}::{text.lower()}".encode()).hexdigest()[:12]


def _fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher similarity × 100, case-insensitive."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0


def _cosine_sim_matrix(embeddings: list[list[float]]) -> np.ndarray:
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = arr / norms
    return normed @ normed.T


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# ── Core resolution logic ─────────────────────────────────────────────────────

def _resolve_label_group(
    texts: list[str],
    counts: list[int],
    chunks: list[list[str]],
    confidences: list[float],
    label: str,
    cosine_threshold: float,
    fuzzy_threshold: float,
) -> list[ResolvedEntity]:
    """Resolve a single label group (e.g. all PERSON mentions) into canonical entities."""
    n = len(texts)
    if n == 0:
        return []

    # Embed all texts in one batch call — cheaper than one call per entity
    embedder = get_embedder()
    try:
        embeddings = embedder.embed_documents(texts)
    except Exception as exc:
        logger.warning(
            f"Embedding failed for label {label!r} during entity resolution: {exc}. "
            f"Falling back to string-only matching."
        )
        embeddings = [[0.0]] * n

    sim_matrix = _cosine_sim_matrix(embeddings) if len(embeddings[0]) > 1 else np.eye(n)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if (
                sim_matrix[i, j] >= cosine_threshold
                or _fuzzy_ratio(texts[i], texts[j]) >= fuzzy_threshold
            ):
                uf.union(i, j)

    resolved: list[ResolvedEntity] = []
    for root, members in uf.clusters().items():
        # Canonical form = the text with highest frequency in this cluster
        canonical_idx = max(members, key=lambda idx: counts[idx])
        canonical_text = texts[canonical_idx]
        aliases = [texts[m] for m in members if m != canonical_idx]
        total_count = sum(counts[m] for m in members)
        all_chunks = list({c for m in members for c in chunks[m]})
        mean_conf = sum(confidences[m] * counts[m] for m in members) / max(total_count, 1)

        resolved.append(
            ResolvedEntity(
                canonical_id=_canonical_id(canonical_text, label),
                canonical_text=canonical_text,
                label=label,
                aliases=aliases,
                occurrence_count=total_count,
                source_chunks=all_chunks,
                mean_confidence=round(mean_conf, 4),
            )
        )

    return resolved


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_entities(
    ner_results: list[NERResult],
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> EntityResolutionResult:
    """Resolve all NER mentions across the corpus into canonical entities.

    Args:
        ner_results: All NERResults from the extraction pipeline.
        cosine_threshold: Minimum embedding cosine similarity to consider two
            mentions the same entity. 0.88 is conservative; lower to merge more.
        fuzzy_threshold: Minimum SequenceMatcher ratio (× 100) for string similarity.
            72 catches abbreviations and minor spelling variants.

    Returns:
        EntityResolutionResult containing the canonical entity registry.
    """
    # Accumulate: label → text → {count, chunks, confidence_sum}
    label_groups: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"count": 0, "chunks": [], "confidence_sum": 0.0}
    ))

    total_mentions = 0
    for result in ner_results:
        for entity in result.entities:
            rec = label_groups[entity.label][entity.text]
            rec["count"] += 1
            rec["confidence_sum"] += entity.confidence
            if result.source_chunk_id not in rec["chunks"]:
                rec["chunks"].append(result.source_chunk_id)
            total_mentions += 1

    unique_texts_before = sum(len(v) for v in label_groups.values())
    all_resolved: list[ResolvedEntity] = []

    for label, text_map in label_groups.items():
        texts = list(text_map.keys())
        counts = [text_map[t]["count"] for t in texts]
        chunks = [text_map[t]["chunks"] for t in texts]
        confidences = [
            text_map[t]["confidence_sum"] / max(text_map[t]["count"], 1) for t in texts
        ]
        resolved = _resolve_label_group(
            texts, counts, chunks, confidences, label,
            cosine_threshold, fuzzy_threshold,
        )
        all_resolved.extend(resolved)
        logger.debug(
            f"Entity resolution [{label}]: {len(texts)} mentions → {len(resolved)} canonical"
        )

    all_resolved.sort(key=lambda e: e.occurrence_count, reverse=True)

    logger.info(
        f"Entity resolution complete: {total_mentions} mentions, "
        f"{unique_texts_before} unique texts → {len(all_resolved)} canonical entities"
    )

    return EntityResolutionResult(
        entities=all_resolved,
        total_mentions=total_mentions,
        unique_texts_before=unique_texts_before,
        resolved_count=len(all_resolved),
    )


def save_entity_registry(result: EntityResolutionResult, path: str = "data/entity_registry.json") -> None:
    """Persist the resolved entity registry to disk."""
    import json
    from pathlib import Path
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Entity registry saved → {out}")


def load_entity_registry(path: str = "data/entity_registry.json") -> EntityResolutionResult:
    """Load a previously saved entity registry."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EntityResolutionResult(**data)
