"""Informational silence detection — two approaches implemented per spec §5.6.

Approach A — Expectation gap:
  For each known event (your-dataset/conflict-events-dataset), query the corpus. Events with low
  retrieval confidence despite high conflict intensity are candidate silences.

Approach B — Density mapping:
  Embed all documents. Cluster by geographic tag and time window.
  Compare cluster density against conflict-events-dataset conflict intensity for the same cell.
  Sparse clusters in high-intensity zones are structural silences.

This is the novel contribution of AI4SAW: no existing NLP pipeline for
conflict research explicitly operationalises informational silence as a
measurable quantity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_chroma import Chroma
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import SilenceCandidate
from ai4saw.core.providers import get_embedder
from ai4saw.ingestion.embedder import get_vector_store


# ── Approach A — Expectation Gap ──────────────────────────────────────────────

def compute_retrieval_confidence(
    query: str,
    store: Chroma,
    top_k: int = 3,
) -> float:
    """Return the mean similarity score of the top-K retrieved chunks for a query.

    ChromaDB returns distances (lower = more similar). We convert to a
    confidence score in [0, 1] via: confidence = 1 - mean_distance.
    Distances are L2-normalised embeddings so this approximates cosine sim.
    """
    results = store.similarity_search_with_relevance_scores(query, k=top_k)
    if not results:
        return 0.0
    scores = [score for _, score in results]
    return float(np.mean(scores))


def detect_silence_expectation_gap(
    reference_events: list[dict],
    intensity_key: str = "conflict_intensity",
    location_key: str = "location",
    date_key: str = "date",
    id_key: str = "event_id",
) -> list[SilenceCandidate]:
    """Approach A: compare known events against corpus retrieval confidence.

    Args:
        reference_events: List of dicts with at minimum:
            event_id, location, date, conflict_intensity.
            These come from your-dataset or conflict-events-dataset exports.
        intensity_key / location_key / date_key / id_key:
            Column names in the reference_events dicts.

    Returns:
        List of SilenceCandidate sorted by silence_score descending.
        Higher score = larger gap between expected coverage and actual coverage.
    """
    store = get_vector_store()
    candidates: list[SilenceCandidate] = []

    for event in reference_events:
        event_id = str(event.get(id_key, "unknown"))
        location = str(event.get(location_key, ""))
        event_date = str(event.get(date_key, ""))
        intensity = float(event.get(intensity_key, 0.0))

        query = f"{location} {event_date}".strip()
        ret_conf = compute_retrieval_confidence(query, store)
        silence_score = round(intensity - ret_conf, 4)

        reason_parts = []
        if intensity > 0.7 and ret_conf < 0.3:
            reason_parts.append("high-intensity event with very low corpus coverage")
        elif intensity > ret_conf + 0.3:
            reason_parts.append("notable gap between conflict intensity and documentation")
        else:
            reason_parts.append("moderate expectation gap")

        candidates.append(
            SilenceCandidate(
                event_id=event_id,
                location=location,
                date=event_date,
                conflict_intensity=round(intensity, 4),
                retrieval_confidence=round(ret_conf, 4),
                silence_score=silence_score,
                candidate_reason="; ".join(reason_parts),
            )
        )
        logger.debug(
            f"Silence A | event={event_id} location={location!r} "
            f"intensity={intensity:.3f} ret_conf={ret_conf:.3f} "
            f"silence={silence_score:.3f}"
        )

    candidates.sort(key=lambda c: c.silence_score, reverse=True)
    logger.info(
        f"Expectation-gap silence detection: {len(candidates)} events assessed, "
        f"{sum(1 for c in candidates if c.silence_score > 0.3)} high-confidence silences."
    )
    return candidates


# ── Approach B — Density Mapping ──────────────────────────────────────────────

@dataclass
class DensityCell:
    geography: str
    time_window: str            # e.g. "1992-Q3"
    document_count: int
    chunk_count: int
    mean_embedding: list[float]
    conflict_intensity: float   # from conflict-events-dataset reference; 0.0 if not available
    silence_score: float        # conflict_intensity - normalised doc density


def detect_silence_density_map(
    acled_reference: dict[str, float],  # {"{geography}_{time_window}": intensity}
    min_chunks_for_coverage: int = 5,
) -> list[DensityCell]:
    """Approach B: cluster corpus by geography+time, compare against conflict-events-dataset density.

    Args:
        acled_reference: Mapping of "location_timewindow" → conflict_intensity.
            Build this from an conflict-events-dataset export filtered to your pilot region.
        min_chunks_for_coverage: Chunk count below which a cell is considered sparse.

    Returns:
        DensityCells sorted by silence_score descending.
    """
    store = get_vector_store()

    # Pull all stored chunks with metadata
    collection = store._collection  # access underlying chromadb collection
    result = collection.get(include=["metadatas", "embeddings"])

    metadatas: list[dict] = result.get("metadatas") or []
    embeddings: list[list[float]] = result.get("embeddings") or []

    # Group chunks by geography + date quarter
    cells: dict[str, dict] = {}
    for meta, emb in zip(metadatas, embeddings):
        geo = meta.get("geography") or "unknown"
        date_str = meta.get("date_published") or ""
        time_window = _date_to_quarter(date_str)
        key = f"{geo}_{time_window}"

        if key not in cells:
            cells[key] = {
                "geography": geo,
                "time_window": time_window,
                "chunks": [],
                "embeddings": [],
            }
        cells[key]["chunks"].append(meta)
        cells[key]["embeddings"].append(emb)

    density_cells: list[DensityCell] = []
    max_chunks = max((len(v["chunks"]) for v in cells.values()), default=1)

    for key, cell_data in cells.items():
        chunk_count = len(cell_data["chunks"])
        normalised_density = chunk_count / max_chunks
        intensity = acled_reference.get(key, 0.0)
        silence_score = round(intensity - normalised_density, 4)

        embs = cell_data["embeddings"]
        mean_emb = np.mean(embs, axis=0).tolist() if embs else []

        density_cells.append(
            DensityCell(
                geography=cell_data["geography"],
                time_window=cell_data["time_window"],
                document_count=len({m.get("source_filename") for m in cell_data["chunks"]}),
                chunk_count=chunk_count,
                mean_embedding=mean_emb,
                conflict_intensity=intensity,
                silence_score=silence_score,
            )
        )

    density_cells.sort(key=lambda c: c.silence_score, reverse=True)
    logger.info(
        f"Density-map silence detection: {len(density_cells)} cells, "
        f"{sum(1 for c in density_cells if c.silence_score > 0.3)} high-silence cells."
    )
    return density_cells


def _date_to_quarter(date_str: str) -> str:
    """Convert ISO date string to year-quarter label, e.g. '1992-07-15' → '1992-Q3'."""
    if not date_str or len(date_str) < 7:
        return "unknown"
    try:
        year, month = int(date_str[:4]), int(date_str[5:7])
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    except ValueError:
        return "unknown"
