"""Dataset export layer — produces flat, interoperable JSON/GeoJSON outputs.

All outputs are designed to be ingested directly into your-dataset, conflict-events-dataset
supplementary datasets, or a your-GIS-system database. No proprietary formats.

Output files (per spec §5.7):
  events.geojson   — classified events with geometry from LOCATION entities
  relations.json   — full triple store
  entities.json    — deduplicated entity registry
  silences.json    — ranked silence candidates with scores
  corpus_stats.json — document count, chunk count, coverage by geography/date
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import geojson
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import (
    CorpusStats,
    Entity,
    EventResult,
    EventType,
    NERResult,
    RelationResult,
    SilenceCandidate,
)

# Deferred import — geocoding is resolved at Phase 5 (spec §11)
try:
    from ai4saw.synthesis.geocoder import geocode_location
    _GEOCODING_AVAILABLE = True
except ImportError:
    _GEOCODING_AVAILABLE = False


def _output_path(filename: str) -> Path:
    path = settings.output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_events(
    events: list[EventResult],
    ner_results: list[NERResult],
) -> Path:
    """Export classified events as GeoJSON.

    Location entities from NER are geocoded (if geocoder available) and
    attached as GeoJSON Point geometry. Falls back to null geometry if
    geocoding is unavailable or fails.
    """
    # Build a lookup from chunk_id to LOCATION entities
    location_lookup: dict[str, list[str]] = defaultdict(list)
    for ner in ner_results:
        for entity in ner.entities:
            if entity.label == "LOCATION":
                location_lookup[ner.source_chunk_id].append(entity.text)

    features = []
    for event in events:
        if event.event_type == EventType.NO_EVENT:
            continue

        locations = location_lookup.get(event.source_chunk_id, [])
        primary_location = event.location or (locations[0] if locations else None)

        geometry = None
        if primary_location and _GEOCODING_AVAILABLE:
            coords = geocode_location(primary_location)
            if coords:
                geometry = geojson.Point((coords["lon"], coords["lat"]))

        properties = {
            "event_type": event.event_type.value,
            "confidence": event.confidence,
            "date": event.date,
            "location": primary_location,
            "perpetrator": event.perpetrator,
            "victim_group": event.victim_group,
            "source_chunk_id": event.source_chunk_id,
        }

        features.append(geojson.Feature(geometry=geometry, properties=properties))

    collection = geojson.FeatureCollection(features)
    out = _output_path("events.geojson")
    with open(out, "w", encoding="utf-8") as f:
        geojson.dump(collection, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(features)} events → {out}")
    return out


def export_relations(results: list[RelationResult]) -> Path:
    """Export all subject–predicate–object triples as JSON."""
    records = []
    for result in results:
        for rel in result.relations:
            records.append({
                "subject": rel.subject,
                "predicate": rel.predicate,
                "object": rel.object,
                "location": rel.location,
                "date": rel.date,
                "confidence": rel.confidence,
                "evidence": rel.evidence,
                "source_chunk_id": result.source_chunk_id,
            })

    out = _output_path("relations.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(records)} relations → {out}")
    return out


def export_entities(results: list[NERResult]) -> Path:
    """Export a deduplicated entity registry as JSON.

    Deduplication is case-insensitive on entity text within each label.
    Mean confidence across all occurrences is recorded.
    """
    registry: dict[str, dict[str, Any]] = {}

    for result in results:
        for entity in result.entities:
            key = f"{entity.label}::{entity.text.lower().strip()}"
            if key not in registry:
                registry[key] = {
                    "text": entity.text,
                    "label": entity.label,
                    "occurrences": 0,
                    "confidence_sum": 0.0,
                    "source_chunks": [],
                }
            registry[key]["occurrences"] += 1
            registry[key]["confidence_sum"] += entity.confidence
            if result.source_chunk_id not in registry[key]["source_chunks"]:
                registry[key]["source_chunks"].append(result.source_chunk_id)

    records = []
    for entry in registry.values():
        records.append({
            "text": entry["text"],
            "label": entry["label"],
            "occurrences": entry["occurrences"],
            "mean_confidence": round(entry["confidence_sum"] / entry["occurrences"], 4),
            "source_chunks": entry["source_chunks"],
        })
    records.sort(key=lambda r: r["occurrences"], reverse=True)

    out = _output_path("entities.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(records)} deduplicated entities → {out}")
    return out


def export_silences(candidates: list[SilenceCandidate]) -> Path:
    """Export ranked silence candidates as JSON."""
    records = [c.model_dump() for c in candidates]

    out = _output_path("silences.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(records)} silence candidates → {out}")
    return out


def export_corpus_stats(chunk_metadatas: list[dict]) -> Path:
    """Compute and export corpus coverage statistics."""
    geo_counter: Counter = Counter()
    date_counter: Counter = Counter()
    type_counter: Counter = Counter()
    lang_counter: Counter = Counter()
    source_files: set = set()

    for meta in chunk_metadatas:
        source_files.add(meta.get("source_filename", ""))
        geo = meta.get("geography") or "unknown"
        geo_counter[geo] += 1
        date_str = meta.get("date_published") or "unknown"
        year = date_str[:4] if date_str and date_str != "unknown" else "unknown"
        date_counter[year] += 1
        type_counter[meta.get("doc_type", "unknown")] += 1
        lang_counter[meta.get("language", "unknown")] += 1

    stats = CorpusStats(
        document_count=len(source_files),
        chunk_count=len(chunk_metadatas),
        coverage_by_geography=dict(geo_counter.most_common()),
        coverage_by_date=dict(sorted(date_counter.items())),
        doc_types=dict(type_counter.most_common()),
        languages=dict(lang_counter.most_common()),
    )

    out = _output_path("corpus_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"Corpus stats → {out}")
    return out


def export_all(
    events: list[EventResult],
    ner_results: list[NERResult],
    relation_results: list[RelationResult],
    silence_candidates: list[SilenceCandidate],
    chunk_metadatas: list[dict],
) -> dict[str, Path]:
    """Run all exports and return a mapping of output name → path."""
    return {
        "events": export_events(events, ner_results),
        "relations": export_relations(relation_results),
        "entities": export_entities(ner_results),
        "silences": export_silences(silence_candidates),
        "corpus_stats": export_corpus_stats(chunk_metadatas),
    }
