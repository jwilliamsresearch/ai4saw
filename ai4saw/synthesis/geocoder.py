"""Geocoding stub — deferred to Phase 5 per spec §11.

Resolution options under evaluation:
  A) Nominatim (free, OSM-backed) — good for modern locations
  B) GeoNames — broader coverage, requires free API key
  C) Few-shot LLM geocoder — best for historical and obscure conflict locations

This module raises ImportError when imported until an approach is chosen,
so export.py degrades gracefully (null geometry) rather than crashing.
"""

raise ImportError(
    "Geocoding is not yet implemented — deferred to Phase 5. "
    "Export will produce null geometry in events.geojson. "
    "See spec §11 for the decision criteria."
)
