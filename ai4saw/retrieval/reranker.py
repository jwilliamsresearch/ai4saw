"""Cross-encoder re-ranking wrapper.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score retrieved chunks
against the query and return the top N by relevance.

This is the single highest-impact quality improvement for RAG pipelines:
MMR retrieval reduces redundancy but a bi-encoder cannot compare query
and chunk jointly. A cross-encoder does — and its scores are calibrated
across the full relevance scale, not just cosine distance.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from loguru import logger

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class ScoredDocument:
    document: Document
    score: float


def rerank(
    query: str,
    documents: list[Document],
    top_n: int,
) -> list[Document]:
    """Re-rank documents against the query using a cross-encoder.

    Returns the top_n documents sorted by descending cross-encoder score.
    Falls back gracefully to the original ordering if the model cannot load
    (e.g. no internet in fully offline mode).
    """
    if not documents:
        return []

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(CROSS_ENCODER_MODEL)
        pairs = [(query, doc.page_content) for doc in documents]
        scores: list[float] = model.predict(pairs).tolist()

        scored = sorted(
            [ScoredDocument(doc, score) for doc, score in zip(documents, scores)],
            key=lambda x: x.score,
            reverse=True,
        )
        top = [s.document for s in scored[:top_n]]
        logger.debug(
            f"Re-ranked {len(documents)} chunks → top {top_n} "
            f"(scores: {[f'{s.score:.3f}' for s in scored[:top_n]]})"
        )
        return top

    except Exception as exc:
        logger.warning(
            f"Cross-encoder re-ranking failed ({exc}); "
            f"falling back to original retrieval order."
        )
        return documents[:top_n]
