"""RAG Q&A chain with MMR retrieval, cross-encoder re-ranking, and source citations.

Pipeline per spec §5.5:
  1. Embed query via get_embedder()
  2. Retrieve top-K chunks from ChromaDB using MMR (reduces redundancy)
  3. Re-rank with cross-encoder to top-N
  4. Stuff context + source metadata into prompt
  5. Generate answer with inline citations
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from ai4saw.core.config import settings
from ai4saw.core.models import ChunkMetadata, QAResponse
from ai4saw.core.providers import get_llm
from ai4saw.ingestion.embedder import get_vector_store
from ai4saw.retrieval.reranker import rerank

_SYSTEM_PROMPT = """You are an expert analyst answering questions about conflict,
human rights, and slavery using only the provided source documents.

Rules:
- Answer only from the context below. Do not hallucinate.
- Cite sources inline using [Source N] notation.
- If the context is insufficient to answer, say so explicitly.
- Be precise and concise. Analysts need facts, not filler.
"""

_USER_TEMPLATE = """\
Question: {question}

Context:
{context}

Answer (with inline [Source N] citations):"""


def _format_context(chunks: list, metadatas: list[dict]) -> str:
    parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), start=1):
        source_info = (
            f"{meta.get('source_filename', 'unknown')}"
            f"{' | ' + meta.get('geography', '') if meta.get('geography') else ''}"
            f"{' | ' + meta.get('date_published', '') if meta.get('date_published') else ''}"
        )
        parts.append(f"[Source {i}] ({source_info})\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def _meta_to_model(meta: dict) -> ChunkMetadata:
    date_published = meta.get("date_published")
    if date_published and isinstance(date_published, str) and date_published:
        try:
            date_published = date.fromisoformat(date_published)
        except ValueError:
            date_published = None

    return ChunkMetadata(
        source_filename=meta.get("source_filename", ""),
        source_url=meta.get("source_url") or None,
        doc_type=meta.get("doc_type", "report"),
        language=meta.get("language", "en"),
        date_published=date_published,
        geography=meta.get("geography") or None,
        chunk_index=int(meta.get("chunk_index", 0)),
    )


def answer(
    question: str,
    top_k: int | None = None,
    top_n: int | None = None,
) -> QAResponse:
    """Answer a natural language question over the indexed corpus.

    Args:
        question: The researcher's question.
        top_k: How many chunks to retrieve (default: settings.retrieval_top_k).
        top_n: How many to keep after re-ranking (default: settings.rerank_top_n).

    Returns:
        QAResponse with answer text, source metadata, and retrieval stats.
    """
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    store = get_vector_store()

    # MMR retrieval — balances relevance with diversity
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 2},
    )
    retrieved = retriever.invoke(question)
    logger.debug(f"Q&A retrieved {len(retrieved)} chunks via MMR for: {question!r}")

    reranked = rerank(question, retrieved, top_n=top_n)

    metadatas = [doc.metadata for doc in reranked]
    context = _format_context(reranked, metadatas)

    user_content = _USER_TEMPLATE.format(question=question, context=context)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    llm = get_llm()
    response = llm.invoke(messages)
    answer_text: str = response.content

    sources = []
    for meta in metadatas:
        try:
            sources.append(_meta_to_model(meta))
        except Exception as exc:
            logger.warning(f"Could not parse source metadata: {exc}")

    # Confidence heuristic: proportion of sources cited in the answer
    cited = sum(
        1 for i in range(1, len(sources) + 1)
        if f"[Source {i}]" in answer_text
    )
    confidence = cited / len(sources) if sources else 0.0

    return QAResponse(
        answer=answer_text,
        sources=sources,
        confidence=round(confidence, 3),
        retrieved_chunks=len(retrieved),
        reranked_to=len(reranked),
    )
