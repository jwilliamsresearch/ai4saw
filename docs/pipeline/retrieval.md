# Retrieval

Two retrieval modes are available: standard RAG Q&A and graph-augmented retrieval. For complex questions, the multi-hop agent combines both.

## Standard RAG Q&A

```bash
ai4saw query ask "What forms of forced labour were documented in Bosnian camps?"
```

**Pipeline:**

1. **MMR retrieval** — retrieve top-K chunks using Maximal Marginal Relevance. MMR balances relevance with diversity, preventing retrieval of 8 near-identical passages.
2. **Cross-encoder reranking** — rerank K chunks to N using `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is the highest-impact quality improvement: the bi-encoder in step 1 scores queries and documents independently; the cross-encoder scores them jointly and is better calibrated.
3. **Cited answer generation** — stuff the N chunks into the context window and generate an answer with `[Source N]` inline citations.

**Why reranking matters:** MMR retrieval alone returns semantically similar but sometimes off-topic chunks. The cross-encoder reranker has seen the query and the document together during training and can reject passages that are topically close but factually irrelevant.

Cross-encoder reranking requires `--extra rerank` (PyTorch). Without it, the reranker falls back to the original retrieval order silently.

## Configuration

```bash
ai4saw query ask "..." --top-k 8 --top-n 3
```

| Parameter | Default | Effect |
|---|---|---|
| `--top-k` | 8 | Chunks retrieved by MMR |
| `--top-n` | 3 | Chunks passed to LLM after reranking |

Set in `.env` for permanent defaults:
```bash
RETRIEVAL_TOP_K=8
RERANK_TOP_N=3
```

## GraphRAG

For questions about specific actors or relations, use GraphRAG:

```bash
ai4saw graph query "What did the Drina Corps do in Location Alpha in 1995?"
```

This retrieves the entity's neighbourhood from the knowledge graph and combines it with vector search. See [GraphRAG](../advanced/graph-rag.md).

## Multi-hop Agent

For complex temporal or causal questions:

```bash
ai4saw graph agent "What was happening in the months before Location Alpha?"
```

The agent chains multiple tool calls until it has enough context. See [Multi-hop Agent](../advanced/agent.md).
