# Providers

AI4SAW is model-agnostic. All LLM and embedding calls route through factory functions in `ai4saw/core/providers.py`. Switching provider requires only a `.env` change — no code changes.

## Available providers

| Provider | LLM | Embeddings | Cost | Best for |
|---|---|---|---|---|
| `ollama` | ✓ | ✓ | Free | Local dev, bulk runs, offline |
| `openrouter` | ✓ | — (falls back to Ollama) | Pay per token | Cloud eval, frontier models |
| `huggingface` | ✓ | ✓ | Free tier | Serverless, no GPU |

## Ollama (default)

```bash title=".env"
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

Pull models before running:

```bash
ollama pull mistral            # 4.1 GB — good extraction quality
ollama pull llama3             # 4.7 GB — stronger CoT reasoning
ollama pull nomic-embed-text   # 274 MB — embeddings
```

Other models to try:

| Model | Pull command | Notes |
|---|---|---|
| Mistral 7B | `ollama pull mistral` | Default, fast, good JSON compliance |
| Llama 3 8B | `ollama pull llama3` | Better chain-of-thought, slightly slower |
| Phi-3 Mini | `ollama pull phi3` | Very fast, lower quality |
| Mixtral 8x7B | `ollama pull mixtral` | Best quality, requires ~26 GB RAM |

## OpenRouter

OpenRouter provides access to frontier models (Claude, GPT-4o, Gemini) and open models via a single OpenAI-compatible API.

```bash title=".env"
PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=mistralai/mistral-7b-instruct
```

Recommended model IDs:

| Use case | Model ID |
|---|---|
| Bulk extraction (cheapest) | `mistralai/mistral-7b-instruct` |
| Better quality | `meta-llama/llama-3-8b-instruct` |
| Best quality (eval) | `anthropic/claude-3-5-sonnet` |
| LLM-as-Judge | `anthropic/claude-3-5-sonnet` |

!!! note "Embeddings with OpenRouter"
    OpenRouter does not provide an embeddings API. When `PROVIDER=openrouter`,
    embeddings automatically fall back to `nomic-embed-text` via Ollama.
    Ensure Ollama is running locally.

## HuggingFace Inference API

```bash title=".env"
PROVIDER=huggingface
HF_API_KEY=hf_...
DEFAULT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
EMBEDDING_MODEL=nomic-embed-text
```

The free tier has rate limits. For bulk runs, use Ollama locally.

## LLM-as-Judge configuration

The judge uses a separate model from the extraction LLM, configured independently:

```bash title=".env"
# Extraction (bulk, cheap)
PROVIDER=ollama
DEFAULT_MODEL=mistral

# Judge (frontier, small sample)
JUDGE_MODEL=anthropic/claude-3-5-sonnet   # must be reachable via PROVIDER's API
```

If `JUDGE_MODEL` is unset, the judge uses `DEFAULT_MODEL` (self-judging — introduces bias).

## Cost control

```bash title=".env"
MAX_TOKENS_PER_RUN=500000   # hard limit per run (approximate)
```

!!! tip "Cost strategy"
    - Ollama for all bulk/iterative work (free)
    - OpenRouter for evaluation runs and demos (small token spend)
    - Set `JUDGE_MODEL` to a frontier model only for final benchmarks
