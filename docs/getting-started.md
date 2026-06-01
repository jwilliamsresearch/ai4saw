# Getting Started

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11–3.13 | 3.12 recommended (pinned in `.python-version`) |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager — `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Ollama](https://ollama.ai/) | Free local LLM inference — download from ollama.ai |
| Git | — |

## Installation

```bash
git clone https://github.com/jwilliamsresearch/ai4saw
cd ai4saw
uv sync
cp .env.example .env
```

## Pull local models

Ollama is the default provider — no API key needed.

```bash
ollama pull mistral           # extraction LLM
ollama pull nomic-embed-text  # embeddings
```

## Configure (optional)

The defaults work out of the box with Ollama. To use a cloud provider:

```bash title=".env"
PROVIDER=openrouter
OPENROUTER_API_KEY=sk-...
DEFAULT_MODEL=mistralai/mistral-7b-instruct
```

See [Providers](providers.md) for full configuration reference.

## Your first run

### 1. Add a document to the corpus

```bash
# Register it in the provenance CSV first (licence required)
echo "myreport.pdf,My Report,,report,en,1996-01-01,Bosnia,public_domain," >> corpus/sources.csv

# Then ingest
ai4saw ingest file corpus/myreport.pdf \
  --doc-type report \
  --language en \
  --geography Bosnia \
  --date-published 1996-01-01
```

### 2. Run extraction

```bash
ai4saw extract pipeline          # NER + relations + events
ai4saw extract resolve           # merge entity aliases
```

### 3. Build the knowledge graph

```bash
ai4saw graph build
```

### 4. Query

```bash
# Standard RAG Q&A
ai4saw query ask "What forms of forced labour were documented?"

# Graph-augmented query (finds specific actor relations)
ai4saw graph query "What did the Drina Corps do in 1995?"

# Multi-hop agent (complex temporal/causal questions)
ai4saw graph agent "What happened before the Srebrenica massacre and who commanded those forces?"
```

### 5. Check status

```bash
ai4saw info
```

This shows your configuration and which data artefacts have been built.

## Optional extras

Cross-encoder reranking and RAGAS evaluation require PyTorch, which currently only has wheels for ARM Mac and Linux (not Intel Mac).

```bash
uv sync --extra rerank   # cross-encoder reranking (ARM/Linux only)
uv sync --extra eval     # RAGAS evaluation (ARM/Linux only)
uv sync --extra docs     # MkDocs documentation build
```

## MCP Server (Claude Desktop)

To use AI4SAW as a tool inside Claude Desktop:

1. Copy `mcp_config.example.json` and edit the `cwd` path
2. Add the `ai4saw` block to your Claude Desktop config:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Restart Claude Desktop

See [MCP Server](advanced/mcp.md) for full setup instructions.

## LLM-as-Judge (optional)

To enable automated quality scoring with a frontier model:

```bash
# In .env:
JUDGE_MODEL=anthropic/claude-3-5-sonnet   # via OpenRouter

ai4saw eval judge --sample 20
```

See [LLM-as-Judge](advanced/llm-judge.md).
