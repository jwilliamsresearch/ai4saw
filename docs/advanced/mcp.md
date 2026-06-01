# MCP Server

The MCP (Model Context Protocol) server exposes AI4SAW as a tool that Claude Desktop, Claude.ai, and other MCP-compatible clients can use directly.

A researcher can open Claude Desktop and ask:

> *"What happened in El Geneina in April 2023?"*

Claude will call `search_corpus` and `query_knowledge_graph`, then synthesise a cited answer grounded in your indexed documents — without the researcher needing to know any CLI commands.

## Setup

### 1. Install and ingest

```bash
# Ensure the pipeline has run
ai4saw ingest corpus ./corpus
ai4saw extract pipeline
ai4saw extract resolve
ai4saw graph build
```

### 2. Configure Claude Desktop

Copy the relevant block from `mcp_config.example.json` and add it to your Claude Desktop configuration file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ai4saw": {
      "command": "uv",
      "args": ["run", "ai4saw-mcp"],
      "cwd": "/absolute/path/to/ai4saw"
    }
  }
}
```

Replace `/absolute/path/to/ai4saw` with your actual project directory.

### 3. Restart Claude Desktop

The AI4SAW tools will appear in Claude's tool list.

## Available tools

| Tool | Description |
|---|---|
| `search_corpus` | Semantic search over indexed documents. Returns top passages with source metadata. |
| `query_knowledge_graph` | Structured graph traversal. Returns entity relations with evidence. Supports `at_date` for temporal queries. |
| `find_entity` | Resolve an entity name to its canonical form and aliases. |
| `ask_question` | Full RAG Q&A with MMR retrieval + reranking + LLM synthesis. |
| `get_corpus_stats` | Coverage overview: document count, geography, date range. |

## Test the server

```bash
# Run standalone (outputs to stdout for debugging)
uv run ai4saw-mcp
```

## Temporal queries via MCP

The `query_knowledge_graph` tool accepts an `at_date` parameter:

> *"Show me the command structure in Bosnia as of July 1995"*

Claude will call `query_knowledge_graph(entity_name="Bosnia command", at_date="1995-07-11")` and return only relations valid on that date.

## Server instructions

The server is initialised with a system-level instruction set that tells Claude:

- Use `search_corpus` for open-ended questions
- Use `query_knowledge_graph` for specific actor/event relations
- Use `find_entity` to resolve abbreviations
- Always cite sources
- Be precise about dates, locations, and actor names
- Say explicitly when evidence is insufficient
