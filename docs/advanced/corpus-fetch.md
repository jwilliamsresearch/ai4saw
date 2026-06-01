# Automated Corpus Fetch

## What this does

`discover fetch` extends the passive discovery module into an active pipeline: it discovers candidate documents via ReliefWeb and GDELT, downloads them, registers them in `corpus/sources.csv`, and ingests them into ChromaDB — in a single command.

This automates **Phase 1** of the pipeline (corpus preparation) for sources that have open APIs.

## Usage

```bash
# Discover and fetch documents about specific entities
ai4saw discover fetch "Location Alpha" "Commander Alpha" "Armed Group Alpha" \
  --geography conflict-region \
  --max-docs 20 \
  --min-relevance 0.5

# Fetch documents for silence candidates (gaps detected by silence detector)
ai4saw discover fetch "Location C" "Location D" \
  --geography conflict-region \
  --silence-mode \
  --max-docs 15

# Preview candidates without downloading anything
ai4saw discover fetch "Location Beta" "Group Beta" \
  --geography conflict-region \
  --dry-run

# Skip review prompt — for scripted/automated runs
ai4saw discover fetch "Armed-Group-Beta" "Province Beta" \
  --geography conflict-region \
  --yes
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--geography` | required | Geography tag written to chunk metadata and sources.csv |
| `--max-docs` | 20 | Hard cap on documents to ingest in one run |
| `--min-relevance` | 0.5 | Minimum relevance score (0–1) to include a candidate |
| `--per-entity-limit` | 10 | Max API results per entity per source |
| `--silence-mode` | off | Treat entities as silence candidates (higher per-entity limit) |
| `--dry-run` | off | Show candidates only — download nothing |
| `--yes` | off | Skip confirmation prompt — ingest all above threshold |

## What gets registered

For each successfully ingested document, a row is appended to `corpus/sources.csv`:

```csv
filename,source_url,date_accessed,licence,geography,notes
sudan_rsf_masalit_abc12345.pdf,https://reliefweb.int/...,2026-06-01,public,Sudan,auto-fetched from reliefweb | conflict-region: Armed-Group-Beta Attacks on Group Beta
```

Licence is inferred automatically: `public` for ReliefWeb, `news` for GDELT.

## Sources supported

| Source | Format | Licence tag | Notes |
|---|---|---|---|
| **OpenAlex** | PDF + HTML | `open-access` | Scholarly papers; 25 per entity; free, no auth |
| **Semantic Scholar** | PDF | `open-access` | Distinct academic corpus; 25 per entity |
| **arXiv** | PDF | `open-access` | Preprints; 25 per entity |
| **Internet Archive** | PDF + HTML | `open-access` | NGO/grey literature; highest value for pre-2010 docs |
| ~~UN Digital Library~~ | — | — | Deferred — API requires authentication |
| **GDELT** | HTML (news) | `news` | 65+ languages; up to 250 per run (single batched query) |

> Set `CONTACT_EMAIL=your@email.com` in `.env` to use OpenAlex's polite pool.

> **GDELT rate limit:** Pass all entity names in one `discover fetch` call — the pipeline batches them into a single GDELT OR query to avoid IP bans.

### Deferred sources (need custom scrapers)

- **International Tribunal/IRMCT** — document database requires session-based scraping
- **ICC** — court records available via web but no public API
- **OHCHR / ReliefWeb** — require registered API keys
- **HRW reports** — HTML with inconsistent structure

## Pipeline behaviour

```
discover fetch → ReliefWeb + GDELT query
              → filter by min_relevance
              → HEAD request (detect PDF vs HTML)
              → PDF: download to corpus/ → load from file
              → HTML: load directly from URL
              → chunk_documents() → embed_and_store()
              → append row to corpus/sources.csv
```

Ingestion is **idempotent**: deterministic chunk IDs mean re-running the same fetch is safe — existing chunks are overwritten, not duplicated.

Already-registered URLs (in `corpus/sources.csv`) are skipped automatically.

## Recommended workflow

```bash
# 1. Run silence detection to find gaps (after graph build)
ai4saw export all   # includes silence output

# 2. Fetch targeted documents for the highest-priority silence locations
ai4saw discover fetch "Location F" "Location G" --geography conflict-region --silence-mode --yes

# 3. Re-run extraction on the new chunks
ai4saw extract pipeline

# 4. Rebuild the graph
ai4saw graph build
```

## Review mode (default)

Without `--yes`, the command shows a candidate table and asks for confirmation before downloading:

```
┌─────────────┬───────────┬──────────────┬──────────────────────────────────────────────────────────┐
│ Source      │ Relevance │ Date         │ Title                                                    │
├─────────────┼───────────┼──────────────┼──────────────────────────────────────────────────────────┤
│ reliefweb   │ 0.80      │ 2023-05-02   │ conflict-region: Armed-Group-Beta Attacks on Group Beta in Location Beta              │
│ reliefweb   │ 0.70      │ 2023-06-14   │ Province Beta: Humanitarian Situation Update                    │
│ gdelt       │ 0.55      │ 2023-04-28   │ Location Beta massacre: what we know                        │
└─────────────┴───────────┴──────────────┴──────────────────────────────────────────────────────────┘
Fetch 3 documents? [y/N]:
```
