# Ingestion

The ingestion layer loads source documents, chunks them, and stores them in ChromaDB with validated metadata.

## Supported formats

| Format | Extension | Loader |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| HTML | `.html`, `.htm` | `BSHTMLLoader` |
| Word | `.docx` | `Docx2txtLoader` |
| Plaintext | `.txt`, `.md` | `TextLoader` |
| URL | `http://`, `https://` | `WebBaseLoader` |

## CLI usage

```bash
# Single document
ai4saw ingest file report.pdf \
  --doc-type report \
  --language en \
  --geography Bosnia \
  --date-published 1996-01-01

# Whole directory (recursive)
ai4saw ingest corpus ./corpus \
  --doc-type report \
  --geography Sudan
```

## Chunking strategy

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 4000 chars (~1000 tokens) | Conflict documents require larger context windows than news |
| Overlap | 800 chars (~200 tokens) | Preserves cross-boundary entity mentions |
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph → sentence → word boundaries |

## Metadata

Every chunk carries validated `ChunkMetadata`:

```python
class ChunkMetadata(BaseModel):
    source_filename: str
    source_url: Optional[str]
    doc_type: Literal["report", "news", "legal", "grey_literature"]
    language: str           # ISO 639-1: "en", "bs", "ar"
    date_published: Optional[date]
    geography: Optional[str]
    chunk_index: int
```

Metadata is the primary filter surface for retrieval. Every field must be explicitly provided or `None` — there are no silent defaults. A `ChunkMetadata` validation error at ingestion time is better than silent null values corrupting silence detection or network analysis later.

## Idempotent ingestion

Chunk IDs are deterministic SHA-256 hashes of `(source_filename, chunk_index)`. Re-running ingestion on the same file overwrites existing chunks rather than creating duplicates. This means it's safe to re-ingest after updating metadata.

## Provenance tracking

All source documents must be registered in `corpus/sources.csv` before ingestion. This file is the licence and provenance register — it records where each document came from and whether it can be used for this purpose.

```csv
filename,title,source_url,doc_type,language,date_published,geography,licence,notes
```

See [Corpus Management](../corpus.md) for details.
