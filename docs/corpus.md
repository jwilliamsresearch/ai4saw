# Corpus Management

## Provenance register

Every document must be registered in `corpus/sources.csv` before ingestion. This file is the licence and provenance register — it records where each document came from and whether it can be used for this research purpose.

```csv
filename,title,source_url,doc_type,language,date_published,geography,licence,notes
```

| Field | Required | Notes |
|---|---|---|
| `filename` | Yes | Filename in `corpus/` directory |
| `title` | Yes | Human-readable title |
| `source_url` | Recommended | Original URL for deduplication in discovery |
| `doc_type` | Yes | `report`, `news`, `legal`, `grey_literature` |
| `language` | Yes | ISO 639-1 code: `en`, `bs`, `ar`, `fr` |
| `date_published` | Recommended | ISO 8601: `1995-07-11` |
| `geography` | Recommended | Geographic tag: `Bosnia`, `Sudan`, `Darfur` |
| `licence` | Yes | e.g. `CC-BY-4.0`, `public_domain`, `research_use` |
| `notes` | Optional | Anything relevant to provenance |

## Document types

| Type | Examples |
|---|---|
| `report` | NGO reports, UN assessments, government reports |
| `news` | Newspaper articles, wire service reports |
| `legal` | ICTY/ICC judgments, indictments, witness statements |
| `grey_literature` | Working papers, conference proceedings, internal docs |

## Supported file formats

PDF, HTML, DOCX, and plaintext. URLs are also supported directly.

## Licence requirements

All documents must be verified as usable for this research purpose before ingestion. Do not ingest:

- Copyrighted news content without a research/fair use basis
- Confidential documents
- Documents with explicit non-research-use restrictions

Maintain the `licence` field for every entry in `sources.csv`.

## Multilingual corpora

The pipeline supports multilingual documents. Set `--language` at ingestion to the ISO 639-1 code.

For Bosnian and Arabic sources, the embedding model (`nomic-embed-text`) has reasonable multilingual coverage. For extraction, Llama 3 performs better than Mistral on non-English text. Consider translation pre-processing for older Bosnian documents that use character sets not well-covered by modern LLMs.

## Corpus organisation tips

```
corpus/
├── sources.csv              # provenance register
├── bosnia/                  # optional subdirectory organisation
│   ├── icty-judgments/
│   └── ngo-reports/
└── sudan/
    ├── acled-context/
    └── hrw-reports/
```

Ingest by subdirectory with a shared geography tag:

```bash
ai4saw ingest corpus corpus/bosnia/ --geography Bosnia --doc-type report
ai4saw ingest corpus corpus/sudan/ --geography Sudan --doc-type report
```

## Checking coverage

```bash
ai4saw export all     # generates corpus_stats.json
ai4saw info           # shows artefact status
```

`output/corpus_stats.json` shows document count, chunk count, and coverage by geography, date, doc type, and language. Use this to identify thin coverage areas before running silence detection.
