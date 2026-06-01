# Extraction

Three extraction tasks run against every chunk: NER, relation extraction, and event classification. All outputs are Pydantic v2 validated and saved to `output/`.

## Run all three

```bash
ai4saw extract pipeline --output output/ --delay 0.25
```

`--delay` adds a pause between LLM calls to avoid rate-limiting on cloud providers. Set to 0 for Ollama.

## Named Entity Recognition (NER)

**Method:** few-shot prompting via 8 domain-specific examples  
**Prompt:** `prompts/ner_few_shot.yaml`

Entity types extracted:

| Label | Examples |
|---|---|
| `PERSON` | Ratko Mladić, Zdravko Tolimir |
| `ORG` | ICTY, Human Rights Watch, Drina Corps |
| `LOCATION` | Srebrenica, El Geneina, West Darfur |
| `FACILITY` | Kravica warehouse, Manjača camp |
| `EVENT` | Operation Deliberate Force |
| `GROUP` | Masalit, Bosniak, Rohingya |
| `LEGAL_INSTRUMENT` | Rome Statute, UN Security Council Resolution 827 |

Output schema:

```python
class Entity(BaseModel):
    text: str
    label: Literal["PERSON", "ORG", "LOCATION", ...]
    confidence: float        # 0.0–1.0, model self-assessed
    span_start: Optional[int]
    span_end: Optional[int]
```

**Failure handling:** if JSON parse fails, retries once with an explicit format reminder. Persistent failures are logged to `logs/ai4saw.log` with the chunk ID.

## Relation Extraction

**Method:** chain-of-thought prompting — model reasons step-by-step before committing to a triple  
**Prompt:** `prompts/relations_cot.yaml`

The CoT approach reduces hallucinated relations compared to direct extraction. The model must identify actors, then actions, then locations and dates, before producing JSON.

Output schema:

```python
class Relation(BaseModel):
    subject: str
    predicate: str      # normalised verb phrase
    object: str
    location: Optional[str]
    date: Optional[str] # ISO 8601
    confidence: float
    evidence: str       # verbatim span supporting the triple
```

The `evidence` field is critical: it prevents hallucination from propagating silently. A triple without a grounded evidence span is flagged by the LLM-as-Judge.

## Event Classification

**Method:** zero-shot first pass; few-shot fallback when confidence < 0.6  
**Prompts:** `prompts/events_zero_shot.yaml`, `prompts/events_few_shot.yaml`

The zero-shot first strategy enables direct comparison of zero-shot vs few-shot performance — a publishable finding. The strategy used per chunk is logged.

Event taxonomy:

| Type | Description |
|---|---|
| `forced_labour` | Compelled work under threat of violence |
| `trafficking` | Recruitment/transport of persons for exploitation |
| `sexual_violence` | Rape, sexual slavery, or other sexual violence |
| `arbitrary_detention` | Imprisonment without legal basis |
| `displacement` | Forced movement of civilian populations |
| `killing` | Unlawful killing of civilians or prisoners |
| `siege` | Military encirclement targeting civilians |
| `no_event` | Chunk does not describe a reportable event |

## Entity Resolution

After running the extraction pipeline, merge entity aliases:

```bash
ai4saw extract resolve
```

This step is required before building the knowledge graph. See [Entity Resolution](../advanced/entity-resolution.md).
