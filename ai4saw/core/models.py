"""All Pydantic v2 schemas used across the pipeline.

Single source of truth — no schema definitions should live in individual modules.
"""

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    source_filename: str
    source_url: Optional[str] = None
    doc_type: Literal["report", "news", "legal", "grey_literature"]
    language: str = Field(..., description="ISO 639-1 language code, e.g. 'en', 'bs', 'ar'")
    date_published: Optional[date] = None
    geography: Optional[str] = None
    chunk_index: int


# ── NER ───────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    text: str
    label: Literal[
        "PERSON", "ORG", "LOCATION", "FACILITY",
        "EVENT", "GROUP", "LEGAL_INSTRUMENT"
    ]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model self-assessed confidence")
    span_start: Optional[int] = None
    span_end: Optional[int] = None


class NERResult(BaseModel):
    entities: list[Entity]
    source_chunk_id: str


# ── Relation Extraction ───────────────────────────────────────────────────────

class Relation(BaseModel):
    subject: str
    predicate: str = Field(..., description="Normalised verb phrase")
    object: str
    location: Optional[str] = None
    date: Optional[str] = Field(None, description="ISO 8601 date string where inferrable")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(..., description="Verbatim span from source text supporting this triple")


class RelationResult(BaseModel):
    relations: list[Relation]
    source_chunk_id: str


# ── Event Classification ──────────────────────────────────────────────────────

class EventType(str, Enum):
    FORCED_LABOUR = "forced_labour"
    TRAFFICKING = "trafficking"
    SEXUAL_VIOLENCE = "sexual_violence"
    ARBITRARY_DETENTION = "arbitrary_detention"
    DISPLACEMENT = "displacement"
    KILLING = "killing"
    SIEGE = "siege"
    NO_EVENT = "no_event"


class EventResult(BaseModel):
    event_type: EventType
    confidence: float = Field(..., ge=0.0, le=1.0)
    date: Optional[str] = None
    location: Optional[str] = None
    perpetrator: Optional[str] = None
    victim_group: Optional[str] = None
    source_chunk_id: str


# ── RAG Q&A ───────────────────────────────────────────────────────────────────

class QAResponse(BaseModel):
    answer: str
    sources: list[ChunkMetadata]
    confidence: float = Field(..., ge=0.0, le=1.0)
    retrieved_chunks: int
    reranked_to: int


# ── Silence Detection ─────────────────────────────────────────────────────────

class SilenceCandidate(BaseModel):
    event_id: str = Field(..., description="CDISaW / ACLED reference ID")
    location: str
    date: str
    conflict_intensity: float = Field(..., description="Conflict intensity score from source dataset")
    retrieval_confidence: float = Field(..., description="Mean similarity of top-3 retrieved chunks")
    silence_score: float = Field(..., description="conflict_intensity minus retrieval_confidence")
    candidate_reason: str


# ── Corpus stats ──────────────────────────────────────────────────────────────

class CorpusStats(BaseModel):
    document_count: int
    chunk_count: int
    coverage_by_geography: dict[str, int]
    coverage_by_date: dict[str, int]
    doc_types: dict[str, int]
    languages: dict[str, int]


# ── Entity Resolution (Feature 1) ─────────────────────────────────────────────

class ResolvedEntity(BaseModel):
    """A canonical entity combining all textual variants across the corpus.

    Entity resolution is critical for conflict research: "the RSF", "Rapid Support
    Forces", and "the paramilitaries" refer to the same actor. Without resolution,
    network analysis and silence detection fragment across aliases.
    """
    canonical_id: str = Field(..., description="SHA-256 of canonical_text + label (first 12 chars)")
    canonical_text: str = Field(..., description="Most frequently occurring form")
    label: str
    aliases: list[str] = Field(default_factory=list, description="All other surface forms seen")
    occurrence_count: int
    source_chunks: list[str]
    mean_confidence: float = Field(..., ge=0.0, le=1.0)


class EntityResolutionResult(BaseModel):
    entities: list[ResolvedEntity]
    total_mentions: int
    unique_texts_before: int
    resolved_count: int = Field(..., description="Canonical entities after merging")
    built_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Knowledge Graph (Feature 2) ───────────────────────────────────────────────

class KnowledgeGraphNode(BaseModel):
    """A node in the knowledge graph, corresponding to a resolved entity."""
    id: str
    text: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    occurrence_count: int = 1


class KnowledgeGraphEdge(BaseModel):
    """A directed edge representing a verified relation between two entities.

    Temporal fields enable time-filtered graph queries: `graph query --at 1995-07-01`
    returns only the command structure that existed on that date. valid_from is
    populated from the relation's date field during graph construction. valid_to
    is None (open-ended) unless a termination date is explicitly known.
    """
    source_id: str
    target_id: str
    predicate: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str
    date: Optional[str] = None
    location: Optional[str] = None
    source_chunk_id: str
    # Temporal validity window — ISO 8601 date strings
    valid_from: Optional[str] = Field(None, description="Date from which this relation holds")
    valid_to: Optional[str] = Field(None, description="Date after which this relation no longer holds (None = open-ended)")


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode]
    edges: list[KnowledgeGraphEdge]
    built_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    node_count: int = 0
    edge_count: int = 0


# ── Contradiction Detection (Feature 3) ───────────────────────────────────────

class ContradictionType(str, Enum):
    FACTUAL = "factual"          # different facts about the same event
    TEMPORAL = "temporal"        # different dates for the same event
    ATTRIBUTION = "attribution"  # different perpetrators or commanders
    NUMERICAL = "numerical"      # different numbers (casualties, victims, duration)


class ContradictionPair(BaseModel):
    """Two source chunks making incompatible claims about the same subject.

    Contradictions are not errors — in conflict research they often signal
    contested narratives, propaganda, or different witness perspectives.
    The pipeline surfaces them; researchers interpret them.
    """
    chunk_id_a: str
    chunk_id_b: str
    source_a: str = Field(..., description="Filename of first source")
    source_b: str = Field(..., description="Filename of second source")
    claim_a: str = Field(..., description="Relevant extract from source A")
    claim_b: str = Field(..., description="Relevant extract from source B")
    contradiction_type: ContradictionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Why this is a contradiction and what it implies")


class ContradictionReport(BaseModel):
    pairs: list[ContradictionPair]
    total_chunks_analysed: int
    candidate_pairs_assessed: int
    high_confidence_count: int = Field(
        ..., description="Pairs with confidence >= 0.7"
    )


# ── Multi-hop Agent (Feature 4) ───────────────────────────────────────────────

class AgentStep(BaseModel):
    """A single reasoning step in the multi-hop agent's chain."""
    sub_question: str
    tool_used: str
    result_summary: str


class AgentResponse(BaseModel):
    """Full response from the multi-hop reasoning agent.

    Includes the decomposed reasoning chain so researchers can audit
    which sources and graph nodes contributed to the answer.
    """
    question: str
    steps: list[AgentStep]
    answer: str
    sources_consulted: list[str]
    graph_nodes_consulted: list[str]
    iterations: int


# ── Active Corpus Discovery (Feature 5) ───────────────────────────────────────

class DiscoveredDocument(BaseModel):
    """A document discovered via external API search that is not yet in the corpus."""
    title: str
    url: str
    source: Literal["reliefweb", "gdelt", "manual"]
    date: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    trigger_entity: str = Field(..., description="Entity that triggered this discovery")
    snippet: Optional[str] = None


class DiscoveryResult(BaseModel):
    trigger_entities: list[str]
    documents: list[DiscoveredDocument]
    query_count: int
    new_documents: int = Field(..., description="Documents not already in the corpus by URL")


# ── Perpetrator Command Network (Feature 6) ───────────────────────────────────

class NetworkNode(BaseModel):
    """A node in the perpetrator command network with graph-theoretic metrics.

    Betweenness centrality identifies actors who control information flow
    between otherwise disconnected parts of the command structure — typically
    mid-level commanders who are critical to attribution chains.
    """
    id: str
    label: str
    entity_type: str
    betweenness_centrality: float = Field(..., ge=0.0, description="Normalised 0–1")
    in_degree: int = Field(..., description="Number of incoming command relations")
    out_degree: int = Field(..., description="Number of outgoing command relations")
    community_id: int


class NetworkEdge(BaseModel):
    source: str
    target: str
    predicate: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str
    is_command_relation: bool = Field(
        ..., description="True if predicate is a command/order verb"
    )


class NetworkAnalysis(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    communities: dict[str, list[str]] = Field(
        ..., description="community_id -> list of node labels"
    )
    key_actors: list[str] = Field(
        ..., description="Top nodes by betweenness centrality"
    )
    total_nodes: int
    total_edges: int
    command_edges: int
    built_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── LLM-as-Judge Evaluation ───────────────────────────────────────────────────

class JudgeScore(BaseModel):
    """Quality assessment for a single chunk's extraction results.

    Scored by a frontier model acting as judge. Three dimensions are assessed
    independently so researchers can identify which extraction task degrades
    on their specific corpus — which may differ from the general benchmark.
    """
    chunk_id: str
    source_filename: str
    ner_score: float = Field(..., ge=0.0, le=1.0, description="Entity accuracy (0=wrong, 1=perfect)")
    relation_score: float = Field(..., ge=0.0, le=1.0, description="Triple groundedness in source text")
    event_score: float = Field(..., ge=0.0, le=1.0, description="Event type + field correctness")
    overall_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    explanation: str


class JudgeReport(BaseModel):
    """Aggregate LLM-as-Judge quality report across a sample of chunks."""
    scores: list[JudgeScore]
    aggregate: dict[str, float] = Field(
        ..., description="Mean scores: ner, relation, event, overall"
    )
    sample_size: int
    model_used: str
    built_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
