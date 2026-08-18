"""
Data contracts shared across all pipeline phases.

Using pydantic models (instead of loose dicts) gives us:
  * validation at every phase boundary — a malformed LLM output fails loudly
    instead of silently corrupting the dataset;
  * a single source of truth for the final .xlsx columns;
  * the schema we hand to the LLM for *structured generation* (Innovation #1),
    which forces the model to return the exact supporting span it used.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """A single collected source document (Phase 1 output)."""
    id: str
    url: str
    license: str
    type: str
    date_accessed: str
    n_chars: int
    text_path: str  # where the raw text is stored on disk


class Chunk(BaseModel):
    """A semantically coherent passage (Phase 2 output)."""
    chunk_id: str
    source_id: str
    source_url: str
    license: str
    topic_tag: str
    text: str
    n_tokens: int


class QACandidate(BaseModel):
    """
    Structured output the GENERATOR model must return per passage.

    `supporting_quote` is the crux of the groundedness guarantee: the model is
    required to copy the exact sentence(s) from the passage that justify the
    answer. Phase 4 verifies this span really occurs in the source text.
    """
    question: str = Field(..., description="A procurement/SC question the passage answers.")
    answer: str = Field(..., description="Answer grounded ONLY in the passage.")
    supporting_quote: str = Field(..., description="Exact span from the passage that supports the answer.")
    question_type: str = Field(..., description="factual | analytical | scenario")
    difficulty: str = Field(..., description="easy | medium | hard")


class QAPair(BaseModel):
    """A fully validated pair (Phase 4 output → Phase 5 xlsx row)."""
    id: str
    question: str
    answer: str
    source_passage: str
    source_url: str
    topic_tag: str
    question_type: str
    difficulty: str
    groundedness_score: float
    relevance_score: float
    judge_quality: int
    supporting_quote: str
