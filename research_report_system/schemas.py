"""Pydantic v2 schemas shared across the research & report pipeline."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Subtask(BaseModel):
    id: str
    description: str
    route: Literal["library", "web", "arxiv", "wikipedia"]
    status: Literal["pending", "approved", "running", "done"] = "pending"


class ResearchPlan(BaseModel):
    topic: str
    subtasks: list[Subtask] = Field(min_length=4, max_length=6)
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def _check_subtask_count(self) -> "ResearchPlan":
        if not 4 <= len(self.subtasks) <= 6:
            raise ValueError("a plan must have between 4 and 6 subtasks")
        return self


class Source(BaseModel):
    source_id: str
    url: HttpUrl
    title: str
    fetched_date: date
    source_type: Literal["web", "paper", "wiki", "library"]
    credibility_tier: Literal["high", "medium", "low"]
    topic: str


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class Claim(BaseModel):
    claim_id: str
    text: str
    quote: str
    chunk_id: str
    strength: Literal["supported", "single-source", "contradicted", "unsupported"]
    cross_check_source_id: str | None = None

    @model_validator(mode="after")
    def _supported_needs_cross_check(self) -> "Claim":
        if self.strength == "supported" and not self.cross_check_source_id:
            raise ValueError("strength='supported' requires cross_check_source_id")
        return self


class Insight(BaseModel):
    insight_id: str
    source_id: str
    summary: str
    claims: list[Claim]


class ReportDraft(BaseModel):
    title: str
    sections: list[str]
    claims: list[Claim]
    created_at: datetime = Field(default_factory=datetime.now)


class GateDecision(BaseModel):
    gate: Literal["plan", "report"]
    decision: Literal["approved", "edited", "rejected"]
    comment: str = ""
