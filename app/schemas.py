from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RequirementMatch(BaseModel):
    requirement: str
    matched: bool
    score: float
    evidence: str = ""


class RewriteSuggestion(BaseModel):
    requirement: str
    original_bullet: str
    rewritten_bullet: str
    rationale: str


class ResumeAnalysis(BaseModel):
    resume_name: str
    match_score: int
    semantic_similarity: float
    keyword_coverage: float
    summary: str = ""
    audience_summary: str = ""
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    rewrite_suggestions: List[RewriteSuggestion] = Field(default_factory=list)
    hiring_manager_notes: List[str] = Field(default_factory=list)
    matched_requirements: List[RequirementMatch] = Field(default_factory=list)
    extracted_resume_text: str


class BatchAnalysisResponse(BaseModel):
    job_description: str
    analyzed_count: int
    top_resume: str = ""
    view_mode: str
    embedding_provider: str
    vector_backend: str
    parser_backend: str
    results: List[ResumeAnalysis]
