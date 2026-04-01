from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from app.embeddings import BaseEmbeddingProvider
from app.schemas import RequirementMatch, ResumeAnalysis, RewriteSuggestion
from app.vectorstores import BaseVectorStore


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "you",
    "your",
    "will",
    "our",
    "their",
    "they",
    "we",
    "this",
    "that",
    "have",
    "has",
    "had",
    "into",
    "using",
    "use",
    "need",
    "needs",
    "required",
    "requirements",
    "looking",
    "seeking",
    "experience",
    "plus",
    "bonus",
    "must",
    "should",
    "role",
    "intern",
    "candidate",
    "ability",
    "familiarity",
    "knowledge",
    "exposure",
    "awareness",
    "potential",
    "strong",
    "clear",
}

KNOWN_SKILLS = [
    "python",
    "fastapi",
    "flask",
    "django",
    "react",
    "next.js",
    "streamlit",
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "transformers",
    "hugging face",
    "sentence transformers",
    "faiss",
    "chromadb",
    "pinecone",
    "rag",
    "retrieval",
    "vector database",
    "openai",
    "gemini",
    "javascript",
    "typescript",
    "node.js",
    "sql",
    "postgresql",
    "docker",
    "aws",
    "gcp",
    "azure",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "api",
    "microservices",
]

LEADERSHIP_TERMS = {"led", "mentored", "managed", "owner", "headed", "coordinated", "leadership"}
CLOUD_TERMS = {"aws", "azure", "gcp", "cloud", "s3", "lambda", "ec2", "vertex ai"}
SYSTEM_DESIGN_TERMS = {"system design", "scalability", "architecture", "distributed", "microservices", "high availability"}


@dataclass
class AnalysisFeatures:
    summary: str
    audience_summary: str
    matched_skills: List[str]
    missing_skills: List[str]
    requirement_matches: List[RequirementMatch]
    semantic_similarity: float
    keyword_coverage: float
    strengths: List[str]
    gaps: List[str]
    suggestions: List[str]
    rewrite_suggestions: List[RewriteSuggestion]
    hiring_manager_notes: List[str]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9.+#-]{1,}", text.lower())


def split_into_segments(text: str) -> List[str]:
    rough = re.split(r"\n{2,}|(?<=[.!?])\s+", text)
    segments = [segment.strip(" -\n\t") for segment in rough if len(segment.strip()) > 25]
    return segments[:30] or [text.strip()]


def extract_requirements(job_description: str) -> List[str]:
    lines = [line.strip(" -\t") for line in job_description.splitlines() if line.strip()]
    extracted: list[str] = []

    for line in lines:
        if len(line) < 10:
            continue
        parts = split_requirement_line(line)
        extracted.extend(parts or [line])

    if extracted:
        return dedupe_preserve_order(extracted)[:12]
    return split_into_segments(job_description)[:12]


def extract_skill_hits(text: str) -> List[str]:
    lowered = text.lower()
    return sorted({skill for skill in KNOWN_SKILLS if skill in lowered})


def compute_keyword_coverage(job_description: str, resume_text: str) -> tuple[float, List[str], List[str]]:
    jd_tokens = [token for token in tokenize(job_description) if token not in STOP_WORDS and len(token) > 2]
    resume_tokens = set(tokenize(resume_text))
    counts = Counter(jd_tokens)
    important = [token for token, _ in counts.most_common(20) if token not in STOP_WORDS]
    matched = [token for token in important if token in resume_tokens]
    missing = [token for token in important if token not in resume_tokens]
    coverage = len(matched) / max(len(important), 1)
    return coverage, matched[:8], missing[:8]


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0
    score = float(np.dot(vector_a, vector_b))
    return max(0.0, min(1.0, score))


def analyze_resume(
    resume_name: str,
    resume_text: str,
    job_description: str,
    embedding_provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    view_mode: str = "candidate",
) -> ResumeAnalysis:
    clean_resume = resume_text.strip()
    clean_jd = job_description.strip()
    resume_segments = split_into_segments(clean_resume)
    requirements = extract_requirements(clean_jd)

    features = analyze_features(
        resume_text=clean_resume,
        job_description=clean_jd,
        resume_segments=resume_segments,
        requirements=requirements,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        view_mode=view_mode,
    )

    skill_alignment = min(
        len(features.matched_skills)
        / max(len(features.matched_skills) + len(features.missing_skills), 1),
        1.0,
    )
    requirement_alignment = (
        sum(1 for item in features.requirement_matches if item.matched)
        / max(len(features.requirement_matches), 1)
    )
    evidence_alignment = average_requirement_score(features.requirement_matches)
    project_alignment = min(count_resume_projects(clean_resume) / 2, 1.0)
    match_score = int(
        round(
            (
                features.semantic_similarity * 0.10
                + requirement_alignment * 0.15
                + evidence_alignment * 0.20
                + features.keyword_coverage * 0.20
                + skill_alignment * 0.25
                + project_alignment * 0.10
            )
            * 100
        )
    )
    match_score = max(0, min(match_score, 100))

    return ResumeAnalysis(
        resume_name=resume_name,
        match_score=match_score,
        semantic_similarity=round(features.semantic_similarity, 3),
        keyword_coverage=round(features.keyword_coverage, 3),
        summary=features.summary,
        audience_summary=features.audience_summary,
        matched_skills=features.matched_skills,
        missing_skills=features.missing_skills,
        strengths=features.strengths,
        gaps=features.gaps,
        suggestions=features.suggestions,
        rewrite_suggestions=features.rewrite_suggestions,
        hiring_manager_notes=features.hiring_manager_notes,
        matched_requirements=features.requirement_matches,
        extracted_resume_text=clean_resume[:4000],
    )


def analyze_features(
    resume_text: str,
    job_description: str,
    resume_segments: Sequence[str],
    requirements: Sequence[str],
    embedding_provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    view_mode: str,
) -> AnalysisFeatures:
    document_embeddings = embedding_provider.embed_texts([job_description, resume_text])
    jd_vector = document_embeddings[0]
    resume_vector = document_embeddings[1]
    semantic_similarity = cosine_similarity(jd_vector, resume_vector)

    segment_embeddings = embedding_provider.embed_texts(list(resume_segments))
    requirement_embeddings = embedding_provider.embed_texts(list(requirements))
    search_results = vector_store.search(segment_embeddings, requirement_embeddings)

    requirement_matches: List[RequirementMatch] = []
    for requirement, search_result in zip(requirements, search_results):
        semantic_score = float(search_result.score)
        evidence = resume_segments[int(search_result.index)] if len(resume_segments) else ""
        lexical_score = token_overlap_score(requirement, evidence or resume_text)
        score = max(semantic_score, lexical_score * 0.75 + semantic_score * 0.25)
        requirement_matches.append(
            RequirementMatch(
                requirement=requirement,
                matched=score >= 0.33,
                score=round(max(0.0, min(score, 1.0)), 3),
                evidence=evidence[:220],
            )
        )

    keyword_coverage, matched_keywords, missing_keywords = compute_keyword_coverage(
        job_description, resume_text
    )
    matched_skills = extract_skill_hits(resume_text)
    jd_skills = extract_skill_hits(job_description)
    missing_skills = [skill for skill in jd_skills if skill not in matched_skills]
    resume_bullets = extract_resume_bullets(resume_text)

    strengths = build_strengths(matched_skills, requirement_matches, matched_keywords)
    gaps = build_gaps(missing_skills, requirement_matches, missing_keywords)
    suggestions = build_suggestions(missing_skills, requirement_matches)
    rewrite_suggestions = build_rewrite_suggestions(
        resume_bullets=resume_bullets,
        requirement_matches=requirement_matches,
        missing_skills=missing_skills,
    )
    hiring_manager_notes = build_hiring_manager_notes(
        requirement_matches=requirement_matches,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_text=resume_text,
    )
    summary = build_summary(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        requirement_matches=requirement_matches,
        resume_text=resume_text,
    )
    audience_summary = build_audience_summary(
        view_mode=view_mode,
        summary=summary,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        hiring_manager_notes=hiring_manager_notes,
        rewrite_suggestions=rewrite_suggestions,
    )

    return AnalysisFeatures(
        summary=summary,
        audience_summary=audience_summary,
        matched_skills=matched_skills[:10],
        missing_skills=missing_skills[:10],
        requirement_matches=requirement_matches,
        semantic_similarity=semantic_similarity,
        keyword_coverage=keyword_coverage,
        strengths=strengths,
        gaps=gaps,
        suggestions=suggestions,
        rewrite_suggestions=rewrite_suggestions,
        hiring_manager_notes=hiring_manager_notes,
    )


def build_strengths(
    matched_skills: Sequence[str],
    requirement_matches: Sequence[RequirementMatch],
    matched_keywords: Sequence[str],
) -> List[str]:
    strengths: List[str] = []
    if matched_skills:
        strengths.append(
            "Strong skill alignment in " + ", ".join(matched_skills[:5]) + "."
        )

    project_count = count_resume_projects(" ".join(requirement.evidence for requirement in requirement_matches))
    if project_count:
        strengths.append(f"Shows {project_count} relevant project example{'s' if project_count != 1 else ''}.")

    top_requirements = [item for item in requirement_matches if item.matched][:3]
    for item in top_requirements:
        strengths.append(
            f"Clearly addresses '{item.requirement[:72]}' with matching resume evidence."
        )

    if matched_keywords:
        strengths.append(
            "JD keyword coverage includes " + ", ".join(matched_keywords[:6]) + "."
        )

    return strengths[:4]


def build_gaps(
    missing_skills: Sequence[str],
    requirement_matches: Sequence[RequirementMatch],
    missing_keywords: Sequence[str],
) -> List[str]:
    gaps: List[str] = []
    if missing_skills:
        gaps.append("Missing or weak skill coverage in " + ", ".join(missing_skills[:5]) + ".")

    low_matches = [item for item in requirement_matches if not item.matched][:3]
    for item in low_matches:
        gaps.append(f"Resume does not clearly show evidence for '{item.requirement[:88]}'.")

    if missing_keywords:
        gaps.append("Important JD terms not visible in the resume: " + ", ".join(missing_keywords[:6]) + ".")

    return gaps[:4]


def build_suggestions(
    missing_skills: Sequence[str],
    requirement_matches: Sequence[RequirementMatch],
) -> List[str]:
    suggestions: List[str] = []
    for skill in list(missing_skills)[:3]:
        suggestions.append(
            f"Add a bullet that shows measurable work with {skill}, including the project scope, tools, and outcome."
        )

    for item in requirement_matches:
        if not item.matched:
            suggestions.append(
                f"Rewrite one experience bullet to directly address: {item.requirement[:90]}."
            )
        if len(suggestions) >= 4:
            break

    if not suggestions:
        suggestions.append(
            "Add stronger impact metrics such as latency improvements, accuracy gains, or user adoption numbers."
        )
    return suggestions[:4]


def build_rewrite_suggestions(
    resume_bullets: Sequence[str],
    requirement_matches: Sequence[RequirementMatch],
    missing_skills: Sequence[str],
) -> List[RewriteSuggestion]:
    suggestions: List[RewriteSuggestion] = []
    unmatched = [item for item in requirement_matches if not item.matched]
    candidate_bullets = list(resume_bullets) or ["Worked on relevant software projects and responsibilities."]

    for index, requirement in enumerate(unmatched[:3]):
        original = candidate_bullets[min(index, len(candidate_bullets) - 1)]
        rewritten = rewrite_bullet_for_requirement(
            original_bullet=original,
            requirement=requirement.requirement,
            missing_skills=missing_skills,
        )
        suggestions.append(
            RewriteSuggestion(
                requirement=requirement.requirement,
                original_bullet=original,
                rewritten_bullet=rewritten,
                rationale=f"Links the resume bullet to the JD requirement '{requirement.requirement[:72]}'.",
            )
        )

    return suggestions


def rewrite_bullet_for_requirement(
    original_bullet: str,
    requirement: str,
    missing_skills: Sequence[str],
) -> str:
    cleaned = original_bullet.lstrip("-* ").strip().rstrip(".")
    requirement_phrase = requirement[0].lower() + requirement[1:] if requirement else "the role requirements"
    emphasis = ", ".join(missing_skills[:2]) if missing_skills else "measurable impact"
    return (
        f"Delivered {cleaned}, directly supporting {requirement_phrase}, "
        f"with explicit emphasis on {emphasis} and measurable outcomes."
    )


def build_hiring_manager_notes(
    requirement_matches: Sequence[RequirementMatch],
    matched_skills: Sequence[str],
    missing_skills: Sequence[str],
    resume_text: str,
) -> List[str]:
    notes: List[str] = []
    matched_count = sum(1 for item in requirement_matches if item.matched)
    notes.append(f"Matches {matched_count} of {len(requirement_matches)} extracted requirements.")
    if matched_skills:
        notes.append("Most evident skills: " + ", ".join(matched_skills[:5]) + ".")
    if missing_skills:
        notes.append("Primary risk areas: " + ", ".join(missing_skills[:4]) + ".")
    if count_resume_projects(resume_text):
        notes.append(f"Relevant project evidence count: {count_resume_projects(resume_text)}.")
    return notes[:4]


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token for token in tokenize(left) if token not in STOP_WORDS}
    right_tokens = {token for token in tokenize(right) if token not in STOP_WORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def build_summary(
    matched_skills: Sequence[str],
    missing_skills: Sequence[str],
    requirement_matches: Sequence[RequirementMatch],
    resume_text: str,
) -> str:
    matched_count = sum(1 for item in requirement_matches if item.matched)
    project_count = count_resume_projects(resume_text)
    leadership = has_any_term(resume_text, LEADERSHIP_TERMS)
    cloud = has_any_term(resume_text, CLOUD_TERMS)
    system_design = has_any_term(resume_text, SYSTEM_DESIGN_TERMS)

    strong_parts: list[str] = []
    if matched_skills:
        strong_parts.append(f"Strong {', '.join(matched_skills[:3])} coverage")
    if project_count:
        strong_parts.append(f"{project_count} relevant project{'s' if project_count != 1 else ''}")
    if matched_count:
        strong_parts.append(f"{matched_count} matched requirement{'s' if matched_count != 1 else ''}")

    gap_parts: list[str] = []
    if missing_skills:
        gap_parts.append(f"missing {', '.join(missing_skills[:3])}")
    if not cloud:
        gap_parts.append("no clear cloud platform evidence")
    if not leadership:
        gap_parts.append("no clear leadership examples")
    if not system_design:
        gap_parts.append("no clear system design evidence")

    strong_text = "; ".join(strong_parts[:3]) if strong_parts else "Limited strong alignment signals detected"
    gap_text = "; ".join(gap_parts[:3]) if gap_parts else "no major gaps detected"
    return f"{strong_text}. Gaps: {gap_text}."


def count_resume_projects(text: str) -> int:
    lowered = text.lower()
    explicit_match = re.search(r"\b(\d+|one|two|three|four|five)\s+relevant?\s+projects?\b", lowered)
    if explicit_match:
        mapping = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        raw = explicit_match.group(1)
        return int(raw) if raw.isdigit() else mapping.get(raw, 1)

    generic_match = re.search(r"\b(\d+|one|two|three|four|five)\s+(?:\w+\s+)?projects?\b", lowered)
    if generic_match:
        mapping = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        raw = generic_match.group(1)
        return int(raw) if raw.isdigit() else mapping.get(raw, 1)

    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    project_lines = [
        line for line in lines
        if line.startswith(("-", "*")) and any(term in line for term in ("project", "built", "developed", "created", "implemented"))
    ]
    if "projects" in lowered and project_lines:
        return min(len(project_lines), 5)
    if "projects" in lowered:
        return 1
    return 0


def has_any_term(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def split_requirement_line(line: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", line.strip())
    if len(normalized) < 12:
        return []

    if normalized.endswith(":"):
        return []

    parts = re.split(r",|;|\band\b|\bor\b", normalized, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        candidate = part.strip(" .:-")
        if len(candidate) < 12:
            continue
        cleaned.append(candidate[0].upper() + candidate[1:])
    return cleaned[:6] or [normalized]


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def average_requirement_score(requirement_matches: Sequence[RequirementMatch]) -> float:
    if not requirement_matches:
        return 0.0
    return sum(item.score for item in requirement_matches) / len(requirement_matches)


def extract_resume_bullets(text: str) -> List[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")) and len(stripped) > 12:
            bullets.append(stripped)
    if bullets:
        return bullets[:8]

    segments = split_into_segments(text)
    return [segment for segment in segments if len(segment) > 20][:5]


def build_audience_summary(
    view_mode: str,
    summary: str,
    matched_skills: Sequence[str],
    missing_skills: Sequence[str],
    hiring_manager_notes: Sequence[str],
    rewrite_suggestions: Sequence[RewriteSuggestion],
) -> str:
    if view_mode == "hiring_manager":
        strengths = ", ".join(matched_skills[:3]) if matched_skills else "limited matched skills"
        risk_terms = list(missing_skills[:3])
        if not risk_terms:
            if has_any_term(summary, CLOUD_TERMS):
                risk_terms.append("cloud exposure")
            if has_any_term(summary, LEADERSHIP_TERMS):
                risk_terms.append("leadership evidence")
            if has_any_term(summary, SYSTEM_DESIGN_TERMS):
                risk_terms.append("system design evidence")
        risks = ", ".join(risk_terms) if risk_terms else "no major risks"
        note = hiring_manager_notes[0] if hiring_manager_notes else "Review supporting evidence."
        return f"Hiring manager view: strengths in {strengths}; risks in {risks}. {note}"

    if rewrite_suggestions:
        return (
            f"Candidate view: {summary} Top rewrite focus: "
            f"{rewrite_suggestions[0].requirement[:80]}."
        )
    return f"Candidate view: {summary}"
