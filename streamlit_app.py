from __future__ import annotations

import streamlit as st

from app.analyzer import analyze_resume
from app.embeddings import build_embedding_provider
from app.parsers import extract_text_from_upload, get_parser_backend_name
from app.vectorstores import build_vector_store


st.set_page_config(page_title="ResumeIQ", page_icon="AI", layout="wide")

st.title("ResumeIQ")
st.caption("Recruiter-style resume matching with PyMuPDF, FAISS, and OpenAI-ready embeddings.")

with st.sidebar:
    st.subheader("How it works")
    st.write("1. Upload resumes or paste text.")
    st.write("2. Paste the target job description.")
    st.write("3. Review match score, strengths, gaps, and suggestions.")

job_description = st.text_area(
    "Job Description",
    height=240,
    placeholder="Paste the full role description here.",
)

resume_text = st.text_area(
    "Paste Resume Text",
    height=220,
    placeholder="Optional if you are uploading PDFs below.",
)

uploads = st.file_uploader(
    "Upload one or more resumes",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Analyze", type="primary"):
    if not job_description.strip():
        st.error("Please provide a job description.")
        st.stop()

    sources: list[tuple[str, str]] = []
    if resume_text.strip():
        sources.append(("Pasted Resume", resume_text.strip()))

    for upload in uploads:
        text = extract_text_from_upload(upload.name, upload.getvalue())
        if text.strip():
            sources.append((upload.name, text))

    if not sources:
        st.error("Please paste a resume or upload at least one file.")
        st.stop()

    embedding_provider = build_embedding_provider()
    vector_store = build_vector_store()
    st.caption(
        f"Active stack: embeddings={embedding_provider.name}, vectors={vector_store.name}, parser={get_parser_backend_name()}"
    )
    results = [
        analyze_resume(
            resume_name=name,
            resume_text=text,
            job_description=job_description,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        for name, text in sources
    ]
    results.sort(key=lambda item: item.match_score, reverse=True)

    st.subheader("Ranked Shortlist")
    for index, result in enumerate(results, start=1):
        st.write(f"{index}. {result.resume_name} - {result.match_score}%")

    for result in results:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.subheader(result.resume_name)
            right.metric("Match", f"{result.match_score}%")
            st.write(
                f"Semantic similarity: {result.semantic_similarity * 100:.1f}% | "
                f"Keyword coverage: {result.keyword_coverage * 100:.1f}%"
            )
            st.write("Matched skills:", ", ".join(result.matched_skills) or "None")
            st.write("Missing skills:", ", ".join(result.missing_skills) or "None")

            col1, col2, col3 = st.columns(3)
            col1.write("Strengths")
            col1.write("\n".join(f"- {item}" for item in result.strengths) or "- None")
            col2.write("Gaps")
            col2.write("\n".join(f"- {item}" for item in result.gaps) or "- None")
            col3.write("Suggestions")
            col3.write("\n".join(f"- {item}" for item in result.suggestions) or "- None")

            st.write("Requirement Evidence")
            for match in result.matched_requirements:
                state = "Matched" if match.matched else "Weak match"
                st.write(f"- {state} ({match.score * 100:.1f}%): {match.requirement}")
                st.caption(match.evidence or "No supporting evidence found.")
