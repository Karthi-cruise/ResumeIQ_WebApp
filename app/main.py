from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analyzer import analyze_resume
from app.config import settings
from app.embeddings import build_embedding_provider
from app.parsers import ResumeParseError, extract_text_from_upload, get_parser_backend_name
from app.schemas import BatchAnalysisResponse
from app.vectorstores import build_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "settings": settings,
        },
    )


@app.get("/api/health")
async def health() -> dict:
    provider = build_embedding_provider()
    vector_store = build_vector_store()
    return {
        "status": "ok",
        "embedding_provider": provider.name,
        "vector_backend": vector_store.name,
        "parser_backend": get_parser_backend_name(),
    }


async def read_optional_upload(upload: UploadFile | None, fallback_name: str) -> str:
    if upload is None or not upload.filename:
        return ""
    content = await upload.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} exceeds the {settings.max_file_size_mb} MB file size limit.",
        )
    try:
        return extract_text_from_upload(upload.filename or fallback_name, content)
    except ResumeParseError as exc:
        raise HTTPException(status_code=400, detail=f"{upload.filename}: {exc}") from exc


@app.post("/api/analyze", response_model=BatchAnalysisResponse)
async def analyze(
    job_description: str = Form(""),
    resume_text: str = Form(""),
    view_mode: str = Form("candidate"),
    job_description_file: UploadFile | None = File(default=None),
    resume_files: List[UploadFile] = File(default=[]),
) -> BatchAnalysisResponse:
    if view_mode not in {"candidate", "hiring_manager"}:
        raise HTTPException(status_code=400, detail="view_mode must be either 'candidate' or 'hiring_manager'.")

    job_description_parts: list[str] = []
    if job_description.strip():
        job_description_parts.append(job_description.strip())
    uploaded_job_description = await read_optional_upload(job_description_file, "job_description.pdf")
    if uploaded_job_description.strip():
        job_description_parts.append(uploaded_job_description.strip())
    combined_job_description = "\n\n".join(part for part in job_description_parts if part)
    if not combined_job_description.strip():
        raise HTTPException(status_code=400, detail="Provide a job description by paste or file upload.")

    sources: list[tuple[str, str]] = []
    if resume_text.strip():
        sources.append(("Pasted Resume", resume_text.strip()))

    for upload in resume_files:
        if upload is None or not upload.filename:
            continue
        text = await read_optional_upload(upload, "resume.pdf")
        sources.append((upload.filename or "Uploaded Resume", text))

    if not sources:
        raise HTTPException(status_code=400, detail="Provide at least one resume file or pasted resume text.")

    embedding_provider = build_embedding_provider()
    vector_store = build_vector_store()
    results = [
        analyze_resume(
            resume_name=name,
            resume_text=text,
            job_description=combined_job_description,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            view_mode=view_mode,
        )
        for name, text in sources
    ]
    results.sort(key=lambda item: item.match_score, reverse=True)

    return BatchAnalysisResponse(
        job_description=combined_job_description,
        analyzed_count=len(results),
        top_resume=results[0].resume_name if results else "",
        view_mode=view_mode,
        embedding_provider=embedding_provider.name,
        vector_backend=vector_store.name,
        parser_backend=get_parser_backend_name(),
        results=results,
    )
