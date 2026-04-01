from __future__ import annotations

import re
from pathlib import Path

import fitz
from PIL import Image

from app.config import settings


TEXT_EXTENSIONS = {".txt", ".md", ".rst"}


class ResumeParseError(ValueError):
    """Raised when a resume cannot be parsed into usable text."""


def normalize_whitespace(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def extract_text_from_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(content)
    if suffix in TEXT_EXTENSIONS:
        return normalize_whitespace(content.decode("utf-8", errors="ignore"))
    return normalize_whitespace(content.decode("utf-8", errors="ignore"))


def extract_text_from_pdf(content: bytes) -> str:
    parser_choice = settings.resume_parser.lower()
    if parser_choice in {"pdfplumber", "docling", "pymupdf"}:
        selected = {
            "pdfplumber": extract_text_with_pdfplumber,
            "docling": extract_text_with_docling,
            "pymupdf": extract_text_with_pymupdf,
        }[parser_choice]
        try:
            text = selected(content)
            validate_extracted_pdf_text(text)
            return text
        except ResumeParseError:
            ocr_text = extract_text_with_ocr(content)
            validate_extracted_pdf_text(ocr_text)
            return ocr_text

    for extractor in (
        extract_text_with_pymupdf,
        extract_text_with_pdfplumber,
        extract_text_with_docling,
    ):
        try:
            text = extractor(content)
            validate_extracted_pdf_text(text)
            return text
        except ResumeParseError:
            continue
    ocr_text = extract_text_with_ocr(content)
    validate_extracted_pdf_text(ocr_text)
    return ocr_text


def extract_text_with_pymupdf(content: bytes) -> str:
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except Exception as exc:
        raise ResumeParseError("The uploaded PDF appears to be invalid or unreadable.") from exc
    return normalize_whitespace("\n".join(pages))


def extract_text_with_pdfplumber(content: bytes) -> str:
    try:
        import io
        import pdfplumber
    except Exception:
        return ""

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise ResumeParseError("The uploaded PDF appears to be invalid or unreadable.") from exc
    return normalize_whitespace("\n".join(pages))


def extract_text_with_docling(content: bytes) -> str:
    try:
        import tempfile
        from docling.document_converter import DocumentConverter
    except Exception:
        return ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(content)
            handle.flush()
            result = DocumentConverter().convert(handle.name)
    except Exception as exc:
        raise ResumeParseError("The uploaded PDF appears to be invalid or unreadable.") from exc
    return normalize_whitespace(result.document.export_to_markdown())


def get_parser_backend_name() -> str:
    choice = settings.resume_parser.lower()
    if choice in {"pymupdf", "pdfplumber", "docling"}:
        return choice
    if choice == "auto":
        return "auto"
    return choice


def validate_extracted_pdf_text(text: str) -> None:
    normalized = normalize_whitespace(text)
    if not normalized:
        raise ResumeParseError(
            "The PDF does not contain extractable text. It may be image-only or require OCR before upload."
        )
    if len(normalized) < 30:
        raise ResumeParseError(
            "The PDF text is too limited to analyze reliably. Please upload a text-based resume PDF or paste the resume text."
        )


def extract_text_with_ocr(content: bytes) -> str:
    try:
        import pytesseract
    except Exception as exc:
        raise ResumeParseError(
            "The PDF does not contain extractable text and OCR is not available in this environment."
        ) from exc

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            pages = []
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                pages.append(pytesseract.image_to_string(image))
    except Exception as exc:
        raise ResumeParseError(
            "The PDF does not contain extractable text and OCR could not recover readable content."
        ) from exc

    return normalize_whitespace("\n".join(pages))
