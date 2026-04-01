from __future__ import annotations

import io
import os
import unittest

import fitz
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["VECTOR_BACKEND"] = "faiss"
os.environ["RESUME_PARSER"] = "pymupdf"

from app.main import app


def make_pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def make_blank_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page()
    payload = document.tobytes()
    document.close()
    return payload


def make_image_only_pdf_bytes(text: str) -> bytes:
    image = Image.new("RGB", (1200, 400), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 120), text, fill="black")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    document = fitz.open()
    page = document.new_page(width=1200, height=400)
    rect = fitz.Rect(0, 0, 1200, 400)
    page.insert_image(rect, stream=image_bytes.getvalue())
    payload = document.tobytes()
    document.close()
    return payload


class ResumeIQApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("embedding_provider", response.json())
        self.assertIn("vector_backend", response.json())

    def test_analyze_pasted_resume(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={
                "job_description": "Need Python, FastAPI, FAISS, and OpenAI experience.",
                "resume_text": "Built a Python FastAPI app using OpenAI embeddings and FAISS.",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["analyzed_count"], 1)
        self.assertGreaterEqual(payload["results"][0]["match_score"], 40)
        self.assertEqual(payload["vector_backend"], "faiss")
        self.assertTrue(payload["results"][0]["summary"])

    def test_analyze_uploaded_pdf_and_rank_multiple_resumes(self) -> None:
        strong_pdf = make_pdf_bytes(
            "Python FastAPI OpenAI FAISS NLP project with PDF parsing and ranking."
        )
        weak_pdf = make_pdf_bytes("Front-end design portfolio with CSS and branding work.")
        response = self.client.post(
            "/api/analyze",
            data={
                "job_description": "Looking for Python, FastAPI, FAISS, NLP, and PDF parsing experience."
            },
            files=[
                ("resume_files", ("strong.pdf", io.BytesIO(strong_pdf), "application/pdf")),
                ("resume_files", ("weak.pdf", io.BytesIO(weak_pdf), "application/pdf")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["analyzed_count"], 2)
        self.assertEqual(payload["results"][0]["resume_name"], "strong.pdf")
        self.assertEqual(payload["top_resume"], "strong.pdf")

    def test_step_one_demo_flow_has_recruiter_style_output(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={
                "job_description": (
                    "We need an AI intern with Python, FastAPI, NLP, cloud exposure, "
                    "leadership potential, and system design awareness."
                ),
                "resume_text": (
                    "Built a Python FastAPI resume matcher using SentenceTransformers and FAISS. "
                    "Developed two NLP projects and shipped an end-to-end recruiting workflow."
                ),
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertGreaterEqual(result["match_score"], 45)
        self.assertIn("Strong", result["summary"])
        self.assertTrue(result["strengths"])
        self.assertTrue(result["gaps"])

    def test_candidate_view_returns_tailored_rewrite_suggestions(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={
                "view_mode": "candidate",
                "job_description": "Need Python, FastAPI, cloud deployment, and leadership examples.",
                "resume_text": (
                    "- Built a Python FastAPI resume matcher.\n"
                    "- Developed NLP workflows for candidate screening.\n"
                ),
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertTrue(result["rewrite_suggestions"])
        self.assertIn("Candidate view", result["audience_summary"])

    def test_hiring_manager_view_returns_manager_notes(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={
                "view_mode": "hiring_manager",
                "job_description": "Need Python, FastAPI, NLP, and cloud deployment experience.",
                "resume_text": "Built Python FastAPI NLP services and shipped project features.",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        result = payload["results"][0]
        self.assertEqual(payload["view_mode"], "hiring_manager")
        self.assertTrue(result["hiring_manager_notes"])
        self.assertIn("Hiring manager view", result["audience_summary"])

    def test_requires_resume_input(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={"job_description": "Python developer role"},
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_invalid_pdf(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={"job_description": "Python developer role"},
            files=[
                ("resume_files", ("broken.pdf", io.BytesIO(b"not-a-real-pdf"), "application/pdf")),
            ],
        )
        self.assertEqual(response.status_code, 400)
        self.assertTrue(
            "invalid or unreadable" in response.json()["detail"].lower()
            or "ocr could not recover" in response.json()["detail"].lower()
        )

    def test_rejects_image_only_or_blank_pdf(self) -> None:
        response = self.client.post(
            "/api/analyze",
            data={"job_description": "Python developer role"},
            files=[
                ("resume_files", ("blank.pdf", io.BytesIO(make_blank_pdf_bytes()), "application/pdf")),
            ],
        )
        self.assertEqual(response.status_code, 400)
        self.assertTrue(
            "does not contain extractable text" in response.json()["detail"].lower()
            or "too limited" in response.json()["detail"].lower()
        )

    def test_ocr_recovers_image_only_pdf(self) -> None:
        pdf_bytes = make_image_only_pdf_bytes("Python FastAPI NLP FAISS project experience")
        response = self.client.post(
            "/api/analyze",
            data={"job_description": "Python FastAPI NLP FAISS role"},
            files=[
                ("resume_files", ("scan.pdf", io.BytesIO(pdf_bytes), "application/pdf")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertGreaterEqual(result["match_score"], 35)


if __name__ == "__main__":
    unittest.main()
