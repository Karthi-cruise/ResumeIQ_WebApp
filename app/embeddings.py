from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np
import requests
from google import genai
from openai import OpenAI

from app.config import settings


class BaseEmbeddingProvider:
    name = "base"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class LocalHashEmbeddingProvider(BaseEmbeddingProvider):
    name = "local-hash"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = [self._embed_one(text) for text in texts]
        return np.vstack(vectors).astype("float32")

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector

        for token in tokens:
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            weight = 1.0 + math.log(1 + len(token))
            vector[bucket] += weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )
            vector = np.array(response.embeddings[0].values, dtype="float32")
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector)
        return np.vstack(vectors).astype("float32")


class HuggingFaceInferenceEmbeddingProvider(BaseEmbeddingProvider):
    name = "huggingface"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            response = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            vector = np.array(_coerce_huggingface_vector(payload), dtype="float32")
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector)
        return np.vstack(vectors).astype("float32")


def _coerce_huggingface_vector(payload: object) -> List[float]:
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        if payload and payload[0] and isinstance(payload[0][0], list):
            token_vectors = np.array(payload[0], dtype="float32")
            return token_vectors.mean(axis=0).tolist()
        token_vectors = np.array(payload, dtype="float32")
        return token_vectors.mean(axis=0).tolist()
    if isinstance(payload, list):
        return [float(item) for item in payload]
    raise ValueError("Unexpected Hugging Face embedding response shape.")


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.fallback = LocalHashEmbeddingProvider()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        try:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return np.array(vectors, dtype="float32")
        except Exception:
            return self.fallback.embed_texts(texts)


def build_embedding_provider() -> BaseEmbeddingProvider:
    provider_choice = settings.embedding_provider.lower()
    if provider_choice == "openai" and settings.openai_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    if provider_choice == "gemini" and settings.gemini_api_key:
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
        )
    if provider_choice == "huggingface" and settings.huggingface_api_key:
        return HuggingFaceInferenceEmbeddingProvider(
            api_key=settings.huggingface_api_key,
            model=settings.huggingface_embedding_model,
        )
    if provider_choice == "sentence-transformers":
        try:
            return SentenceTransformerEmbeddingProvider(
                model_name=settings.sentence_transformer_model,
            )
        except Exception:
            return LocalHashEmbeddingProvider()
    if provider_choice == "local":
        return LocalHashEmbeddingProvider()
    if settings.openai_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    if settings.gemini_api_key:
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
        )
    if settings.huggingface_api_key:
        return HuggingFaceInferenceEmbeddingProvider(
            api_key=settings.huggingface_api_key,
            model=settings.huggingface_embedding_model,
        )
    return LocalHashEmbeddingProvider()
