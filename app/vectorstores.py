from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import faiss
import numpy as np

from app.config import settings


@dataclass
class SearchResult:
    score: float
    index: int


class BaseVectorStore:
    name = "base"

    def search(self, corpus_embeddings: np.ndarray, query_embeddings: np.ndarray) -> list[SearchResult]:
        raise NotImplementedError


class FaissVectorStore(BaseVectorStore):
    name = "faiss"

    def search(self, corpus_embeddings: np.ndarray, query_embeddings: np.ndarray) -> list[SearchResult]:
        index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)
        similarities, nearest = index.search(query_embeddings, 1)
        return [
            SearchResult(score=float(score_row[0]), index=int(index_row[0]))
            for score_row, index_row in zip(similarities, nearest)
        ]


class ChromaVectorStore(BaseVectorStore):
    name = "chromadb"

    def __init__(self) -> None:
        import chromadb

        self.client = chromadb.EphemeralClient()

    def search(self, corpus_embeddings: np.ndarray, query_embeddings: np.ndarray) -> list[SearchResult]:
        collection = self.client.get_or_create_collection(name="resumeiq-session")
        ids = [f"doc-{i}" for i in range(len(corpus_embeddings))]
        collection.upsert(
            ids=ids,
            embeddings=corpus_embeddings.tolist(),
        )
        results = []
        for row in query_embeddings.tolist():
            response = collection.query(query_embeddings=[row], n_results=1)
            match_id = response["ids"][0][0]
            distance = float(response["distances"][0][0])
            doc_index = ids.index(match_id)
            results.append(SearchResult(score=max(0.0, 1.0 - distance), index=doc_index))
        return results


class PineconeVectorStore(BaseVectorStore):
    name = "pinecone"

    def __init__(self) -> None:
        from pinecone import Pinecone

        if not settings.pinecone_api_key or not settings.pinecone_index_name:
            raise ValueError("Pinecone requires PINECONE_API_KEY and PINECONE_INDEX_NAME.")
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index_name)

    def search(self, corpus_embeddings: np.ndarray, query_embeddings: np.ndarray) -> list[SearchResult]:
        vectors = [
            {"id": f"doc-{i}", "values": vector.tolist(), "metadata": {"ordinal": i}}
            for i, vector in enumerate(corpus_embeddings)
        ]
        namespace = "resumeiq"
        self.index.upsert(vectors=vectors, namespace=namespace)
        results = []
        for row in query_embeddings:
            response = self.index.query(
                vector=row.tolist(),
                namespace=namespace,
                top_k=1,
                include_metadata=True,
            )
            match = response["matches"][0]
            results.append(
                SearchResult(
                    score=float(match["score"]),
                    index=int(match["metadata"]["ordinal"]),
                )
            )
        return results


def build_vector_store() -> BaseVectorStore:
    choice = settings.vector_backend.lower()
    if choice == "chromadb":
        try:
            return ChromaVectorStore()
        except Exception:
            return FaissVectorStore()
    if choice == "pinecone":
        try:
            return PineconeVectorStore()
        except Exception:
            return FaissVectorStore()
    return FaissVectorStore()
