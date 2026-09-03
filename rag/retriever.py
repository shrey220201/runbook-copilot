"""Retriever module for querying Qdrant vector database."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    score: float
    content: str
    title: str
    source: str
    url: str
    header_path: str
    has_code: bool
    metadata: Dict[str, Any]


class QdrantRetriever:
    """Semantic vector search against Qdrant runbooks collection."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.model_name = model_name or EMBEDDING_MODEL_NAME

        self.client = QdrantClient(host=self.host, port=self.port)
        self.encoder = SentenceTransformer(self.model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Search the vector database for relevant runbook chunks."""
        # Encode query
        query_vector = self.encoder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Build optional Qdrant filter
        query_filter = None
        if source_filter:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="source",
                        match=rest_models.MatchValue(value=source_filter),
                    )
                ]
            )

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        ).points

        results: List[RetrievedChunk] = []
        for hit in search_results:
            payload = hit.payload or {}
            chunk = RetrievedChunk(
                id=str(hit.id),
                score=hit.score,
                content=payload.get("content", ""),
                title=payload.get("title", "Untitled Runbook"),
                source=payload.get("source", "Unknown Source"),
                url=payload.get("url", ""),
                header_path=payload.get("header_path", ""),
                has_code=payload.get("has_code", False),
                metadata=payload,
            )
            results.append(chunk)

        return results


# Default singleton instance
retriever = QdrantRetriever()
