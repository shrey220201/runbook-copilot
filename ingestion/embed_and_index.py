"""Embedding and Qdrant indexing pipeline for Runbook Copilot.

Embeds technical document chunks using SentenceTransformers (EMBEDDING_MODEL_NAME)
and upserts them into Qdrant vector database (QDRANT_HOST / QDRANT_PORT / QDRANT_COLLECTION_NAME).
Ensures idempotency using deterministic point IDs.
"""

import uuid
import logging
from typing import List, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from ingestion.loaders import load_all_sources, load_documents_from_disk
from ingestion.chunking import chunk_all_documents, DocumentChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_deterministic_id(chunk_id: str) -> str:
    """Generate a consistent UUID5 string for a given chunk ID to guarantee idempotency."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class Indexer:
    """Handles embedding creation and idempotent indexing into Qdrant."""

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

        logger.info(f"Initializing Qdrant client at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)

        logger.info(f"Loading embedding model: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def ensure_collection(self, recreate: bool = False):
        """Ensure that the Qdrant collection exists with proper vector configuration."""
        collections = [c.name for c in self.client.get_collections().collections]

        if recreate and self.collection_name in collections:
            logger.warning(f"Recreating existing collection: {self.collection_name}")
            self.client.delete_collection(collection_name=self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            logger.info(f"Creating collection '{self.collection_name}' (dim={self.embedding_dim}, distance=Cosine)")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=self.embedding_dim,
                    distance=rest_models.Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> int:
        """Embed and upsert document chunks into Qdrant idempotently."""
        if not chunks:
            logger.warning("No chunks provided for indexing.")
            return 0

        self.ensure_collection(recreate=False)

        total = len(chunks)
        logger.info(f"Starting indexing for {total} chunks in batches of {batch_size}...")

        total_upserted = 0
        for i in tqdm(range(0, total, batch_size), desc="Indexing Chunks"):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            # Compute embeddings
            embeddings = self.encoder.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Build Qdrant PointStructs
            points = []
            for chunk, vector in zip(batch, embeddings):
                point_id = generate_deterministic_id(chunk.chunk_id)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "source": chunk.source,
                    "url": chunk.url,
                    "header_path": chunk.header_path,
                    "content": chunk.content,
                    "has_code": chunk.has_code,
                    **chunk.metadata,
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            # Idempotent upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)

        logger.info(f"Successfully upserted {total_upserted} points into '{self.collection_name}'.")
        return total_upserted


def run_pipeline(
    recreate_collection: bool = False,
    fetch_live: bool = False,
    batch_size: int = 32,
):
    """Run complete ingestion, chunking, and indexing pipeline."""
    # Check if documents already exist on disk, otherwise fetch
    docs = []
    if not fetch_live:
        docs = load_documents_from_disk()

    if not docs:
        logger.info("No cached documents found on disk (or --fetch-live requested). Fetching from live sources...")
        docs = load_all_sources(save_to_disk=True)

    if not docs:
        logger.warning("No documents were loaded. Pipeline aborted.")
        return

    chunks = chunk_all_documents(docs)
    logger.info(f"Generated {len(chunks)} technical chunks across {len(docs)} documents.")

    indexer = Indexer()
    if recreate_collection:
        indexer.ensure_collection(recreate=True)
    indexer.index_chunks(chunks, batch_size=batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Runbook Ingestion & Indexing Pipeline")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection from scratch",
    )
    parser.add_argument(
        "--fetch-live",
        action="store_true",
        help="Force re-fetching from live sources rather than reading cached raw JSON from disk",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding and upserting (default: 32)",
    )
    args = parser.parse_args()

    run_pipeline(
        recreate_collection=args.recreate,
        fetch_live=args.fetch_live,
        batch_size=args.batch_size,
    )
