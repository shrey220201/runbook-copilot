"""
Targeted addition: index only the NAKIVO documents into the existing
Qdrant collection, without re-embedding everything else. Safe to run
even though embed_and_index already ran, since chunk IDs are deterministic
(idempotent) - this only adds the 8 NAKIVO docs' chunks, which have never
been indexed before.
"""

from ingestion.loaders import load_documents_from_disk
from ingestion.chunking import chunk_all_documents
from ingestion.embed_and_index import Indexer

# Load only the nakivo documents
docs = load_documents_from_disk(source="nakivo")
print(f"Loaded {len(docs)} NAKIVO document(s)")

chunks = chunk_all_documents(docs)
print(f"Generated {len(chunks)} chunks from NAKIVO docs")

indexer = Indexer()
indexer.index_chunks(chunks, batch_size=32)