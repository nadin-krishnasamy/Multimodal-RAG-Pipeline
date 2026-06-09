# indexing/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector
)
from typing import List
from ingestion.chunker import Chunk
from indexing.embedder import BGEEmbedder
import uuid, json

class QdrantVectorStore:
    """
    Qdrant collection with:
    - Dense vectors (BGE-M3, 1024-dim, cosine) for semantic search
    - Sparse vectors (BM25 term weights) for keyword search
    Both are needed for hybrid retrieval via RRF fusion.
    """
    def __init__(self, url: str, collection_name: str, dim: int = 1024):
        self.client = QdrantClient(url=url)
        self.collection = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                ),
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            print(f"[STORE] Created collection: {self.collection}")
        else:
            print(f"[STORE] Collection exists: {self.collection}")

    def upsert_chunks(self, chunks: List[Chunk], embedder: BGEEmbedder):
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts, batch_size=embedder.dim // 32)
        tokenized = [t.lower().split() for t in texts]

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            sparse_indices = [hash(token) % 65536 for token in tokenized[i]]
            sparse_values = [1.0] * len(tokenized[i])

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "": emb.tolist(),
                    "bm25": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_file": chunk.source_file,
                    "page": chunk.page,
                    "content_type": chunk.content_type,
                    "text": chunk.text,
                    "metadata": json.dumps(chunk.metadata)
                }
            )
            points.append(point)

        # Batch upsert in groups of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i + 100]
            )

        print(f"[STORE] Upserted {len(points)} chunks")

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status)
        }