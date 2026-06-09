# retrieval/hybrid_retriever.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, Prefetch, FusionQuery
)
from indexing.embedder import BGEEmbedder
from dataclasses import dataclass
from typing import List
import json

@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    content_type: str
    text: str
    score: float
    metadata: dict

class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) over dense + sparse vectors.
    RRF formula: score = Σ 1/(k + rank_i) where k=60 by default.
    Result: keyword-exact matches AND semantic matches both surface.
    """
    def __init__(self, client: QdrantClient, collection: str, embedder: BGEEmbedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        query_emb = self.embedder.embed_query(query)

        tokens = query.lower().split()
        sparse_indices = [hash(token) % 65536 for token in tokens]
        sparse_values = [1.0] * len(tokens)

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=query_emb.tolist(),
                    using="",           # dense vector branch
                    limit=top_k * 2
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="bm25",       # sparse vector branch
                    limit=top_k * 2
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k
        ).points

        retrieved = []
        for r in results:
            retrieved.append(RetrievedChunk(
                chunk_id=r.payload["chunk_id"],
                doc_id=r.payload["doc_id"],
                source_file=r.payload["source_file"],
                page=r.payload["page"],
                content_type=r.payload["content_type"],
                text=r.payload["text"],
                score=r.score,
                metadata=json.loads(r.payload.get("metadata", "{}"))
            ))

        return retrieved