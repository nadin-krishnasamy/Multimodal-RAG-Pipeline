# indexing/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch

class BGEEmbedder:
    """
    BGE-M3: Free after one-time 570MB download.
    1024-dim, multilingual, handles up to 8192 tokens per chunk.
    Instruction-prefixed embeddings for better retrieval accuracy.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: str = "./model_cache"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EMBED] Loading BGE-M3 on {device}")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device
        )
        self.dim = 1024

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch embed documents. Use during ingestion."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            prefixed = [f"Represent this document for retrieval: {t}" for t in batch]
            embeddings = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=(len(texts) > 100)
            )
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Uses different instruction prefix."""
        prefixed = f"Represent this query for retrieving relevant documents: {query}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0]