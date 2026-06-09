# config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = "multimodal_rag"
    VECTOR_DIM: int = 1024  # BGE-M3

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_BATCH_SIZE: int = 32
    EMBED_CACHE_DIR: str = "./model_cache"

    # Retrieval
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    HYBRID_ALPHA: float = 0.6

    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = "claude-haiku-3-5-20241022"
    MAX_CONTEXT_TOKENS: int = 3000
    MAX_RESPONSE_TOKENS: int = 1024

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

cfg = Config()