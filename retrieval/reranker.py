# retrieval/reranker.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from retrieval.hybrid_retriever import RetrievedChunk
import torch
from typing import List

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder (query + chunk joint encoding).
    ms-marco-MiniLM-L-6-v2: 22MB model, runs on CPU in <100ms for 20 chunks.
    Critical: reduces LLM context from 20 chunks → 5 = ~75% token savings.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[RERANK] Cross-encoder loaded on {self.device}")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        features = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1).cpu().tolist()

        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = [chunk for _, chunk in scored[:top_k]]

        print(f"[RERANK] {len(chunks)} → {len(top)} chunks")
        return top