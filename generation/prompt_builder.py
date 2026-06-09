# generation/prompt_builder.py
import tiktoken
from retrieval.hybrid_retriever import RetrievedChunk
from typing import List, Tuple

class TokenAwarePromptBuilder:
    """
    Assembles context under a hard token ceiling.
    Lower-ranked chunks are truncated first. Query is never truncated.
    """
    ENCODER = tiktoken.get_encoding("cl100k_base")

    SYSTEM_PROMPT = """You are a precise document analyst. Answer ONLY from the provided context.

Rules:
1. Every factual claim MUST cite its source as [SOURCE_FILE, Page X, Chunk Y].
2. If the answer spans multiple chunks, cite ALL relevant chunks.
3. If context is insufficient, say exactly: "Insufficient context to answer."
4. Never hallucinate. Never use prior knowledge outside the context.
5. End your response with: Confidence: High / Medium / Low (based on how completely the context covers the question)."""

    def count_tokens(self, text: str) -> int:
        return len(self.ENCODER.encode(text))

    def build(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_context_tokens: int = 3000
    ) -> Tuple[str, List[RetrievedChunk]]:

        token_budget = max_context_tokens
        token_budget -= self.count_tokens(self.SYSTEM_PROMPT)
        token_budget -= self.count_tokens(query) + 50

        context_parts = []
        used_chunks = []

        for chunk in chunks:
            label = f"[{chunk.source_file}, Page {chunk.page}, Chunk {chunk.chunk_id}]"
            formatted = f"{label}\nType: {chunk.content_type}\n{chunk.text}"
            chunk_tokens = self.count_tokens(formatted)

            if chunk_tokens <= token_budget:
                context_parts.append(formatted)
                used_chunks.append(chunk)
                token_budget -= chunk_tokens
            else:
                available_chars = token_budget * 3
                if available_chars > 100:
                    truncated = formatted[:available_chars] + "... [truncated]"
                    context_parts.append(truncated)
                    used_chunks.append(chunk)
                break

        context_block = "\n\n---\n\n".join(context_parts)

        user_message = f"""CONTEXT DOCUMENTS:
{context_block}

---
QUESTION: {query}

Answer based strictly on the context above. Include citations for every claim."""

        return user_message, used_chunks