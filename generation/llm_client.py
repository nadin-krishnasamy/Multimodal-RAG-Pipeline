# generation/llm_client.py
import anthropic
from typing import List, Dict
from generation.prompt_builder import TokenAwarePromptBuilder
from generation.citation_extractor import CitationExtractor
from retrieval.hybrid_retriever import RetrievedChunk

class CitationAwareLLM:
    def __init__(self, api_key: str, model: str = "claude-haiku-3-5-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.prompt_builder = TokenAwarePromptBuilder()
        self.citation_extractor = CitationExtractor()

    def answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_context_tokens: int = 3000,
        max_response_tokens: int = 1024
    ) -> Dict:
        user_message, used_chunks = self.prompt_builder.build(
            query, chunks, max_context_tokens
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_response_tokens,
            system=TokenAwarePromptBuilder.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        raw_answer = response.content[0].text
        citations = self.citation_extractor.extract(raw_answer, used_chunks)

        return {
            "query": query,
            "answer": raw_answer,
            "citations": citations,
            "chunks_used": len(used_chunks),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_cost_usd": self._estimate_cost(response.usage)
        }

    def _estimate_cost(self, usage) -> float:
        # Claude Haiku 3.5 pricing ($0.80 per million input, $4.00 per million output tokens)
        input_cost = (usage.input_tokens / 1_000_000) * 0.80
        output_cost = (usage.output_tokens / 1_000_000) * 4.00
        return round(input_cost + output_cost, 6)