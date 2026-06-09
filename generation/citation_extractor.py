# generation/citation_extractor.py
import re
from typing import List, Dict
from retrieval.hybrid_retriever import RetrievedChunk

class CitationExtractor:
    """
    Parses [SOURCE_FILE, Page X, Chunk Y] patterns from LLM output.
    Maps each citation back to the actual chunk for preview text.
    """
    PATTERN = r'\[([^\]]+),\s*Page\s*(\d+),\s*Chunk\s*([^\]]+)\]'

    def extract(self, answer: str, chunks: List[RetrievedChunk]) -> List[Dict]:
        matches = re.findall(self.PATTERN, answer)
        citations = []

        for source, page, chunk_id in matches:
            matched = next(
                (c for c in chunks
                 if c.source_file in source or chunk_id.strip() in c.chunk_id),
                None
            )
            citations.append({
                "source_file": source.strip(),
                "page": int(page),
                "chunk_id": chunk_id.strip(),
                "chunk_text_preview": matched.text[:150] + "..." if matched else "",
                "content_type": matched.content_type if matched else "unknown"
            })

        return citations