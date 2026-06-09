# ingestion/chunker.py
import re
from dataclasses import dataclass, field
from typing import List
from ingestion.pdf_parser import ParsedElement

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    content_type: str
    text: str
    metadata: dict = field(default_factory=dict)

class SemanticChunker:
    """
    Paragraph-aware chunking. Tables and images are kept atomic (never split).
    Text chunks use overlap to preserve context across boundaries.
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_elements(self, elements: List[ParsedElement]) -> List[Chunk]:
        chunks = []
        for elem in elements:
            if elem.content_type in ("image", "table"):
                chunks.append(Chunk(
                    chunk_id=f"{elem.doc_id}_p{elem.page}_{elem.content_type}",
                    doc_id=elem.doc_id,
                    source_file=elem.source_file,
                    page=elem.page,
                    content_type=elem.content_type,
                    text=elem.content,
                    metadata={**elem.metadata, "is_atomic": True}
                ))
            else:
                paragraphs = self._split_paragraphs(elem.content)
                text_chunks = self._merge_paragraphs(paragraphs)
                for idx, chunk_text in enumerate(text_chunks):
                    chunks.append(Chunk(
                        chunk_id=f"{elem.doc_id}_p{elem.page}_t{idx}",
                        doc_id=elem.doc_id,
                        source_file=elem.source_file,
                        page=elem.page,
                        content_type="text",
                        text=chunk_text,
                        metadata={**elem.metadata, "chunk_idx": idx}
                    ))
        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n{2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        chunks = []
        current = ""
        overlap_buffer = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= self.chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                    overlap_buffer = current[-self.overlap:] if len(current) > self.overlap else current
                current = (overlap_buffer + "\n\n" + para).strip()

        if current:
            chunks.append(current)

        return chunks