# tests/test_pipeline.py
import pytest
from ingestion.chunker import SemanticChunker
from ingestion.pdf_parser import ParsedElement

def test_semantic_chunker_keeps_tables_atomic():
    chunker = SemanticChunker(chunk_size=100, overlap=10)
    
    # Create a mock element mimicking an extracted table
    table_elem = ParsedElement(
        doc_id="test_doc",
        source_file="test.pdf",
        page=1,
        content_type="table",
        content="| Col 1 | Col 2 |\n|---|---|\n| Data A | Data B |"
    )
    
    chunks = chunker.chunk_elements([table_elem])
    
    assert len(chunks) == 1
    assert chunks[0].content_type == "table"
    assert chunks[0].metadata["is_atomic"] is True