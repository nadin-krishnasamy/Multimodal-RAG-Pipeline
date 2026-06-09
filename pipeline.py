# pipeline.py
from ingestion.pdf_parser import PDFParser
from ingestion.table_parser import TableParser
from ingestion.image_parser import ImageDescriber
from ingestion.chunker import SemanticChunker
from indexing.embedder import BGEEmbedder
from indexing.vector_store import QdrantVectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.llm_client import CitationAwareLLM
from config import cfg
from qdrant_client import QdrantClient

class MultimodalRAGPipeline:
    def __init__(self):
        print("[INIT] Loading pipeline components...")

        self.pdf_parser = PDFParser()
        self.table_parser = TableParser()
        self.image_describer = ImageDescriber(cfg.ANTHROPIC_API_KEY)
        self.chunker = SemanticChunker(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)

        self.embedder = BGEEmbedder(cfg.EMBED_MODEL, cfg.EMBED_CACHE_DIR)
        self.vector_store = QdrantVectorStore(
            cfg.QDRANT_URL, cfg.QDRANT_COLLECTION, cfg.VECTOR_DIM
        )

        qdrant_client = QdrantClient(url=cfg.QDRANT_URL)
        self.retriever = HybridRetriever(
            qdrant_client, cfg.QDRANT_COLLECTION, self.embedder
        )
        self.reranker = CrossEncoderReranker(cfg.RERANK_MODEL)
        self.llm = CitationAwareLLM(cfg.ANTHROPIC_API_KEY, cfg.LLM_MODEL)

        print("[INIT] Pipeline ready.")

    def ingest_pdf(self, pdf_path: str) -> int:
        print(f"\n[INGEST] Processing file: {pdf_path}")

        elements = self.pdf_parser.parse(pdf_path)
        elements = self.image_describer.describe_batch(elements)
        table_elements = self.table_parser.parse_tables(pdf_path)
        elements.extend(table_elements)

        chunks = self.chunker.chunk_elements(elements)
        print(f"[INGEST] {len(chunks)} chunks produced")

        self.vector_store.upsert_chunks(chunks, self.embedder)
        return len(chunks)

    def query(self, question: str) -> dict:
        print(f"\n[QUERY] Processing question: {question}")

        candidates = self.retriever.retrieve(question, top_k=cfg.TOP_K_RETRIEVAL)
        top_chunks = self.reranker.rerank(question, candidates, top_k=cfg.TOP_K_RERANK)

        result = self.llm.answer(
            question, top_chunks,
            max_context_tokens=cfg.MAX_CONTEXT_TOKENS,
            max_response_tokens=cfg.MAX_RESPONSE_TOKENS
        )

        print(f"[COST] ${result['total_cost_usd']} | "
              f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
        return result