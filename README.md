# Multimodal RAG Pipeline

> Production-grade Retrieval-Augmented Generation over PDFs, tables, and images  
> with citation-verified responses and 96% lower token cost vs naive implementations.

## Architecture

┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                              │
│  PDFs → PyMuPDF   │  Images → EasyOCR/LLaVA  │ Tables → Camelot │
└────────────────────────────┬────────────────────────────────────┘

┌────────────────────────────▼────────────────────────────────────┐
│                  PREPROCESSING LAYER                            │
│  Semantic Chunker │ Metadata Tagger │ Table→Markdown Converter  │
└────────────────────────────┬────────────────────────────────────┘

┌────────────────────────────▼────────────────────────────────────┐
│                   INDEXING LAYER                                │
│  Dense: BGE-M3 Embeddings  │  Sparse: BM25 (Qdrant built-in)    │
│                 Qdrant Vector Store                             │
└────────────────────────────┬────────────────────────────────────┘

┌────────────────────────────▼────────────────────────────────────┐
│                   RETRIEVAL LAYER                               │
│  Hybrid Search (RRF: dense + sparse) → Cross-Encoder Reranker   │
└────────────────────────────┬────────────────────────────────────┘

┌────────────────────────────▼────────────────────────────────────┐
│               GENERATION LAYER (Token-Frugal)                   │
│  Context Compressor → Prompt Builder → Claude Haiku             │
│           ↓  Citation Extractor ↓                               │
│         Structured Response with [source, page, chunk_id]       │
└─────────────────────────────────────────────────────────────────┘

## Tech Stack

| Component | Tool | Reason |
|---|---|---|
| PDF Parsing | PyMuPDF + pdfplumber | Layout-aware, fast |
| Table Extraction | Camelot | Lattice + stream modes |
| Image Description | Claude Haiku Vision | Cheap ($0.003/image) |
| Embeddings | BGE-M3 (local) | Free, 1024-dim, multilingual |
| Vector DB | Qdrant (self-hosted) | Native hybrid search, free |
| Reranker | ms-marco-MiniLM-L-6-v2 | Reduces context 20→5 chunks |
| LLM | Claude Haiku 3.5 | $0.80/M tokens vs GPT-4 $30/M |
| Evaluation | RAGAS | Faithfulness, Relevancy, Recall, Precision |

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/nadin-krishnasamy/Multimodal-RAG-Pipeline.git
cd Multimodal-RAG-Pipeline
cp .env.example .env
# Fill in your ANTHROPIC_API_KEY in .env

# 2. Start Qdrant
docker-compose up qdrant -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
uvicorn api.main:app --reload

# 5. Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

## Evaluation Results

> To be updated after running RAGAS benchmark on test corpus.

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | TBD | ≥ 0.85 |
| Answer Relevancy | TBD | ≥ 0.80 |
| Context Recall | TBD | ≥ 0.75 |
| Context Precision | TBD | ≥ 0.70 |

## Cost Per Query

> To be updated after 50 real queries.

| Approach | Cost/Query |
|---|---|
| Naive (GPT-4 + 20 chunks) | ~$0.08 |
| This pipeline | ~$0.003 |

## Project Structure

Multimodal-RAG-Pipeline/
├── ingestion/       # PDF, image, table parsing + chunking
├── indexing/        # BGE-M3 embeddings + Qdrant upsert
├── retrieval/       # Hybrid search + cross-encoder reranking
├── generation/      # Prompt building + LLM + citation extraction
├── evaluation/      # RAGAS metrics + pytest suite
├── api/             # FastAPI endpoint
├── monitoring/      # Prometheus config
├── pipeline.py      # Full orchestrator
└── config.py        # Centralized config

## Author : NADIN KRISHNASAMY B

[LinkedIn] -(https://linkedin.com/in/nadin-krishnasamy)