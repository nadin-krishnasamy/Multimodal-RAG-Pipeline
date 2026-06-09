# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile, shutil, os
from pipeline import MultimodalRAGPipeline

app = FastAPI(
    title="Multimodal RAG API",
    description="Production RAG over PDFs with citations",
    version="1.0.0"
)

pipeline = MultimodalRAGPipeline()

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest", summary="Upload and index a PDF")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        count = pipeline.ingest_pdf(tmp_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/query", summary="Query the indexed documents")
async def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = pipeline.query(req.question)
    return JSONResponse(content=result)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "multimodal-rag-v1"}

@app.get("/collection-info")
def collection_info():
    return pipeline.vector_store.get_collection_info()