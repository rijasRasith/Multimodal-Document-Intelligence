from __future__ import annotations

import os
import time
import uuid
import json
from typing import Optional, Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from PIL import Image
import io

from data.schema import EXTRACTION_PROMPT, InvoiceSchema
from evaluation.metrics import parse_model_output
from retrieval.rag_pipeline import retrieve, compose_answer
from retrieval.embeddings import embed_text
from retrieval.vector_store import (
    get_connection,
    initialize_schema,
    upsert_document,
    insert_chunks,
    log_request,
    log_feedback,
)
from retrieval.ingest import ingest_pdf, ingest_image
from retrieval.embeddings import embed_batch

import mlflow

MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))
API_KEY = os.environ.get("API_KEY", "dev-secret-key")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
PGVECTOR_DSN = os.environ.get("PGVECTOR_DSN", "postgresql://postgres:password@localhost:5432/docint")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", None)

app = FastAPI(
    title="Multimodal Document Intelligence API",
    version="1.0.0",
    description="Invoice extraction and multimodal RAG over documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

_model = None
_processor = None
_db_conn = None


def get_api_key(key: str = Security(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")
    return key


def get_model():
    global _model, _processor
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return _model, _processor


def get_db():
    global _db_conn
    if _db_conn is None:
        raise HTTPException(status_code=503, detail="Database not available.")
    return _db_conn


@app.on_event("startup")
async def startup_event():
    global _model, _processor, _db_conn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from transformers import AutoProcessor, AutoModelForVision2Seq

    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if ADAPTER_PATH:
        from peft import PeftModel
        base = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    else:
        _model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    _model.eval()

    try:
        _db_conn = get_connection(PGVECTOR_DSN)
        initialize_schema(_db_conn)
    except Exception as e:
        print(f"[WARNING] Database connection failed: {e}. RAG features will be unavailable.")
        _db_conn = None


class ExtractionResponse(BaseModel):
    request_id: str
    result: Optional[dict] = None
    json_valid: bool
    schema_valid: bool
    schema_violations: list[str] = []
    processing_time_ms: float
    error: Optional[str] = None


class IngestResponse(BaseModel):
    request_id: str
    doc_id: str
    chunks_indexed: int
    processing_time_ms: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    doc_id_filter: Optional[str] = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    retrieved_chunks: int
    avg_retrieval_score: float
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    request_id: str
    endpoint: str
    rating: int = Field(..., ge=-1, le=1)
    corrected_fields: Optional[dict] = None
    original_output: Optional[dict] = None


class FeedbackResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    version: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        database_connected=_db_conn is not None,
        version="1.0.0",
    )


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract(
    file: UploadFile = File(...),
    api_key: str = Security(get_api_key),
):
    request_id = str(uuid.uuid4())
    start_ms = time.time() * 1000

    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    model, processor = get_model()

    try:
        if file.content_type == "application/pdf":
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            page = doc[0]
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        else:
            image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode file: {e}")

    def _run_inference(img: Image.Image) -> str:
        inputs = processor(text=EXTRACTION_PROMPT, images=img, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    raw_output = _run_inference(image)
    parsed = parse_model_output(raw_output)

    if parsed is None:
        raw_output = _run_inference(image)
        parsed = parse_model_output(raw_output)

    json_valid = parsed is not None
    schema_violations = []
    schema_valid = False

    if json_valid:
        try:
            validated = InvoiceSchema(**parsed)
            schema_valid = True
        except Exception as e:
            schema_violations = [str(e)]

    elapsed_ms = time.time() * 1000 - start_ms

    if _db_conn:
        try:
            null_count = sum(1 for f in ["vendor", "date", "invoice_number", "total"] if (parsed or {}).get(f) is None)
            log_request(_db_conn, {
                "request_id": request_id,
                "endpoint": "/extract",
                "latency_ms": elapsed_ms,
                "input_size_bytes": file_size,
                "json_valid": json_valid,
                "schema_violations": len(schema_violations),
                "null_field_count": null_count,
            })
        except Exception:
            pass

    return ExtractionResponse(
        request_id=request_id,
        result=parsed,
        json_valid=json_valid,
        schema_valid=schema_valid,
        schema_violations=schema_violations,
        processing_time_ms=round(elapsed_ms, 2),
        error=None if json_valid else "Model output could not be parsed as JSON.",
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    api_key: str = Security(get_api_key),
):
    request_id = str(uuid.uuid4())
    start_ms = time.time() * 1000

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    db = get_db()
    model, processor = get_model()
    doc_id = doc_id or str(uuid.uuid4())

    try:
        if file.content_type == "application/pdf":
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                from pathlib import Path
                nodes = ingest_pdf(Path(tmp_path), doc_id=doc_id, vlm_model=model, processor=processor)
            finally:
                _os.unlink(tmp_path)
        else:
            from pathlib import Path
            img = Image.open(io.BytesIO(content)).convert("RGB")
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            try:
                nodes = ingest_image(Path(tmp_path), doc_id=doc_id, vlm_model=model, processor=processor)
            finally:
                _os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {e}")

    texts_to_embed = [n["content"] for n in nodes]
    embeddings = embed_batch(texts_to_embed)
    for node, emb in zip(nodes, embeddings):
        node["embedding"] = emb

    upsert_document(db, doc_id, file.filename or "unknown")
    insert_chunks(db, nodes)

    elapsed_ms = time.time() * 1000 - start_ms
    return IngestResponse(
        request_id=request_id,
        doc_id=doc_id,
        chunks_indexed=len(nodes),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(
    body: AskRequest,
    api_key: str = Security(get_api_key),
):
    request_id = str(uuid.uuid4())
    start_ms = time.time() * 1000

    db = get_db()
    model, processor = get_model()

    retrieved = retrieve(
        conn=db,
        question=body.question,
        top_k=body.top_k,
        use_hybrid=body.use_hybrid,
        doc_id_filter=body.doc_id_filter,
    )

    if not retrieved:
        elapsed_ms = time.time() * 1000 - start_ms
        return AskResponse(
            request_id=request_id,
            answer="I do not have enough information to answer this question.",
            citations=[],
            retrieved_chunks=0,
            avg_retrieval_score=0.0,
            processing_time_ms=round(elapsed_ms, 2),
        )

    answer_data = compose_answer(
        question=body.question,
        retrieved_chunks=retrieved,
        vlm_model=model,
        processor=processor,
    )

    elapsed_ms = time.time() * 1000 - start_ms
    avg_score = answer_data.get("avg_retrieval_score", 0.0)

    if _db_conn:
        try:
            log_request(db, {
                "request_id": request_id,
                "endpoint": "/ask",
                "latency_ms": elapsed_ms,
                "input_size_bytes": len(body.question.encode()),
                "json_valid": True,
                "retrieval_score": avg_score,
            })
        except Exception:
            pass

    return AskResponse(
        request_id=request_id,
        answer=answer_data["answer"],
        citations=answer_data["citations"],
        retrieved_chunks=answer_data["retrieved_chunks"],
        avg_retrieval_score=avg_score,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(
    body: FeedbackRequest,
    api_key: str = Security(get_api_key),
):
    db = get_db()
    try:
        log_feedback(db, {
            "request_id": body.request_id,
            "endpoint": body.endpoint,
            "rating": body.rating,
            "corrected_fields": body.corrected_fields,
            "original_output": body.original_output,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")

    return FeedbackResponse(status="accepted")
