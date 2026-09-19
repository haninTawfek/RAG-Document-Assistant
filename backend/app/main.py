"""FastAPI entrypoint. The vector store and the LLM client are loaded ONCE at
startup via the lifespan handler, never per request."""
from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query as query_routes
from app.core.config import get_settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.utils.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading vector store and LLM client...")
    try:
        app.state.retriever = Retriever(
            store_dir="data/vector_store2",
            collection_name="manuals_v2",
            embedding_model=settings.embedding_model,
        )
    except Exception as exc:                          # noqa: BLE001
        app.state.retriever = None
        logger.error("Could not load vector store: %s", exc)

    # تهيئة كلاس الـ Generator ليتوافق مع Groq بدلاً من Ollama
    app.state.generator = Generator(
        model="qwen/qwen3.8-27b"
    )
    logger.info("Startup complete - Groq generator ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG Document Assistant API",
    description="Ask questions about an indexed document collection and get grounded, cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_routes.router)


def process_pdf_background(file_path: str, retriever):
    """دالة معالجة الملف وتقطيعه وإضافته للـ Vector Store في الخلفية"""
    try:
        logger.info("Starting background processing for: %s", file_path)
        
        if retriever is not None:
            # استدعاء دالة الإضافة التي أضفناها في الـ Retriever
            retriever.add_pdf_document(file_path)
            logger.info("Finished indexing: %s. New total chunks: %d", file_path, retriever.collection.count())
        else:
            logger.error("Retriever instance is not initialized.")
            
    except Exception as exc:
        logger.error("Error processing document in background: %s", exc)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint لرفع ملف PDF جديد ومعالجته فوراً مع إرجاع العدد الجديد."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        # 1. حفظ الملف
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        
        logger.info("Uploaded file saved successfully: %s", file_path)
        
        # 2. التقطيع والإضافة مباشرة تنتظر حتى تنتهي عملية الـ Embedding
        if app.state.retriever is not None:
            app.state.retriever.add_pdf_document(file_path)
            new_total = app.state.retriever.collection.count()
            logger.info("Indexing finished! New total chunks: %d", new_total)
        else:
            raise HTTPException(status_code=500, detail="Retriever instance is not initialized.")

        return {
            "status": "success", 
            "filename": file.filename, 
            "total_chunks": new_total,
            "message": f"File indexed successfully! Total chunks: {new_total}"
        }
    except Exception as exc:
        logger.error("Failed to upload file: %s", exc)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "RAG Document Assistant", "docs": "/docs"}