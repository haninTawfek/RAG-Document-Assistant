"""API routes: GET /health and POST /query."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.schemas.query import QueryRequest, QueryResponse, SourceChunk
from app.services.generation import Generator
from app.services.retrieval import Retriever

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependencies read from app.state (populated once in the lifespan handler).
def get_retriever(request: Request) -> Retriever:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Vector store is not loaded.")
    return retriever


def get_generator(request: Request) -> Generator:
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(status_code=503, detail="LLM client is not ready.")
    return generator


@router.get("/health", tags=["system"])
def health(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)
    return {
        "status": "ok" if retriever is not None else "degraded",
        "vector_store_loaded": retriever is not None,
        "indexed_chunks": retriever.collection.count() if retriever else 0,
        "llm_available": generator.health() if generator else False,
        "model": getattr(generator, "model", "llama-3.1-8b-instant"),
    }


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> QueryResponse:
    started = time.perf_counter()
    question = payload.question.strip()
    logger.info("Query: %r", question)

    try:
        chunks = retriever.retrieve(question, k=settings.top_k)
    except Exception as exc:                          # noqa: BLE001
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc

    try:
        # تمرير الـ history لدالة answer
        answer = generator.answer(
            question=question, 
            chunks=chunks, 
            history=getattr(payload, "history", None)
        )
    except Exception as exc:                          # noqa: BLE001
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=502, detail="The language model is unavailable. Is Groq API Key set?"
        ) from exc

    return QueryResponse(
        answer=answer,
        sources=sorted({c.label for c in chunks}),
        chunks=[
            SourceChunk(
                label=c.label,
                document=c.document,
                page=c.page,
                score=round(c.score, 4),
                preview=c.text[:300].replace("\n", " ").strip(),
            )
            for c in chunks
        ],
        model=getattr(generator, "model", "llama-3.1-8b-instant"),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )