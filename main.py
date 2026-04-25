"""
EazeNote – AI Smart Notebook with RAG
Main FastAPI application entry point.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_pipeline import RAGPipeline
from utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_text,
    validate_file_size,
    validate_file_type,
)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    app.state.rag = RAGPipeline()
    yield
    # Cleanup (if needed) goes here


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EazeNote API",
    description="AI Smart Notebook backend with RAG pipeline (LLaMA + ChromaDB)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow any Vercel frontend (and local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eazenote.vercel.app"],          # Tighten to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Returns service liveness status."""
    return {"status": "ok", "service": "EazeNote API"}


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Accept a PDF or TXT file, extract + chunk its text, embed it,
    and persist the vectors in ChromaDB under the given session_id.

    Returns the session_id so the client can reference it later.
    """
    # --- Validate ---
    validate_file_type(file.filename)

    raw_bytes = await file.read()
    validate_file_size(raw_bytes)

    # Generate a session if the caller didn't supply one
    if not session_id:
        session_id = str(uuid.uuid4())

    # --- Extract text ---
    filename_lower = file.filename.lower()
    if filename_lower.endswith(".pdf"):
        text = extract_text_from_pdf(raw_bytes)
    else:
        text = extract_text_from_txt(raw_bytes)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    # --- Chunk ---
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text chunking produced no results.")

    # --- Embed + store ---
    rag: RAGPipeline = app.state.rag
    rag.store_chunks(chunks, session_id=session_id, doc_name=file.filename)

    return {
        "status": "success",
        "session_id": session_id,
        "document": file.filename,
        "chunks_stored": len(chunks),
    }


# ---------------------------------------------------------------------------
# Ask / Q&A endpoint
# ---------------------------------------------------------------------------

@app.get("/ask", tags=["Q&A"])
async def ask_question(
    query: str = Query(..., min_length=1, description="User question"),
    session_id: str = Query(..., description="Session identifier from /upload"),
    top_k: int = Query(default=4, ge=1, le=10),
):
    """
    Retrieve relevant chunks for the query and generate an answer via LLaMA.
    """
    rag: RAGPipeline = app.state.rag

    chunks = rag.retrieve(query, session_id=session_id, top_k=top_k)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found for this session. Please upload a document first.",
        )

    answer = rag.answer(query, chunks)
    return {
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "sources_used": len(chunks),
    }


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------

@app.get("/summary", tags=["Analysis"])
async def summarize_document(
    session_id: str = Query(..., description="Session identifier from /upload"),
):
    """
    Retrieve a broad sample of stored chunks and ask the LLM for a concise summary.
    """
    rag: RAGPipeline = app.state.rag

    # Pull a wide sample (top_k=8) to cover the document
    chunks = rag.retrieve("summarize the document", session_id=session_id, top_k=8)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this session.",
        )

    summary = rag.summarize(chunks)
    return {"session_id": session_id, "summary": summary}


# ---------------------------------------------------------------------------
# Quiz endpoint
# ---------------------------------------------------------------------------

@app.get("/quiz", tags=["Analysis"])
async def generate_quiz(
    session_id: str = Query(..., description="Session identifier from /upload"),
    num_questions: int = Query(default=5, ge=1, le=10),
):
    """
    Generate multiple-choice questions from the uploaded document content.
    Returns a JSON list with question, options (A-D), and correct_answer.
    """
    rag: RAGPipeline = app.state.rag

    chunks = rag.retrieve("key concepts facts definitions", session_id=session_id, top_k=8)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this session.",
        )

    quiz = rag.generate_quiz(chunks, num_questions=num_questions)
    return {"session_id": session_id, "questions": quiz}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
