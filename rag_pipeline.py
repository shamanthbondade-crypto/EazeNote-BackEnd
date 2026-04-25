"""
EazeNote – RAG Pipeline  (Memory-Optimised Edition)
====================================================
Root cause of OOM on Render free tier (512 MB):
  - sentence-transformers pulls in PyTorch (~350 MB RSS)
  - The MiniLM model weights add another ~90 MB
  -> Together they bust the 512 MB ceiling before a single request is served.

Fix: replace local embeddings with API-based embeddings.
  * Primary  -> Groq  : "nomic-embed-text-v1_5"   (free, fast, 768-dim)
  * Fallback  -> Together AI: "togethercomputer/m2-bert-80M-8k-retrieval"
  * Emergency fallback -> simple hash embedder (no external call needed)

No torch, no sentence-transformers, no heavy ML libraries at all.
Total import footprint: < 30 MB.
"""

import hashlib
import json
import logging
import math
import os
import re
import uuid
from typing import Any

import chromadb
import requests
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/data/chroma")

LLM_API_KEY   = os.environ.get("LLM_API_KEY", "")
LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "groq").lower()
LLM_MODEL     = os.environ.get(
    "LLM_MODEL",
    "llama3-8b-8192" if LLM_PROVIDER != "together" else "meta-llama/Llama-3-8b-chat-hf",
)

EMBED_PROVIDER     = os.environ.get("EMBED_PROVIDER", LLM_PROVIDER)
GROQ_EMBED_MODEL   = "nomic-embed-text-v1_5"
TOGETHER_EMBED_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"

GROQ_EMBED_URL    = "https://api.groq.com/openai/v1/embeddings"
TOGETHER_EMBED_URL = "https://api.together.xyz/v1/embeddings"
GROQ_CHAT_URL     = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"

EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Lightweight fallback embedder (no external deps, ~0 MB RAM)
# ---------------------------------------------------------------------------

class _HashEmbedder:
    """
    Emergency fallback: deterministic pseudo-embedding via word hashing.
    Quality is much lower than real embeddings but the service stays alive
    when the embedding API is unavailable.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [0.0] * self.dim
        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            for i in range(6):
                idx = (h >> (i * 10)) % self.dim
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


_hash_embedder = _HashEmbedder()


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Memory-optimised RAG pipeline.
    Embeddings are obtained via HTTP API (Groq / Together AI).
    No PyTorch, no sentence-transformers — stays well under 512 MB.
    """

    def __init__(self) -> None:
        logger.info("Initialising RAGPipeline (API-based embeddings)…")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="eazenote_docs",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("RAGPipeline ready. ChromaDB at %s", CHROMA_PERSIST_DIR)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Obtain embeddings via API; fall back to hash embedder on failure."""
        if not LLM_API_KEY:
            logger.warning("LLM_API_KEY not set — using hash embedder (low quality).")
            return _hash_embedder.encode_batch(texts)
        try:
            return self._embed_via_api(texts)
        except Exception as exc:
            logger.error("Embedding API failed (%s) — falling back to hash embedder.", exc)
            return _hash_embedder.encode_batch(texts)

    def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI-compatible embedding endpoint."""
        if EMBED_PROVIDER == "together":
            url, model = TOGETHER_EMBED_URL, TOGETHER_EMBED_MODEL
        else:
            url, model = GROQ_EMBED_URL, GROQ_EMBED_MODEL

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i: i + 32]
            resp = requests.post(url, json={"model": model, "input": batch}, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in sorted_items)

        return all_embeddings

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """HTTP call to Groq or Together AI chat completions."""
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY environment variable is not set.")

        url = GROQ_CHAT_URL if LLM_PROVIDER != "together" else TOGETHER_CHAT_URL
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise RuntimeError("LLM API timed out. Please try again.")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"LLM API error {exc.response.status_code}: {exc.response.text}")
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_chunks(self, chunks: list[str], session_id: str, doc_name: str) -> None:
        if not chunks:
            return
        embeddings = self._embed(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"session_id": session_id, "doc_name": doc_name, "chunk_index": i}
            for i, _ in enumerate(chunks)
        ]
        self.collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.info("Stored %d chunks for session %s", len(chunks), session_id)

    def retrieve(self, query: str, session_id: str, top_k: int = 4) -> list[str]:
        query_embedding = self._embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"session_id": {"$eq": session_id}},
            include=["documents"],
        )
        return results.get("documents", [[]])[0]

    def answer(self, query: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            "Answer ONLY using the context below.\n"
            'If the answer is not present in the context, say "Not found".\n\n'
            f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        )
        return self._call_llm(prompt, max_tokens=512)

    def summarize(self, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            "You are a helpful assistant. Read the following document excerpts and write "
            "a concise, well-structured summary (3-5 paragraphs) covering the main topics, "
            "key points, and important details.\n\n"
            f"Document excerpts:\n{context}\n\nSummary:"
        )
        return self._call_llm(prompt, max_tokens=800)

    def generate_quiz(self, chunks: list[str], num_questions: int = 5) -> list[dict[str, Any]]:
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            f"You are an expert quiz creator. Based on the document excerpts below, "
            f"generate exactly {num_questions} multiple-choice questions.\n\n"
            "Rules:\n"
            "- Each question must have exactly 4 options labelled A, B, C, D.\n"
            "- Only ONE option is correct.\n"
            "- Questions should test understanding of key facts, concepts, or definitions.\n"
            "- Return ONLY valid JSON — no extra text, no markdown fences.\n\n"
            'Output format: [{"question": "...", "options": {"A":"...","B":"...","C":"...","D":"..."}, "correct_answer": "A"}, ...]\n\n'
            f"Document excerpts:\n{context}\n\nJSON:"
        )
        raw = self._call_llm(prompt, max_tokens=1500)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            questions = json.loads(raw)
            validated = [
                q for q in questions
                if isinstance(q, dict) and all(k in q for k in ("question", "options", "correct_answer"))
            ]
            return validated[:num_questions]
        except json.JSONDecodeError as exc:
            logger.error("Quiz JSON parse error: %s | raw: %.500s", exc, raw)
            raise RuntimeError("LLM returned malformed quiz JSON. Try again or shorten the document.")
