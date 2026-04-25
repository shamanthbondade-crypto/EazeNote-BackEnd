"""
EazeNote – RAG Pipeline
Handles embedding, vector storage (ChromaDB), retrieval, and LLM calls.
"""
import torch
# Force PyTorch to use only 1 CPU thread to save massive amounts of RAM
torch.set_num_threads(1)

import json
import logging
import os
import re
import uuid
from typing import Any

import chromadb
import requests
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/data/chroma")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM provider config
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()   # "groq" | "together"
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "llama3-8b-8192" if LLM_PROVIDER == "groq" else "meta-llama/Llama-3-8b-chat-hf",
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


# ---------------------------------------------------------------------------
# RAGPipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Encapsulates:
    - Sentence-Transformers embeddings
    - ChromaDB vector store with persistence
    - LLaMA-based generation via Groq / Together AI
    """

    def __init__(self) -> None:
        logger.info("Initializing RAGPipeline…")

        # Embedding model (downloads on first run, cached after)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # ChromaDB persistent client
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # We use a single collection; session_id is stored as metadata
        self.collection = self.chroma_client.get_or_create_collection(
            name="eazenote_docs",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("RAGPipeline ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return normalized embeddings for a list of strings."""
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call the configured LLM provider (Groq or Together AI).
        Uses only the requests library – no vendor SDK required.
        """
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY environment variable is not set.")

        api_url = GROQ_API_URL if LLM_PROVIDER == "groq" else TOGETHER_API_URL

        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise RuntimeError("LLM API timed out. Please try again.")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"LLM API error {exc.response.status_code}: {exc.response.text}")
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        chunks: list[str],
        session_id: str,
        doc_name: str,
    ) -> None:
        """
        Embed text chunks and upsert them into ChromaDB.
        Each document is identified by a unique ID (uuid) and tagged with
        session_id + doc_name in its metadata.
        """
        if not chunks:
            return

        embeddings = self._embed(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"session_id": session_id, "doc_name": doc_name, "chunk_index": i}
            for i, _ in enumerate(chunks)
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for session %s", len(chunks), session_id)

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 4,
    ) -> list[str]:
        """
        Embed the query and retrieve the top_k most similar chunks
        belonging to the given session_id.
        """
        query_embedding = self._embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"session_id": {"$eq": session_id}},
            include=["documents"],
        )

        docs: list[str] = results.get("documents", [[]])[0]
        return docs

    def answer(self, query: str, chunks: list[str]) -> str:
        """
        Build a RAG prompt from retrieved chunks and return the LLM answer.
        """
        context = "\n\n---\n\n".join(chunks)
        prompt = f"""Answer ONLY using the context below.
If the answer is not present in the context, say "Not found".

Context:
{context}

Question:
{query}

Answer:"""

        return self._call_llm(prompt, max_tokens=512)

    def summarize(self, chunks: list[str]) -> str:
        """
        Ask the LLM to summarize the content captured in the retrieved chunks.
        """
        context = "\n\n---\n\n".join(chunks)
        prompt = f"""You are a helpful assistant. Read the following document excerpts and write a concise, well-structured summary (3-5 paragraphs) covering the main topics, key points, and important details.

Document excerpts:
{context}

Summary:"""

        return self._call_llm(prompt, max_tokens=800)

    def generate_quiz(self, chunks: list[str], num_questions: int = 5) -> list[dict[str, Any]]:
        """
        Generate multiple-choice questions (MCQs) from the document content.
        Returns a list of dicts: {question, options, correct_answer}.
        """
        context = "\n\n---\n\n".join(chunks)
        prompt = f"""You are an expert quiz creator. Based on the document excerpts below, generate exactly {num_questions} multiple-choice questions.

Rules:
- Each question must have exactly 4 options labeled A, B, C, D.
- Only ONE option is correct.
- Questions should test understanding of key facts, concepts, or definitions from the text.
- Return ONLY valid JSON — no extra text, no markdown fences.

Output format (strict JSON array):
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A"
  }},
  ...
]

Document excerpts:
{context}

JSON:"""

        raw = self._call_llm(prompt, max_tokens=1500)

        # Strip accidental markdown fences if the LLM adds them
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            questions = json.loads(raw)
            # Basic structural validation
            validated = []
            for q in questions:
                if (
                    isinstance(q, dict)
                    and "question" in q
                    and "options" in q
                    and "correct_answer" in q
                ):
                    validated.append(q)
            return validated[:num_questions]
        except json.JSONDecodeError as exc:
            logger.error("Quiz JSON parse error: %s\nRaw: %s", exc, raw[:500])
            raise RuntimeError(
                "LLM returned malformed quiz JSON. Try again or reduce document size."
            )
