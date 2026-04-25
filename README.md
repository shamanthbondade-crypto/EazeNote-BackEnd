# EazeNote – AI Smart Notebook Backend

> **FastAPI + ChromaDB + sentence-transformers + LLaMA (Groq / Together AI)**

A production-ready Retrieval-Augmented Generation (RAG) backend that powers the EazeNote AI Smart Notebook. Upload PDF or TXT documents and instantly ask questions, get summaries, or generate quizzes — all grounded in your own content.

---

## Table of Contents

1. [Architecture](#architecture)
2. [API Reference](#api-reference)
3. [Local Development](#local-development)
4. [Deployment on Render](#deployment-on-render)
5. [Environment Variables](#environment-variables)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)

---

## Architecture

```
Client (Vercel frontend)
        │  HTTP
        ▼
┌───────────────────┐
│   FastAPI (main)  │
│  CORS Middleware  │
└────────┬──────────┘
         │
    ┌────┴─────────────────────────┐
    │        RAG Pipeline          │
    │  sentence-transformers       │  ← Embedding model (MiniLM-L6-v2)
    │  ChromaDB (persistent)       │  ← Vector store (/data/chroma)
    │  Groq / Together AI (HTTP)   │  ← LLaMA 3 inference
    └──────────────────────────────┘
```

### RAG Flow

```
Document Upload                 Question Answering
─────────────                   ──────────────────
PDF / TXT                       Query
   │                               │
   ▼                               ▼
Extract Text                   Embed Query
   │                               │
   ▼                               ▼
Chunk (500 tok, 50 overlap)    ChromaDB Similarity Search
   │                               │
   ▼                               ▼
Embed (MiniLM)                 Top-K Chunks
   │                               │
   ▼                               ▼
Upsert ChromaDB            Build Prompt + Call LLaMA
(tagged with session_id)           │
                                   ▼
                              JSON Answer
```

---

## API Reference

All endpoints return JSON. Base URL on Render: `https://<your-service>.onrender.com`

### `GET /health`

Returns service liveness.

```json
{ "status": "ok", "service": "EazeNote API" }
```

---

### `POST /upload`

Upload a PDF or TXT document and index it for RAG.

| Field        | Type            | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| `file`      | `multipart/form-data` | PDF or TXT file (max 10 MB)            |
| `session_id`| `string` (optional) | Reuse an existing session or omit to auto-generate |

**Response**

```json
{
  "status": "success",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "document": "lecture_notes.pdf",
  "chunks_stored": 42
}
```

---

### `GET /ask`

Ask a question grounded in the uploaded document.

| Param        | Type   | Description                    |
|-------------|--------|--------------------------------|
| `query`     | string | Your question                  |
| `session_id`| string | Session from `/upload`         |
| `top_k`     | int    | Chunks to retrieve (default 4) |

**Response**

```json
{
  "session_id": "3fa85f64...",
  "query": "What is photosynthesis?",
  "answer": "Photosynthesis is the process by which plants convert sunlight...",
  "sources_used": 4
}
```

---

### `GET /summary`

Summarize the entire uploaded document.

| Param        | Type   | Description            |
|-------------|--------|------------------------|
| `session_id`| string | Session from `/upload` |

**Response**

```json
{
  "session_id": "3fa85f64...",
  "summary": "This document covers..."
}
```

---

### `GET /quiz`

Generate multiple-choice questions from the document.

| Param           | Type   | Description                      |
|----------------|--------|----------------------------------|
| `session_id`   | string | Session from `/upload`           |
| `num_questions`| int    | Number of MCQs (default 5, max 10)|

**Response**

```json
{
  "session_id": "3fa85f64...",
  "questions": [
    {
      "question": "What is the powerhouse of the cell?",
      "options": { "A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Vacuole" },
      "correct_answer": "B"
    }
  ]
}
```

---

## Local Development

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free) or [Together AI key](https://www.together.ai)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/eazenote-backend.git
cd eazenote-backend

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your LLM_API_KEY

# 5. Run the server
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

Visit [http://localhost:10000/docs](http://localhost:10000/docs) for the interactive Swagger UI.

---

## Deployment on Render

### Step 1 – Create the Web Service

1. Push this repo to GitHub.
2. Log in to [Render](https://render.com) and click **New → Web Service**.
3. Connect your GitHub repository.

### Step 2 – Configure the Service

| Setting          | Value                                             |
|-----------------|---------------------------------------------------|
| **Runtime**      | Python 3                                         |
| **Build Command**| `pip install -r requirements.txt`                |
| **Start Command**| `uvicorn main:app --host 0.0.0.0 --port 10000`   |

### Step 3 – Add Environment Variables

In the Render dashboard → **Environment** tab, add:

| Key                 | Value                        |
|--------------------|------------------------------|
| `LLM_API_KEY`      | `your_groq_or_together_key`  |
| `LLM_PROVIDER`     | `groq`                       |
| `CHROMA_PERSIST_DIR` | `/data/chroma`             |

### Step 4 – Add a Persistent Disk (Required for ChromaDB)

1. In your service → **Disks** tab → **Add Disk**.
2. Set **Mount Path** to `/data` and choose a size (1 GB is enough to start).

> ⚠️ Without a persistent disk, ChromaDB data is lost on every redeploy.

### Step 5 – Deploy

Click **Deploy** — Render will install dependencies and start the service. Your API will be live at `https://<service-name>.onrender.com`.

---

## Environment Variables

| Variable             | Required | Default                              | Description                                    |
|---------------------|----------|--------------------------------------|------------------------------------------------|
| `LLM_API_KEY`       | ✅ Yes   | –                                    | API key for Groq or Together AI                |
| `LLM_PROVIDER`      | No       | `groq`                               | `"groq"` or `"together"`                       |
| `LLM_MODEL`         | No       | `llama3-8b-8192` (Groq)             | Model name override                            |
| `CHROMA_PERSIST_DIR`| No       | `/data/chroma`                       | ChromaDB storage path                          |
| `PORT`              | No       | `10000`                              | Server port (injected by Render automatically) |

---

## Project Structure

```
backend/
├── main.py           # FastAPI app, routes, middleware
├── rag_pipeline.py   # RAG logic: embed, store, retrieve, LLM calls
├── utils.py          # File validation, text extraction, chunking
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md         # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `LLM_API_KEY not set` | Add the key to your `.env` or Render environment variables |
| ChromaDB data lost on redeploy | Attach a Render persistent disk at `/data` |
| `422 Unprocessable Entity` on upload | Check file is PDF/TXT and under 10 MB |
| LLM returns garbled quiz JSON | Reduce `num_questions` or upload a shorter document |
| Slow cold start on Render free tier | First request downloads the MiniLM model (~90 MB); subsequent requests are fast |
| `torch` install fails | Ensure the `--extra-index-url` line in `requirements.txt` is intact |

---

## License

MIT © EazeNote
