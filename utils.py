"""
EazeNote – Utility Functions
File validation, text extraction, and text chunking helpers.
"""

import re
from io import BytesIO

import fitz  # PyMuPDF
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_file_type(filename: str) -> None:
    """Raise 415 if the file extension is not in ALLOWED_EXTENSIONS."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )


def validate_file_size(content: bytes) -> None:
    """Raise 413 if the file exceeds MAX_FILE_SIZE_BYTES."""
    if len(content) > MAX_FILE_SIZE_BYTES:
        limit_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {limit_mb} MB.",
        )


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract all text from a PDF's pages using PyMuPDF.
    Returns a single string with pages separated by newlines.
    """
    try:
        doc = fitz.open(stream=BytesIO(content), filetype="pdf")
        pages_text = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                pages_text.append(page_text)
        doc.close()
        return "\n\n".join(pages_text)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse PDF: {exc}",
        )


def extract_text_from_txt(content: bytes) -> str:
    """
    Decode a TXT file, trying UTF-8 first then falling back to latin-1.
    """
    for encoding in ("utf-8", "latin-1", "utf-16"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(
        status_code=422,
        detail="Could not decode text file. Please ensure it is a valid UTF-8 or latin-1 encoded file.",
    )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _approximate_token_count(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (GPT-style heuristic).
    Good enough for chunking purposes without importing a tokenizer.
    """
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split `text` into overlapping chunks where each chunk is approximately
    `chunk_size` tokens (estimated at 4 chars/token).

    Strategy:
    1. Split on sentence boundaries to preserve semantic units.
    2. Pack sentences into chunks until the size limit is reached.
    3. Carry the last `overlap` tokens from the previous chunk into the next.

    Args:
        text:        Raw document text.
        chunk_size:  Target chunk size in approximate tokens.
        overlap:     Number of tokens to overlap between adjacent chunks.

    Returns:
        List of non-empty text chunks.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split into sentences (keep delimiter)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    char_limit = chunk_size * 4        # approx characters per chunk
    overlap_chars = overlap * 4        # approx overlap in characters

    chunks: list[str] = []
    current_chunk = ""
    carry_over = ""                    # overlap text from previous chunk

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        prospective = (carry_over + " " + current_chunk + " " + sentence).strip()

        if _approximate_token_count(prospective) > chunk_size and current_chunk:
            # Flush current chunk
            chunks.append(current_chunk.strip())
            # Build carry-over from the tail of the flushed chunk
            carry_over = current_chunk[-overlap_chars:] if overlap_chars else ""
            current_chunk = sentence
        else:
            current_chunk = prospective if not current_chunk else current_chunk + " " + sentence

    # Flush remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very short chunks (likely noise)
    chunks = [c for c in chunks if len(c) > 40]

    return chunks
