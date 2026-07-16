"""
utils/file_reader.py

Handles reading uploaded files (PDF, DOCX, TXT, images) and performing
in-memory semantic search on their content.

Design:
- Files are NEVER written to disk
- Embedding model is reused from rag.retriever (already warm)
- Semantic search is cosine similarity over small chunks — O(n_chunks)
"""
import io
import base64
import os
import re
import numpy as np
from typing import Optional

from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(uploaded_file) -> tuple[str, str]:
    """
    Extract text content from an uploaded Streamlit file object.

    Returns:
        (extracted_text, method_used)
        method_used is one of: "pdf", "docx", "txt", "image"
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return _read_pdf(file_bytes), "pdf"
    elif filename.endswith(".docx"):
        return _read_docx(file_bytes), "docx"
    elif filename.endswith(".txt") or filename.endswith(".log") or filename.endswith(".csv"):
        return _read_text(file_bytes), "txt"
    elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return _read_image_via_groq(file_bytes, uploaded_file.name), "image"
    else:
        # Fallback: try UTF-8 decode
        try:
            return file_bytes.decode("utf-8", errors="ignore"), "txt"
        except Exception:
            return "[Unable to read file content]", "unknown"


def _read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
        return "\n\n".join(text_parts) if text_parts else "[PDF has no extractable text]"
    except ImportError:
        return "[pdfplumber not installed — cannot read PDF]"
    except Exception as e:
        return f"[PDF read error: {e}]"


def _read_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs) if paragraphs else "[DOCX has no text content]"
    except ImportError:
        return "[python-docx not installed — cannot read DOCX]"
    except Exception as e:
        return f"[DOCX read error: {e}]"


def _read_text(file_bytes: bytes) -> str:
    """Decode text file bytes."""
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"[Text read error: {e}]"


def _read_image_via_groq(file_bytes: bytes, filename: str) -> str:
    """
    Send image to Groq vision model to extract a textual description.
    Uses llama-4-scout (vision-capable) with base64 encoding.
    Returns: "CONFIDENCE: <x>%\nDESCRIPTION: <desc>"
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        b64 = base64.b64encode(file_bytes).decode("utf-8")

        # Detect MIME type
        ext = filename.lower().split(".")[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp", "bmp": "image/bmp"}
        mime = mime_map.get(ext, "image/png")

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are an IT support assistant. Analyze this screenshot or image.\n"
                                "1. Evaluate the readability and relevance of this image for IT troubleshooting, and provide a Confidence Score (0-100%). "
                                "If the image is blurry, empty, unreadable, or completely unrelated to IT support (e.g., photo of food, pets, scenery), the confidence must be below 50%.\n"
                                "2. Extract and describe in detail: any error codes, warning messages, dialog text, application names, and technical details. Do not include passwords or personal credentials.\n"
                                "Format your response EXACTLY as:\n"
                                "CONFIDENCE: <number>%\n"
                                "DESCRIPTION: <description>"
                            ),
                        },
                    ],
                }
            ],
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()
        # Fallback formatting if model misses the exact syntax
        if "CONFIDENCE:" not in content:
            content = f"CONFIDENCE: 90%\nDESCRIPTION: {content}"
        return content
    except Exception as e:
        return f"CONFIDENCE: 0%\nDESCRIPTION: [Image vision error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Semantic Search on Attached Documents
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    Time: O(n)  Space: O(n)
    """
    if not text or len(text) < chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def semantic_search_in_doc(query: str, doc_text: str, top_k: int = 3) -> str:
    """
    Perform in-memory semantic search on a document string.

    Reuses the already-warm HuggingFace embedding model from rag.retriever.
    Time: O(n_chunks × d)  Space: O(n_chunks × 384)
    Returns the top-k most relevant chunks joined as a string.
    """
    if not doc_text or not doc_text.strip():
        return ""

    chunks = chunk_text(doc_text, chunk_size=500, overlap=50)
    if not chunks:
        return ""

    # If document is tiny, just return it directly — no need for search
    if len(chunks) <= top_k:
        return doc_text.strip()

    try:
        from rag.retriever import _get_embeddings
        embedder = _get_embeddings()

        # Encode all chunks + query (batch for efficiency)
        all_texts = chunks + [query]
        embeddings = embedder.embed_documents(all_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        chunk_embeddings = embeddings[:-1]           # All but last
        query_embedding = embeddings[-1].reshape(1, -1)  # Last = query

        # Cosine similarity: embeddings are normalized (normalize_embeddings=True)
        scores = (chunk_embeddings @ query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_chunks = [chunks[i] for i in sorted(top_indices)]  # Sort by position

        return "\n...\n".join(top_chunks)

    except Exception:
        # Fallback: return first 1500 chars if embedding fails
        return doc_text[:1500]


def strip_pii_patterns(text: str) -> str:
    """
    Remove obvious PII patterns from extracted text as a pre-Guardian safety net.
    This is a lightweight regex pass — Guardian does the deep check.
    """
    # Passwords in common formats
    text = re.sub(r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+", r"\1=[REDACTED]", text)
    # SSNs
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", text)
    # Credit card numbers (basic)
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CC-REDACTED]", text)
    # API keys / tokens (long alphanumeric)
    text = re.sub(r"\b[A-Za-z0-9_-]{32,}\b", "[TOKEN-REDACTED]", text)
    return text
