"""
Central configuration for ITAssist AI.
Reads settings from environment variables or .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── FAISS ────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "faiss_index"
)

# ── SOP Documents ─────────────────────────────────────────────────────────────
SOP_DOCS_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "rag", "sop_documents"
)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "itassist.db"
)
DATABASE_URL: str = f"sqlite:///{DB_PATH}"

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE: int = 800
RAG_CHUNK_OVERLAP: int = 100
RAG_TOP_K: int = 3

# ── Issue Categories ──────────────────────────────────────────────────────────
ISSUE_CATEGORIES = [
    "Performance",
    "VPN / Remote Access",
    "Network",
    "Email",
    "Access / Permissions",
    "Software",
    "Hardware",
    "Backup / Storage",
    "Printer",
    "Security / BitLocker",
    "Windows Update",
    "Remote Desktop",
    "Wireless / WiFi",
    "System Recovery",
    "OneDrive / Cloud",
    "Power / Sleep",
    "Other",
]
