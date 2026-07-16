"""
RAG Ingestion Pipeline for ITAssist AI.

Loads knowledge from:
  1. SOP .txt documents in rag/sop_documents/
  2. The user-provided PDF: troubleshoot-windows-client.pdf

Embeds using sentence-transformers (all-MiniLM-L6-v2) and saves a FAISS index.
Run this once before starting the app, or when documents change.
"""
import os
import sys

# Make sure stdout uses UTF-8 to handle print statement emojis on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import (
    FAISS_INDEX_PATH,
    SOP_DOCS_PATH,
    EMBEDDING_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)

# ── Path to the user-supplied Microsoft PDF ────────────────────────────────────
PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "troubleshoot-windows-client.pdf",
)


def load_sop_documents() -> list:
    """Load all .txt SOP documents from the sop_documents directory."""
    print(f"📄 Loading SOP documents from: {SOP_DOCS_PATH}")
    loader = DirectoryLoader(
        SOP_DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"   ✅ Loaded {len(docs)} SOP text documents")
    return docs


def load_pdf_document() -> list:
    """Load the Microsoft Windows troubleshooting PDF as page reference placeholders."""
    if not os.path.exists(PDF_PATH):
        print(f"   ⚠️  PDF not found at: {PDF_PATH} — skipping PDF ingestion")
        return []

    print(f"📰 Loading PDF page references: {PDF_PATH}")
    import pypdf
    from langchain_core.documents import Document
    
    reader = pypdf.PdfReader(PDF_PATH)
    docs = []
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text() or ""
        
        # Get first 300 characters as a semantic preview/summary of this page
        preview = text.strip()[:300].replace('\n', ' ')
        content = f"Windows client troubleshooting guide page {page_num + 1}. Keywords: {preview}"
        
        doc = Document(
            page_content=content,
            metadata={
                "source": PDF_PATH,
                "page": page_num,
                "is_pdf": True
            }
        )
        docs.append(doc)
        
    print(f"   ✅ Loaded {len(docs)} page reference placeholders from PDF")
    return docs


def split_documents(documents: list) -> list:
    """Split documents into chunks. Splits only SOP text docs, leaves PDF references as-is."""
    sop_docs = [d for d in documents if not d.metadata.get("is_pdf")]
    pdf_docs = [d for d in documents if d.metadata.get("is_pdf")]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    if sop_docs:
        sop_chunks = splitter.split_documents(sop_docs)
        chunks.extend(sop_chunks)
        print(f"✂️  Split SOPs into {len(sop_chunks)} chunks")
        
    if pdf_docs:
        chunks.extend(pdf_docs)
        print(f"📄 Kept {len(pdf_docs)} PDF page references intact")
        
    print(f"📊 Total chunks for indexing: {len(chunks)}")
    return chunks


def build_faiss_index(chunks: list) -> FAISS:
    """Embed chunks and build FAISS vector store."""
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("⚡ Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_index(vectorstore: FAISS):
    """Save FAISS index to disk."""
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"💾 FAISS index saved to: {FAISS_INDEX_PATH}")


def ingest(include_pdf: bool = True):
    """Main ingestion pipeline."""
    print("\n" + "=" * 60)
    print("  ITAssist AI — RAG Ingestion Pipeline (Dynamic Page Mode)")
    print("=" * 60)

    all_docs = []

    # Load SOP text documents
    sop_docs = load_sop_documents()
    all_docs.extend(sop_docs)

    # Load PDF page reference placeholders
    if include_pdf:
        pdf_docs = load_pdf_document()
        all_docs.extend(pdf_docs)

    if not all_docs:
        print("❌ No documents loaded. Check your file paths.")
        return

    # Split (splits only SOPs, keeps PDF intact)
    chunks = split_documents(all_docs)

    # Build and save FAISS index
    vectorstore = build_faiss_index(chunks)
    save_index(vectorstore)

    print("\n✅ Ingestion complete! FAISS index ready for queries.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    ingest()
