"""
RAG Retriever for ITAssist AI.

Loads the pre-built FAISS index and provides a search() function
that returns the top-k most relevant document chunks for a given query.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL, RAG_TOP_K

# Singleton — loaded once, reused across requests
_vectorstore: FAISS | None = None
_embeddings: HuggingFaceEmbeddings | None = None


import functools


@functools.lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vectorstore() -> FAISS | None:
    global _vectorstore
    if _vectorstore is None:
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        if not os.path.exists(index_file):
            return None
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def is_index_ready() -> bool:
    """Returns True if the FAISS index exists on disk."""
    return os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss"))


def search(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """
    Search the FAISS vector store for relevant chunks.
    Reads PDF page content dynamically at runtime when matched.

    Args:
        query: Natural language query string
        k: Number of top results to return

    Returns:
        List of dicts: [{content, source, score}, ...]
    """
    vs = _get_vectorstore()
    if vs is None:
        return []

    try:
        results = vs.similarity_search_with_score(query, k=k)
        output = []
        
        # Optimize PDF loading by opening the file reader once per query search
        pdf_reader = None
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "troubleshoot-windows-client.pdf",
        )
        if os.path.exists(pdf_path):
            import pypdf
            try:
                pdf_reader = pypdf.PdfReader(pdf_path)
            except Exception as e:
                print(f"Failed to open PDF reader: {e}")

        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source) if source else "Unknown"
            content = doc.page_content
            
            # If hit is a PDF page reference, fetch the actual full text dynamically
            if doc.metadata.get("is_pdf") or source_name.endswith(".pdf"):
                page_num = doc.metadata.get("page")
                if page_num is not None and pdf_reader is not None:
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text and page_text.strip():
                            content = (
                                f"[Full content read directly from {source_name} Page {page_num + 1} at query time]:\n"
                                f"{page_text.strip()}"
                            )
                    except Exception as e:
                        print(f"Failed to extract page {page_num} from PDF: {e}")
                        
            output.append({
                "content": content,
                "source": source_name,
                "score": float(score),
            })
        return output
    except Exception as e:
        print(f"RAG search error: {e}")
        return []


def search_text(query: str, k: int = RAG_TOP_K) -> str:
    """
    Convenience function — returns a single formatted string of all retrieved chunks.
    Used directly in LangGraph agent prompts.
    """
    results = search(query, k=k)
    if not results:
        return "No relevant documentation found."

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Source {i}: {r['source']}]\n{r['content']}"
        )
    return "\n\n---\n\n".join(sections)


def reload_index():
    """Force reload the FAISS index from disk (use after re-ingestion)."""
    global _vectorstore
    _vectorstore = None
    return _get_vectorstore() is not None


def add_to_faiss_index(filepath: str, content: str):
    """Dynamically add a new document to the existing FAISS index without full rebuild."""
    vs = _get_vectorstore()
    if vs is None:
        return False
        
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from config.settings import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, FAISS_INDEX_PATH
    
    doc = Document(page_content=content, metadata={"source": filepath})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    
    if chunks:
        vs.add_documents(chunks)
        vs.save_local(FAISS_INDEX_PATH)
        return True
    return False


# Category-to-SOP file mapping
SOP_FILENAMES = {
    "Performance": "performance_sop.txt",
    "VPN / Remote Access": "networking_vpn_sop.txt",
    "Network": "networking_vpn_sop.txt",
    "Email": "email_outlook_sop.txt",
    "Access / Permissions": "active_directory_access_sop.txt",
    "Software": "software_application_sop.txt",
    "Hardware": "hardware_device_sop.txt",
    "Backup / Storage": "backup_storage_sop.txt",
    "Printer": "printing_sop.txt",
    "Security / BitLocker": "security_bitlocker_sop.txt",
    "Windows Update": "windows_update_sop.txt",
    "Remote Desktop": "remote_desktop_teams_sop.txt",
    "Wireless / WiFi": "wireless_networking_sop.txt",
    "System Recovery": "system_recovery_sop.txt",
    "OneDrive / Cloud": "onedrive_sharepoint_sop.txt",
    "Power / Sleep": "power_management_sop.txt",
}


def parse_sop_file_locally(category: str, query: str) -> dict | None:
    """
    Locally parses the matching SOP text file to extract:
    - Best matching issue title
    - Specific resolution steps
    - Specific PowerShell diagnostic script
    Without calling any external LLMs.
    """
    import re
    from config.settings import SOP_DOCS_PATH
    filename = SOP_FILENAMES.get(category)
    if not filename:
        return None
        
    filepath = os.path.join(SOP_DOCS_PATH, filename)
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split file into individual sections using header dashed or equal lines
        blocks = re.split(r'[-=]{30,}\s*\n(?:DIAGNOSTIC STEP|SECTION|ISSUE)\s*\d+:\s*(.*?)\n\s*[-=]{30,}', content, flags=re.I)
        
        if len(blocks) < 3:
            return None
            
        best_title = f"{category} Issue"
        best_body = ""
        best_score = -1
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Iterate in pairs: blocks[i] is title, blocks[i+1] is body
        for i in range(1, len(blocks), 2):
            title = blocks[i].strip()
            body = blocks[i+1] if i+1 < len(blocks) else ""
            
            combined_text = f"{title}\n{body}"
            text_words = set(re.findall(r'\b\w+\b', combined_text.lower()))
            overlap = len(query_words.intersection(text_words))
            
            if overlap > best_score:
                best_score = overlap
                best_title = title
                best_body = body
                
        if best_score == -1:
            # Fallback to the first step block
            best_title = blocks[1].strip()
            best_body = blocks[2] if len(blocks) > 2 else ""
            
        # Parse the matched section:
        # 1. Title
        title = best_title
        
        # 2. Extract Resolution Steps
        steps = []
        res_section_match = re.search(r'Resolution Steps[^\n]*?:(.*?)(?:Diagnostic Commands|===|---|\Z)', best_body, re.DOTALL | re.IGNORECASE)
        if res_section_match:
            res_text = res_section_match.group(1).strip()
            # Extract lines starting with a digit like '1.', '2.'
            for line in res_text.split('\n'):
                line_cleaned = re.sub(r'^\s*\d+\.\s*', '', line).strip()
                if line_cleaned:
                    steps.append(line_cleaned)
        
        # Fallback if no formatted steps
        if not steps:
            steps = ["Perform standard category diagnostic procedures.", "Apply settings from Microsoft Troubleshooting SOP."]
            
        # 3. Extract PowerShell/Diagnostic Script
        script = ""
        diag_section_match = re.search(r'Diagnostic Commands[^\n]*?:(.*?)(?:Resolution Steps|===|---|\Z)', best_body, re.DOTALL | re.IGNORECASE)
        if diag_section_match:
            script_text = diag_section_match.group(1).strip()
            script = script_text
            
        if not script:
            # Default fallback script
            script = f"# Local diagnostic for {category}\nGet-Service -Name * | Select-Object Name, Status -First 10"
            
        return {
            "problem": title,
            "probable_cause": f"Locally identified {category} issue ({title}) in {filename}.",
            "severity": "Medium",
            "confidence": 0.85,
            "analysis": f"Matched keywords from query. Loaded troubleshooting data locally from {filename}.",
            "self_resolvable": True,
            "self_resolution_steps": steps,
            "script": script,
            "script_type": "PowerShell"
        }
    except Exception as e:
        print(f"Error parsing local SOP file: {e}")
        return None
