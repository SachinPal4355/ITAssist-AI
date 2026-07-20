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
        try:
            _vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Error loading local FAISS index: {e}")
            _vectorstore = None
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
            
        # Split file into individual sections, capturing the header type and title
        blocks = re.split(r'[-=]{30,}\s*\n(DIAGNOSTIC STEP|SECTION|ISSUE)\s*\d+:\s*(.*?)\n\s*[-=]{30,}', content, flags=re.I)
        
        if len(blocks) < 4:
            return None
            
        best_title = f"{category} Issue"
        best_body = ""
        best_score = -1
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Iterate in triplets: blocks[i] is type, blocks[i+1] is title, blocks[i+2] is body
        for i in range(1, len(blocks), 3):
            header_type = blocks[i].strip().upper()
            title = blocks[i+1].strip()
            body = blocks[i+2] if i+2 < len(blocks) else ""
            
            # Skip general overview sections
            if header_type == "SECTION":
                continue
                
            title_words = set(re.findall(r'\b\w+\b', title.lower()))
            body_words = set(re.findall(r'\b\w+\b', body.lower()))
            
            title_overlap = len(query_words.intersection(title_words))
            body_overlap = len(query_words.intersection(body_words))
            
            # Give 3x weight to title matches
            score = (title_overlap * 3) + body_overlap
            
            if score > best_score and score > 0:
                best_score = score
                best_title = title
                best_body = body
                
        if best_score == -1:
            # Fallback to the first actual troubleshooting block
            fallback_found = False
            for i in range(1, len(blocks), 3):
                header_type = blocks[i].strip().upper()
                if header_type != "SECTION":
                    best_title = blocks[i+1].strip()
                    best_body = blocks[i+2] if i+2 < len(blocks) else ""
                    fallback_found = True
                    break
            if not fallback_found:
                best_title = blocks[2].strip() if len(blocks) > 2 else f"{category} Issue"
                best_body = blocks[3] if len(blocks) > 3 else ""
            
        # Parse the matched section:
        # 1. Title
        title = best_title
        
        # 2. Extract Resolution Steps
        steps = []
        res_section_match = re.search(r'(?:Resolution Steps|Resolution)[^\n]*?:(.*?)(?:Diagnostic Commands|PowerShell Cleanup Script|Verification Command|===|---|\Z)', best_body, re.DOTALL | re.IGNORECASE)
        if res_section_match:
            res_text = res_section_match.group(1).strip()
            current_step = ""
            for line in res_text.split('\n'):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Check if it starts with a step number: e.g. "1.", "2."
                if re.match(r'^\d+\.\s*', line_stripped):
                    if current_step:
                        steps.append(current_step)
                    current_step = re.sub(r'^\d+\.\s*', '', line_stripped)
                else:
                    if current_step:
                        # Indent sub-bullets and code elements cleanly with HTML tags
                        current_step += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<code>{line_stripped}</code>" if (line_stripped.startswith("-") or "Get-" in line_stripped or "manage-bde" in line_stripped) else f"<br>{line_stripped}"
                    else:
                        current_step = line_stripped
            if current_step:
                steps.append(current_step)
        
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
