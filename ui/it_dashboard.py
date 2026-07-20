"""
IT Engineer Dashboard — Full ticket management with AI analysis views.
Tabs: Overview | All Tickets | Ticket Detail | Knowledge Base
"""
import streamlit as st
from ui.components import (
    render_metric_card,
    render_ticket_card,
    render_page_header,
    render_info_banner,
    render_resolution_steps,
    render_script_block,
    SEVERITY_COLORS,
    STATUS_COLORS,
    CATEGORY_ICONS,
)
from database.crud import (
    get_all_tickets,
    get_ticket_by_id,
    update_ticket_status,
    get_ticket_stats,
    get_all_knowledge_articles,
)
from agents.resolution_agent import generate_resolution
from rag.retriever import is_index_ready

@st.cache_data(show_spinner=False)
def get_cached_resolution(category: str, issue_summary: str, probable_cause: str, severity: str, sop_context: str) -> dict:
    return generate_resolution(
        category=category,
        issue_summary=issue_summary,
        probable_cause=probable_cause,
        severity=severity,
        sop_context=sop_context,
    )


def check_pdf_relevance_via_groq(sample_text: str) -> tuple[bool, str]:
    """Verify if the uploaded document text is relevant to IT Infrastructure Troubleshooting."""
    import os
    import json
    from groq import Groq
    from config.settings import GROQ_API_KEY, GROQ_MODEL
    
    key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    client = Groq(api_key=key)
    
    system_prompt = (
        "You are an IT infrastructure auditor. Analyze the following document text sample and determine if the document "
        "is relevant to IT Infrastructure Troubleshooting, Windows troubleshooting, networking, VPN, printer support, Active Directory, "
        "or software configuration. "
        "Respond in strict JSON format: {\"relevant\": true/false, \"reason\": \"brief explanation\"}."
    )
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document text sample:\n{sample_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=150
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("relevant", False), result.get("reason", "Unknown verification outcome")
    except Exception as e:
        return True, f"Bypassed check due to verification error: {e}"


def _render_full_ticket_detail_view(ticket_id: str):
    # Back button
    if st.button("← Back to Dashboard", key="back_to_dashboard_btn"):
        st.session_state.selected_ticket_id = None
        st.rerun()

    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        st.error("Ticket not found.")
        st.session_state.selected_ticket_id = None
        st.rerun()

    # Page Header
    st.markdown(
        f"""
        <div style="background:#171717; border:1px solid #2f2f2f;
             border-radius:12px; padding:20px; margin-bottom:16px;">
            <div style="font-size:18px; font-weight:700; color:#ececec; margin-bottom:8px;">
                 Ticket Detail: {ticket['ticket_id']}
            </div>
            <div style="color:#b4b4b4; font-size:13px;">
                Detailed engineering view for {ticket['username']}'s issue
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Full ticket card
    render_ticket_card(ticket, compact=False)

    # Status update
    st.markdown(
        '<div style="color:#94a3b8; font-size:13px; font-weight:600; margin:16px 0 8px;"> Update Status:</div>',
        unsafe_allow_html=True,
    )
    status_cols = st.columns(4)
    statuses = ["In Progress", "Resolved", "Escalated", "Cancelled"]
    status_icons = {"In Progress": "", "Resolved": "", "Escalated": "", "Cancelled": ""}
    for i, status in enumerate(statuses):
        with status_cols[i]:
            if st.button(
                f"{status_icons[status]} {status}",
                key=f"status_view_{ticket_id}_{status}",
                use_container_width=True,
            ):
                update_ticket_status(ticket_id, status)
                st.success(f"Ticket status updated to **{status}**")
                st.rerun()

    st.markdown("---")

    # Conversation History
    convos = ticket.get("conversations", [])
    if convos:
        with st.expander(" User Conversation History", expanded=False):
            for msg in convos:
                render_chat_message = _local_chat_msg
                render_chat_message(msg["role"], msg["message"])

    email_draft = ticket.get("resolution_notes", "")
    if email_draft and email_draft != "No email drafted":
        st.markdown(
            f"""
            <div style="background:#2f2f2f; border:1px solid #3f3f3f; border-radius:12px; padding:16px; margin:16px 0;">
                <div style="color:#ececec; font-weight:600; font-size:14px; margin-bottom:8px;"> User-Generated Email Draft (Supportive Response)</div>
                <div style="color:#ececec; font-size:13.5px; white-space:pre-wrap; background:#171717; padding:12px; border-radius:8px; border:1px solid #2f2f2f;">{email_draft}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # AI Resolution Panel
    st.markdown(
        """
        <div style="background:#171717; border:1px solid #2f2f2f;
             border-radius:12px; padding:20px; margin:16px 0;">
            <div style="font-size:17px; font-weight:700; color:#ececec; margin-bottom:8px;">
                 AI Resolution Recommendations
            </div>
            <div style="color:#b4b4b4; font-size:13px;">
                Generated by the Resolution Agent using the ticket analysis and SOP documentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate or retrieve resolution
    res_key = f"resolution_{ticket_id}"
    if res_key not in st.session_state:
        with st.spinner(" Resolution Agent is generating recommendations..."):
            from rag.retriever import search_text
            sop_ctx = search_text(f"{ticket['category']} {ticket['issue_summary']}", k=3)

            resolution = get_cached_resolution(
                category=ticket.get("category", "Other"),
                issue_summary=ticket.get("issue_summary", ""),
                probable_cause=ticket.get("probable_cause", ""),
                severity=ticket.get("severity", "Medium"),
                sop_context=sop_ctx,
            )
            st.session_state[res_key] = resolution

    resolution = st.session_state[res_key]

    # Resolution steps
    steps = resolution.get("resolution_steps", [])
    if steps:
        st.markdown(
            '<div style="color:#94a3b8; font-size:14px; font-weight:600; margin-bottom:10px;"> Resolution Steps:</div>',
            unsafe_allow_html=True,
        )
        render_resolution_steps(steps)

    # Estimated time + escalation
    col_e, col_s = st.columns(2)
    with col_e:
        st.info(f" **Estimated Resolution Time:** {resolution.get('estimated_time', '15 mins')}")
    with col_s:
        if resolution.get("escalation_needed", False):
            st.warning(" **Escalation Recommended:** This issue requires level-2 support.")

    # Script block
    script = resolution.get("powershell_script", "")
    if script:
        st.markdown(
            '<div style="color:#94a3b8; font-size:14px; font-weight:600; margin-top:16px; margin-bottom:10px;"> Recommended Troubleshooting Script:</div>',
            unsafe_allow_html=True,
        )
        render_script_block(script, language="powershell" if "windows" in ticket.get("probable_cause", "").lower() or "win" in ticket.get("probable_cause", "").lower() else "bash")

    # Resolution notes
    st.markdown(
        '<div style="color:#94a3b8; font-size:13px; font-weight:600; margin:16px 0 6px;"> Engineer Notes:</div>',
        unsafe_allow_html=True,
    )
    notes = st.text_area(
        "Add resolution notes",
        value=ticket.get("resolution_notes", ""),
        height=100,
        placeholder="Describe what you did to resolve the issue...",
        label_visibility="collapsed",
        key=f"notes_{ticket_id}",
    )
    col_sn, col_draft = st.columns(2)
    with col_sn:
        if st.button(" Save Notes", key=f"save_notes_{ticket_id}", use_container_width=True):
            update_ticket_status(ticket_id, ticket.get("status", "In Progress"), resolution_notes=notes)
            st.success("Notes saved successfully.")
            st.rerun()
            
    with col_draft:
        if st.button(" Draft Knowledge Article", key=f"draft_kb_{ticket_id}", use_container_width=True):
            with st.spinner(" AI is drafting the Knowledge Base article..."):
                from agents.resolution_agent import draft_knowledge_article
                old_solution = "\n".join(resolution.get("resolution_steps", []))
                draft = draft_knowledge_article(
                    ticket.get("issue_summary", ""), 
                    old_solution, 
                    notes
                )
                st.session_state[f"kb_draft_{ticket_id}"] = draft
                
    if f"kb_draft_{ticket_id}" in st.session_state:
        st.markdown("---")
        st.markdown(
            '<div style="color:#a5b4fc; font-size:15px; font-weight:700; margin-bottom:8px;"> Drafted Knowledge Base Article</div>',
            unsafe_allow_html=True,
        )
        draft_content = st.session_state[f"kb_draft_{ticket_id}"]
        edited_draft = st.text_area(
            "Review and edit the drafted article before approving:",
            value=draft_content,
            height=300,
            key=f"edit_draft_{ticket_id}"
        )
        if st.button(" Approve & Save to Knowledge Base", type="primary", use_container_width=True, key=f"approve_kb_{ticket_id}"):
            from database.crud import approve_and_save_solution
            title = f"{ticket.get('category', 'General')} Issue - {ticket.get('ticket_id')}"
            approve_and_save_solution(
                ticket_id=ticket_id,
                title=title,
                category=ticket.get("category", "General"),
                content=edited_draft
            )
            st.success(" Solution approved and saved to Knowledge Base!")
            del st.session_state[f"kb_draft_{ticket_id}"]
            st.rerun()

def render_it_dashboard(username: str):
    main_col = st.columns([1])[0]
    with main_col:
        st.markdown('<div id="main-col-marker"></div>', unsafe_allow_html=True)
        if "selected_ticket_id" in st.session_state and st.session_state.selected_ticket_id:
            _render_full_ticket_detail_view(st.session_state.selected_ticket_id)
            return

        render_page_header(
            "IT Engineer Dashboard",
            f"Logged in as: {username} · All tickets with AI analysis",
        )

        tabs = st.tabs(["All Tickets", "Knowledge Base"])

        with tabs[0]:
            _render_all_tickets()

        with tabs[1]:
            _render_knowledge_base()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: All Tickets
# ─────────────────────────────────────────────────────────────────────────────

def _render_all_tickets():
    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Open", "In Progress", "Resolved", "Escalated", "Cancelled"],
            key="status_filter",
        )
    with col2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "Critical", "High", "Medium", "Low"],
            key="sev_filter",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Refresh", use_container_width=True, key="refresh_tickets"):
            st.rerun()

    # Fetch and filter tickets
    status_param = None if status_filter == "All" else status_filter
    tickets = get_all_tickets(status=status_param)

    if severity_filter != "All":
        tickets = [t for t in tickets if t.get("severity") == severity_filter]

    if not tickets:
        render_info_banner("No Tickets Found", "No tickets match your filter criteria.", icon=None)
        return

    st.markdown(
        f'<div style="color:#64748b; font-size:13px; margin-bottom:12px;">Showing {len(tickets)} ticket(s)</div>',
        unsafe_allow_html=True,
    )

    for ticket in tickets:
        col_left, col_right = st.columns([5, 1])
        with col_left:
            render_ticket_card(ticket, compact=True)
        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open →", key=f"open_{ticket['ticket_id']}", use_container_width=True):
                st.session_state.selected_ticket_id = ticket["ticket_id"]
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

def _render_knowledge_base():
    st.markdown("<br>", unsafe_allow_html=True)

    # RAG index status
    if is_index_ready():
        render_info_banner(
            "Knowledge Base: Ready ",
            "FAISS vector index is built and ready for semantic search. "
            "Includes Microsoft Windows troubleshooting PDF + 5 SOP documents.",
            color="#22c55e",
            icon=None,
        )
    else:
        render_info_banner(
            "Knowledge Base: Not Built ",
            "Run `python rag/ingest.py` from the project root to build the FAISS index. "
            "This will process the PDF and SOP documents (~2-5 minutes).",
            color="#ef4444",
            icon=None,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # RAG Search Test
    st.markdown(
        '<div style="color:#a5b4fc; font-size:16px; font-weight:700; margin-bottom:12px;"> Test Knowledge Base Search</div>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search knowledge base:",
        placeholder="e.g. BitLocker recovery password not working...",
        key="kb_search_input",
    )

    if st.button(" Search", key="kb_search_btn", type="primary") and search_query:
        if not is_index_ready():
            st.error("Knowledge base not built yet. Run `python rag/ingest.py` first.")
        else:
            from rag.retriever import search
            with st.spinner("Searching..."):
                results = search(search_query, k=3)

            if not results:
                st.warning("No results found.")
            else:
                for i, r in enumerate(results, 1):
                    with st.expander(f"Result {i} — {r['source']} (score: {r['score']:.3f})"):
                        st.markdown(f"```\n{r['content']}\n```")

    st.markdown("---")

    # Knowledge articles list
    st.markdown(
        '<div style="color:#a5b4fc; font-size:16px; font-weight:700; margin-bottom:12px;"> SOP Documents in Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    articles = get_all_knowledge_articles()
    if not articles:
        render_info_banner("No Articles Found", "Run the app once to seed the knowledge articles.", icon=None)
    else:
        for art in articles:
            icon = CATEGORY_ICONS.get(art["category"], "")
            st.markdown(
                f"""
                <div style="background:#1e293b; border:1px solid #334155; border-radius:10px;
                     padding:14px; margin:6px 0; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="color:#e2e8f0; font-size:14px; font-weight:600;">{icon} {art['title']}</div>
                        <div style="color:#64748b; font-size:12px; margin-top:4px;">Category: {art['category']} | File: {art['filename']}</div>
                    </div>
                    <a href="{art.get('source_url', '#')}" target="_blank"
                       style="color:#6366f1; font-size:12px; text-decoration:none;"> Source</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # PDF info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:16px;">
            <div style="color:#a5b4fc; font-size:14px; font-weight:700; margin-bottom:8px;"> Microsoft PDF Knowledge Source</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.6;">
                <strong style="color:#e2e8f0;">File:</strong> troubleshoot-windows-client.pdf (68 MB)<br>
                <strong style="color:#e2e8f0;">Source:</strong> Microsoft Learn — Windows Client Troubleshooting Documentation<br>
                <strong style="color:#e2e8f0;">URL:</strong> https://learn.microsoft.com/en-us/troubleshoot/windows-client/welcome-windows-client<br>
                <strong style="color:#e2e8f0;">Content:</strong> Performance, Networking, Security, BitLocker, Printing, Backup, Identity & Access, Virtualization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Upload PDF SOP to Knowledge Base ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#a5b4fc; font-size:16px; font-weight:700; margin-bottom:12px;"> Upload SOP / Reference PDF</div>',
        unsafe_allow_html=True,
    )
    
    with st.form("kb_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Select troubleshooting PDF file:",
            type=["pdf"],
            help="Upload an official vendor guide or local SOP. The PDF will be scanned, chunked, and embedded into the search database."
        )
        
        col_cat, col_title = st.columns(2)
        with col_cat:
            categories_list = ["Security / BitLocker", "Performance", "VPN / Remote Access", "Network", "Email", "Access / Permissions", "Software", "Hardware", "Backup / Storage", "Printer", "Windows Update", "Other"]
            pdf_category = st.selectbox("Category Mapping:", categories_list)
        with col_title:
            pdf_title = st.text_input("Document Name / Topic:", placeholder="e.g. Dell Latitude Diagnostic Steps")
            
        submit_upload = st.form_submit_button(" Ingest and Index PDF", type="primary")
        
        if submit_upload:
            if not uploaded_file:
                st.error("Please upload a PDF file first.")
            elif not pdf_title.strip():
                st.error("Please provide a name/topic for the document.")
            else:
                with st.spinner("Processing PDF: extracting pages and building embeddings..."):
                    try:
                        import os
                        from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL
                        # 1. Save file locally to persist it
                        dest_dir = os.path.join("rag", "uploaded_documents")
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, uploaded_file.name)
                        with open(dest_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # 2. Extract pages using PyPDF
                        import pypdf
                        from langchain_core.documents import Document
                        from langchain_community.vectorstores import FAISS
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        
                        pdf_reader = pypdf.PdfReader(dest_path)
                        
                        # Extract full text from all pages
                        full_text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            full_text += page.extract_text() or ""
                            
                        # A. DETERMINISTIC SAFETY SCAN ON FULL TEXT (Layer 1 Guardrail check)
                        from utils.guardrail import decode_obfuscated_text, check_credentials_deterministic
                        cleaned_full_text = decode_obfuscated_text(full_text)
                        has_creds, cred_reason = check_credentials_deterministic(cleaned_full_text)
                        if has_creds:
                            if os.path.exists(dest_path):
                                os.remove(dest_path)
                            st.error(f"❌ **Security Violation:** The document contains sensitive credentials or API keys and cannot be uploaded.\n\n*Reason:* {cred_reason}")
                            return

                        # B. GROQ RELEVANCE VERIFICATION (1 API Call with sample of full text)
                        sample_text = full_text.strip()[:8000] # Take first 8000 characters to capture table of contents, intro, etc.
                        is_relevant, reason = check_pdf_relevance_via_groq(sample_text)
                        if not is_relevant:
                            if os.path.exists(dest_path):
                                os.remove(dest_path)
                            st.error(f"❌ **Relevance Verification Failed:** This document does not appear to be related to IT Infrastructure Troubleshooting.\n\n*Reason:* {reason}")
                            return

                        new_docs = []
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text() or ""
                            if not text.strip():
                                continue
                            
                            # Chunk size matching standard character splitter
                            chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
                            for c_idx, chunk in enumerate(chunks):
                                # Include context category in chunk text to enhance search relevance
                                enhanced_text = f"Category: {pdf_category} | Title: {pdf_title} | Page {page_num + 1} | Content: {chunk}"
                                new_docs.append(Document(
                                    page_content=enhanced_text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "page": page_num,
                                        "chunk": c_idx,
                                        "is_pdf": True,
                                        "category": pdf_category
                                    }
                                ))
                                
                        if not new_docs:
                            st.warning("The PDF appears to be empty or contains scanned images without selectable text.")
                        else:
                            # 3. Add to FAISS index
                            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                            if is_index_ready():
                                db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                                db.add_documents(new_docs)
                                db.save_local(FAISS_INDEX_PATH)
                            else:
                                db = FAISS.from_documents(new_docs, embeddings)
                                db.save_local(FAISS_INDEX_PATH)
                                
                            # 4. Insert dynamic article row in SQLite knowledge_articles table
                            from database.crud import add_knowledge_article
                            add_knowledge_article(
                                title=pdf_title,
                                category=pdf_category,
                                filename=uploaded_file.name,
                                content_preview=f"Uploaded PDF: {uploaded_file.name} ({len(pdf_reader.pages)} pages)"
                            )
                            st.success(f"🎉 **{uploaded_file.name}** successfully parsed! {len(new_docs)} semantic chunks added to knowledge base.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")




