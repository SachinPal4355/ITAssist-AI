"""
User Portal — Interactive Chat Page
Full multi-turn workflow:
  1. User types their IT issue
  2. AI classifies + searches RAG knowledge base
  3. AI asks 2-3 diagnostic questions  ← INTERACTIVE
  4. User answers each question        ← INTERACTIVE
  5. AI analyzes root cause
  6. AI suggests self-resolution steps ← INTERACTIVE (Resolved / Still Not Working)
  7. If unresolved → ticket preview    ← INTERACTIVE (Approve / Edit / Cancel)
  8. Ticket stored in SQLite
"""
import streamlit as st
import uuid
import re
from datetime import datetime
from ui.components import (
    render_chat_message,
    render_agent_step,
    render_ticket_card,
    render_info_banner,
    render_resolution_steps,
    render_script_block,
    render_page_header,
    CATEGORY_ICONS,
    SEVERITY_COLORS,
)
from agents.graph import run_intake_and_knowledge, run_analysis_and_resolution
from database.crud import create_ticket, get_or_create_user, log_message
from rag.retriever import is_index_ready


# ─────────────────────────────────────────────────────────────────────────────
# Session State Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "portal_stage": "idle",
        "agent_state": None,
        "current_question_idx": 0,
        "user_answers": [],
        "chat_history": [],
        "session_id": str(uuid.uuid4()),
        "ticket_data": None,
        "edit_summary": "",
        "console_logs": [],              # Animated console entries
        "attached_files_context": "",   # Extracted text from uploaded files
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _add_chat(role: str, content: str, agent_step: str = ""):
    st.session_state.chat_history.append(
        {"role": role, "content": content, "agent_step": agent_step}
    )


def _reset_portal():
    keys = [
        "portal_stage", "agent_state", "current_question_idx", "user_answers",
        "chat_history", "ticket_data", "edit_summary",
        "console_logs", "attached_files_context", "email_draft",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.session_id = str(uuid.uuid4())


def _render_console_logs_html(logs) -> str:
    if not logs:
        return '<div style="color:#b4b4b4; font-style:italic; font-size:12px;">Waiting for input...</div>'
    log_parts = []
    for item in logs:
        log_parts.append(
            f'<div style="margin-bottom: 12px; border-left: 2px solid #b4b4b4; padding-left: 10px; font-family: \'Inter\', sans-serif;">'
            f'<div style="display:flex; justify-content:space-between; font-size:10px; margin-bottom:2px;">'
            f'<span style="color:#ffffff; font-weight:700;">{item["agent"]}</span>'
            f'<span style="color:#b4b4b4;">{item["time"]}</span>'
            f'</div>'
            f'<div style="color:#ffffff; font-size:12px; line-height:1.4;">{item["detail"]}</div>'
            f'</div>'
        )
    return "".join(log_parts)


def _log(msg: str):
    """Append a timestamped agent dialogue entry to the console and update UI in real-time."""
    ts = datetime.now().strftime("%H:%M:%S")
    
    agent = "🤖 System Agent"
    detail = msg
    
    # ── Safety Guardrail ──
    if "Guardrail scanning" in msg:
        agent = "🛡️ Safety Guardrail Agent"
        detail = "<b>Doing:</b> Running Relevance Classifier, Safety Validation, Moderation, and Safety Classification checks..."
    elif "Guardrail BLOCKED" in msg:
        agent = "🛡️ Safety Guardrail Agent"
        reason = msg.split("BLOCKED:")[-1].strip() if "BLOCKED:" in msg else ""
        detail = f"<b>Found:</b> Violations detected. Request blocked: {reason}"
    elif "Guardrail: Content cleared" in msg:
        agent = "🛡️ Safety Guardrail Agent"
        detail = "<b>Found:</b> All guardrail checks passed. Content is safe to process."
        
    # ── Search & RAG retrieval ──
    elif "Scanning FAISS" in msg:
        agent = "🔍 Retrieval Agent"
        detail = "<b>Doing:</b> Querying local FAISS database for matching troubleshooting SOP documentation..."
    elif "Matched" in msg:
        agent = "🔍 Retrieval Agent"
        match = re.search(r"Matched (\d+) local", msg)
        count = match.group(1) if match else "some"
        detail = f"<b>Found:</b> Matched <b>{count}</b> related local SOP article(s) in the database."
    elif "No exact local" in msg:
        agent = "🔍 Retrieval Agent"
        detail = "<b>Found:</b> No direct matching SOP found. Falling back to general knowledge database."
    elif "FAISS search skipped" in msg:
        agent = "🔍 Retrieval Agent"
        detail = "<b>Found:</b> Index not ready. Bypassing search."
        
    # ── Intake and Classification ──
    elif "Sending to Groq for classification" in msg:
        agent = "🧠 Triage Agent"
        detail = "<b>Doing:</b> Calling Groq API (LLaMA-3.1) to classify issue category and check for missing context..."
    elif "Classified:" in msg:
        agent = "🧠 Triage Agent"
        match = re.search(r"Classified:\s*(.*?)\s*\((.*?)\)", msg)
        if match:
            category, conf = match.group(1), match.group(2)
            detail = f"<b>Found:</b> Classified issue category as <b>{category}</b> ({conf} confidence)."
        else:
            detail = f"<b>Found:</b> Classified issue category."
    elif "Cross-questions generated" in msg:
        agent = "🧠 Triage Agent"
        detail = "<b>Clarifying:</b> Generating diagnostic verification questions for user response..."
        
    # ── Ingestion and Vision AI ──
    elif "Reading" in msg:
        agent = "📁 Ingestion Agent"
        detail = "<b>Doing:</b> Accessing uploaded attachments to extract context..."
    elif "Vision AI processed" in msg:
        agent = "📁 Ingestion Agent"
        filename = msg.split("processed:")[-1].strip() if "processed:" in msg else ""
        detail = f"<b>Converting:</b> Extracted visual elements of <b>{filename}</b> using Vision API."
    elif "Extracted text" in msg:
        agent = "📁 Ingestion Agent"
        filename = msg.split("from:")[-1].strip() if "from:" in msg else ""
        detail = f"<b>Converting:</b> Extracted text elements of <b>{filename}</b>."
    elif "File context ready" in msg:
        agent = "📁 Ingestion Agent"
        detail = "<b>Found:</b> Ingested text context successfully loaded into working memory."
        
    # ── Resolution generation ──
    elif "Writing session_finding" in msg:
        agent = "🔧 Resolution Agent"
        detail = "<b>Doing:</b> Compiling findings and writing to session_finding.txt state file..."
    elif "Final resolution ready" in msg:
        agent = "🔧 Resolution Agent"
        detail = "<b>Found:</b> Step-by-step resolution recommendations and scripts generated successfully!"
        
    # ── Error ──
    elif "error" in msg.lower() or "failed" in msg.lower():
        agent = "⚠️ System Diagnostics"
        detail = f"<b>Error:</b> {msg}"

    st.session_state.console_logs.append({
        "time": ts,
        "agent": agent,
        "detail": detail
    })
    
    if "console_placeholder" in st.session_state and st.session_state.console_placeholder:
        badge = st.session_state.get("console_badge", "")
        log_html = _render_console_logs_html(st.session_state.console_logs)
        console_html = f"""
        <div style="background:#171717; border:1px solid #2f2f2f; border-radius:12px; padding:16px; height:360px; overflow-y:auto;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid #2f2f2f; padding-bottom:8px;">
                <span style="color:#ececec; font-weight:700; font-size:14px; font-family:monospace;">⚡ AI Console</span>
                {badge}
            </div>
            <div style="line-height:1.5;">
                {log_html}
            </div>
        </div>
        """
        st.session_state.console_placeholder.markdown(console_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────────────────────────────────────

def render_user_portal(username: str):
    _init_session()

    render_page_header(
        "🤖 ITAssist AI — Service Desk Copilot",
        f"Welcome, {username} · Powered by Groq LLaMA-3 + RAG (Microsoft Windows Docs)",
    )

    # ── 2-Column Layout ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([7.2, 2.8])

    with col_left:
        # ── RAG Index Warning ─────────────────────────────────────────────────
        if not is_index_ready():
            st.warning(
                "⚠️ **Knowledge Base Not Built Yet** — "
                "Run `python rag/ingest.py` from the project root to build the FAISS index from your PDF + SOP documents. "
                "The AI will still work, but without document search.",
                icon="⚠️",
            )

        # ── Chat History Display ──────────────────────────────────────────────
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "assistant" and msg.get("agent_step") == "🔧 Service Desk Copilot Response":
                    # Display the assistant icon and step badge
                    step_badge = f'<span style="font-size:10px; background:#171717; color:#b4b4b4; padding:2px 8px; border-radius:10px; margin-bottom:6px; display:inline-block; border: 1px solid #2f2f2f;">{msg.get("agent_step")}</span><br>'
                    st.markdown(
                        f"""
                        <div style="display: flex; margin: 12px 0; align-items: flex-start; margin-bottom: 2px;">
                            <div style="margin-right: 8px; font-size: 20px; padding-top: 4px;">🤖</div>
                            <div style="background: transparent; color: #ececec; padding: 4px 12px; max-width: 85%; font-size: 14px; line-height: 1.6;">
                                {step_badge}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Display the structured step cards
                    col_spacer, col_content = st.columns([0.06, 0.94])
                    with col_content:
                        render_structured_resolution(msg["content"])
                else:
                    render_chat_message(msg["role"], msg["content"], msg.get("agent_step", ""))

            st.markdown("---")

        stage = st.session_state.portal_stage

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: IDLE — Initial issue input
        # ══════════════════════════════════════════════════════════════════════
        if stage == "idle":
            _render_idle_stage(username)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: CLASSIFYING — AI processing intake & search
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "classifying":
            _render_classifying_stage(username)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: ANSWERING QUESTIONS — User replies to dynamic questions
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "answering_questions":
            _render_answering_questions_stage()

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: GENERATING RESOLUTION — RAG context file + Groq Final
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "generating_resolution":
            _render_generating_resolution_stage()

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: SELF RESOLVE — Show AI recommendations, user decides
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "self_resolve":
            _render_self_resolve_stage(username)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: TICKET PREVIEW — User approves ticket
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "ticket_preview":
            _render_ticket_preview_stage(username)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE: DONE
        # ══════════════════════════════════════════════════════════════════════
        elif stage == "done":
            _render_done_stage()

    with col_right:
        # ── Animated AI Console ───────────────────────────────────────────────
        is_live = stage in ["classifying", "generating_resolution"]
        badge = "<span style='background:#10b981;color:white;font-size:10px;padding:2px 7px;border-radius:10px;font-weight:600;'>🟢 LIVE</span>" if is_live else "<span style='background:#3f3f3f;color:#b4b4b4;font-size:10px;padding:2px 7px;border-radius:10px;'>⚪ IDLE</span>"

        # Create placeholder for real-time console logs
        console_placeholder = st.empty()
        
        logs = st.session_state.get("console_logs", [])
        log_html = _render_console_logs_html(logs)

        console_html = f"""
        <div style="background:#171717; border:1px solid #2f2f2f; border-radius:12px; padding:16px; height:360px; overflow-y:auto;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid #2f2f2f; padding-bottom:8px;">
                <span style="color:#ececec; font-weight:700; font-size:14px; font-family:monospace;">⚡ AI Console</span>
                {badge}
            </div>
            <div style="line-height:1.5;">
                {log_html}
            </div>
        </div>
        """
        console_placeholder.markdown(console_html, unsafe_allow_html=True)
        
        # Cache placeholder and badge for _log real-time rendering
        st.session_state.console_placeholder = console_placeholder
        st.session_state.console_badge = badge

        # ── Attachment Summary ────────────────────────────────────────────────
        attached = st.session_state.get("attached_files_context", "")
        if attached:
            st.markdown(
                "<div style='margin-top:10px; background:#2f2f2f; border:1px solid #3f3f3f; border-radius:8px; padding:10px;'>"
                "<span style='color:#ececec; font-size:12px; font-weight:600;'>📎 Attachment processed</span>"
                "<br><span style='color:#b4b4b4; font-size:11px;'>Content injected into AI context</span></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Stage Implementations
# ─────────────────────────────────────────────────────────────────────────────

def _render_idle_stage(username: str):
    """Clean professional issue submission form."""

    # ── Section title ──────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:13px; font-weight:600; color:#475569; margin:0 0 6px 0;'>"
        "Describe your IT issue</p>",
        unsafe_allow_html=True,
    )

    # ── Common issue shortcut buttons (single row, no card duplication) ──────
    SHORTCUTS = [
        ("VPN Error 800",       "VPN won't connect — error 800"),
        ("Slow PC / Freezing",  "My laptop is very slow and freezing"),
        ("BitLocker Key",       "BitLocker asking for recovery key"),
        ("Wi-Fi Dropping",      "Wi-Fi keeps disconnecting randomly"),
        ("Outlook Not Opening", "Outlook not opening or crashing on start"),
        ("Printer Offline",     "Printer showing offline and won't print"),
    ]

    st.markdown("<p style='font-size:11px; color:#94a3b8; margin:0 0 4px 0;'>Common issues:</p>", unsafe_allow_html=True)
    cols = st.columns(6)
    for idx, (label, full_text) in enumerate(SHORTCUTS):
        with cols[idx]:
            if st.button(label, key=f"shortcut_{idx}", use_container_width=True):
                st.session_state.user_issue_input = full_text
                st.rerun()

    # ── Text Input ────────────────────────────────────────────────────────────
    user_input = st.text_area(
        "Describe your issue:",
        value="",
        height=90,
        placeholder="Example: My laptop freezes and disk usage hits 100%...",
        label_visibility="collapsed",
        key="user_issue_input",
    )

    # ── File Uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "📎 Attach files (screenshots, logs, PDFs, DOCX)",
        type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "txt", "log"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Files are read in memory only and never stored on disk.",
    )

    if uploaded_files:
        names = ", ".join(f.name for f in uploaded_files)
        st.markdown(
            f"<div style='background:#162032; border:1px solid #0ea5e9; border-radius:8px; "
            f"padding:8px 12px; font-size:12px; color:#38bdf8; margin-top:6px;'>"
            f"📎 {len(uploaded_files)} file(s) attached: {names}</div>",
            unsafe_allow_html=True,
        )

    # ── Submit / Reset ────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 Analyze My Issue", type="primary", use_container_width=True, key="submit_issue")
    with col2:
        if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
            _reset_portal()
            st.rerun()

    if submit:
        if not user_input.strip():
            st.error("Please describe your issue before submitting.")
            return

        # ── Step 1: Extract text from attached files ──────────────────────────
        attached_context = ""
        if uploaded_files:
            _log(f"📎 Reading {len(uploaded_files)} attached file(s)...")
            from utils.file_reader import extract_text_from_file, semantic_search_in_doc, strip_pii_patterns
            parts = []
            full_parts = []
            for f in uploaded_files:
                raw_text, method = extract_text_from_file(f)
                if method == "image":
                    _log(f"🖼️ Vision AI processed: {f.name}")
                else:
                    _log(f"📄 Extracted text from: {f.name}")
                # Strip obvious PII before validation
                clean_text = strip_pii_patterns(raw_text)
                # Save full file content for complete guardrail validation
                full_parts.append(f"[{f.name} - Full Content]:\n{clean_text}")
                # Semantic search: find the most relevant chunks for user query processing
                relevant = semantic_search_in_doc(user_input.strip(), clean_text, top_k=3)
                if relevant:
                    parts.append(f"[{f.name}]:\n{relevant}")
            attached_context = "\n\n".join(parts)
            st.session_state.attached_files_context = attached_context
            st.session_state.full_uploaded_text = "\n\n".join(full_parts)
            _log("✅ File context ready")

        _add_chat("user", user_input.strip())
        st.session_state.user_issue = user_input.strip()
        st.session_state.portal_stage = "classifying"
        st.rerun()


def _render_classifying_stage(username: str):
    """Guardian check → RAG → cross-questions generation."""
    user_message = st.session_state.get("user_issue", "")
    attached_context = st.session_state.get("attached_files_context", "")

    with st.spinner("🛡️ Guardrail checking content safety..."):
        # ── Step 2: FAISS local search (fast, no API call) ────────────────────
        _log("🔍 Scanning FAISS knowledge base...")
        from rag.retriever import search as rag_search
        try:
            local_docs = rag_search(user_message, k=3)
            local_context = "\n\n".join(d.get("content", "") for d in local_docs) if local_docs else ""
            if local_docs:
                _log(f'📄 Matched {len(local_docs)} local SOP article(s)')
            else:
                _log("📄 No exact local SOP match — using general knowledge")
        except Exception:
            local_context = ""
            _log("📄 FAISS search skipped (index not ready)")


        # ── Step 3: Guardrail safety check ────────────────────────────────────
        _log("🛡️ Guardrail scanning input + attachments...")
        from utils.guardrail import guardrail_check, build_enriched_context
        guard = guardrail_check(
            user_issue=user_message,
            local_rag_context=local_context,
            attached_doc_context=st.session_state.get("full_uploaded_text", ""),
        )

        if not guard.safe:
            _log(f"🚫 Guardrail BLOCKED: {guard.reason[:60]}")
            st.error(
                f"🛡️ **Guardrail blocked this request**\n\n"
                f"**Reason:** {guard.reason}\n\n"
                f"Please review your input and attachments, remove any off-topic or sensitive content, and try again."
            )
            st.session_state.portal_stage = "idle"
            st.rerun()
            return

        _log("✅ Guardrail: Content cleared")

        # ── Step 4: Build enriched context for Groq ───────────────────────────
        enriched_context = build_enriched_context(
            user_issue=user_message,
            local_rag_context=local_context,
            attached_doc_context=attached_context,
        )
        # Pass enriched context alongside the user message to the agent
        augmented_message = f"{user_message}\n\n--- CONTEXT ---\n{enriched_context}"

    with st.spinner("🤖 AI classifying issue and generating diagnostic questions..."):
        try:
            _log("🤖 Sending to Groq for classification + cross-questions...")
            agent_state = run_intake_and_knowledge(
                user_message=augmented_message,
                session_id=st.session_state.session_id,
            )
            st.session_state.agent_state = agent_state

            cat = agent_state.get("category", "Other")
            conf = agent_state.get("intake_confidence", 0.0)
            icon = CATEGORY_ICONS.get(cat, "❓")
            articles = agent_state.get("articles_used", [])

            _log(f"📊 Classified: {cat} ({conf:.0%} confidence)")

            _add_chat(
                "assistant",
                f"**Issue Classified:** {icon} **{cat}** (confidence: {conf:.0%})\n\n"
                f"**Knowledge Base Searched:** {', '.join(articles) if articles else 'General SOP documentation'}"
                + (f"\n\n📎 *Attached document context also included in analysis.*" if attached_context else ""),
                agent_step="🔍 Intake + 📚 Knowledge Agents",
            )

            questions = agent_state.get("diagnostic_questions", "")
            _log("❓ Cross-questions generated — awaiting user answers")
            _add_chat(
                "assistant",
                f"**To verify the exact cause, please answer these diagnostic verification questions:**\n\n{questions}",
                agent_step="❓ Dynamic Cross-Questioning",
            )

            st.session_state.portal_stage = "answering_questions"

        except Exception as e:
            _log(f"❌ Agent error: {str(e)[:50]}")
            st.error(f"AI processing error: {str(e)}")
            st.session_state.portal_stage = "idle"

    st.rerun()

def _render_answering_questions_stage():
    """Render the user response text area for the dynamic cross-questions."""
    agent_state = st.session_state.agent_state
    if not agent_state:
        st.session_state.portal_stage = "idle"
        st.rerun()
        return

    st.markdown(
        """
        <div style="background:#2f2f2f; border:1px solid #3f3f3f; border-radius:12px; padding:16px; margin-bottom:12px;">
            <div style="color:#ececec; font-weight:700; font-size:14px; margin-bottom:6px;">❓ Diagnostic Clarification</div>
            <div style="color:#ececec; font-size:13px; line-height:1.5;">Please reply to the questions in the chat window to help the copilot find the exact solution.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_answers = st.text_area(
        "Your answers:",
        placeholder="Type your answers / verification details here...",
        height=100,
        key="diagnostic_answers_input"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("→ Submit Answers", type="primary", use_container_width=True, key="submit_answers_btn"):
            if user_answers.strip():
                _add_chat("user", user_answers.strip())
                st.session_state.user_answers_input = user_answers.strip()
                st.session_state.portal_stage = "generating_resolution"
                st.rerun()
            else:
                st.error("Please provide some answers to proceed.")
    with col2:
        if st.button("🔄 Reset", use_container_width=True, key="reset_btn_answering"):
            _reset_portal()
            st.rerun()


def _render_generating_resolution_stage():
    """Run the dynamic resolution agent using the context file."""
    agent_state = st.session_state.agent_state
    if not agent_state:
        st.session_state.portal_stage = "idle"
        st.rerun()
        return

    user_answers = st.session_state.get("user_answers_input", "")

    # ── Safety Guardrail Validation on Diagnostic Answers ──
    with st.spinner("🛡️ Guardrail checking safety of answers..."):
        _log("🛡️ Guardrail scanning user answers...")
        from utils.guardrail import guardrail_check
        guard = guardrail_check(user_issue=user_answers)
        
        if not guard.safe:
            _log(f"🚫 Guardrail BLOCKED answers: {guard.reason[:60]}")
            st.error(
                f"🛡️ **Guardrail blocked this request**\n\n"
                f"**Reason:** {guard.reason}\n\n"
                f"Please review your answers and remove any off-topic or sensitive content."
            )
            # Revert back to answering stage to allow correction
            st.session_state.portal_stage = "answering_questions"
            st.rerun()
            return
            
        _log("✅ Guardrail: Answers cleared")

    with st.spinner("🔧 Generating final step-by-step troubleshooting suggestions..."):
        try:
            _log("✍️ Writing session_finding.txt...")
            from agents.graph import generate_final_resolution
            updated_state = generate_final_resolution(
                state=agent_state,
                user_answers_text=st.session_state.user_answers_input
            )
            st.session_state.agent_state = updated_state

            chat_reply = updated_state.get("analysis_text", "")
            _log("🔧 Final resolution ready!")
            _add_chat(
                "assistant",
                f"**Here are the recommended troubleshooting steps:**\n\n{chat_reply}",
                agent_step="🔧 Service Desk Copilot Response"
            )

            st.session_state.portal_stage = "self_resolve"
        except Exception as e:
            _log(f"❌ Resolution error: {str(e)[:50]}")
            st.error(f"Resolution generation error: {str(e)}")
            st.session_state.portal_stage = "self_resolve"

    st.rerun()


def render_structured_resolution(chat_reply: str):
    """
    Parse a raw markdown chat reply from the agent, extracting introduction,
    numbered steps, and code blocks, and render them using structured UI components.
    """
    # 1. Extract code blocks
    code_blocks = re.findall(r"```(powershell|bash|cmd|shell|python)?\n(.*?)\n```", chat_reply, re.DOTALL | re.IGNORECASE)
    cleaned_reply = re.sub(r"```(?:powershell|bash|cmd|shell|python)?\n.*?\n```", "", chat_reply, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Extract intro paragraph
    steps_start_match = re.search(r"(?:^|\n)\*?\*?Step\s+\d+", cleaned_reply, re.IGNORECASE)
    if steps_start_match:
        intro_text = cleaned_reply[:steps_start_match.start()].strip()
        steps_part = cleaned_reply[steps_start_match.start():].strip()
    else:
        intro_text = cleaned_reply.strip()
        steps_part = ""
        
    # Clean Intro
    if intro_text:
        intro_text = re.sub(r"(?:^|\n)\*?Here are the recommended.*?\*?:?", "", intro_text, flags=re.IGNORECASE)
        intro_text = intro_text.strip()
        if intro_text:
            st.markdown(f"<div style='color:#ececec; font-size:14px; line-height:1.6; margin-bottom:14px;'>{intro_text}</div>", unsafe_allow_html=True)
            
    # 3. Extract individual steps
    if steps_part:
        step_matches = list(re.finditer(r"(?:^|\n)\*?\*?Step\s+(\d+)\s*[:.-]\s*(.*?)(?=\n\*?\*?Step\s+\d+|\Z)", steps_part, re.DOTALL | re.IGNORECASE))
        
        if step_matches:
            steps_list = []
            for match in step_matches:
                step_num = match.group(1)
                step_content = match.group(2).strip()
                
                # Format step title and description cleanly
                first_line_end = step_content.find("\n")
                if first_line_end != -1:
                    first_line = step_content[:first_line_end].strip()
                    rest = step_content[first_line_end:].strip()
                else:
                    first_line = step_content.strip()
                    rest = ""
                
                if first_line.endswith("**"):
                    first_line = first_line[:-2].strip()
                
                if rest:
                    formatted_content = f"<b>{first_line}</b>: {rest}"
                else:
                    formatted_content = f"<b>{first_line}</b>"
                steps_list.append(formatted_content)
                
            if steps_list:
                render_resolution_steps(steps_list)
        else:
            # Fallback if split fails
            st.markdown(f"<div style='color:#ececec; font-size:14px; line-height:1.6; white-space:pre-wrap;'>{steps_part}</div>", unsafe_allow_html=True)
            
    # 4. Render code blocks
    if code_blocks:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        for lang, code in code_blocks:
            lang_name = "PowerShell" if (lang and lang.lower() == "powershell") else "Bash"
            render_script_block(code.strip(), script_type=lang_name)


def _render_self_resolve_stage(username: str):
    """Show resolution suggestions, support email drafting, and ticket options."""
    agent_state = st.session_state.agent_state
    if not agent_state:
        st.session_state.portal_stage = "idle"
        st.rerun()
        return

    problem = agent_state.get("problem_title", "")
    cause = agent_state.get("probable_cause", "")
    chat_reply = agent_state.get("analysis_text", "")

    # The troubleshooting steps are already rendered inline in the chat history.
    pass

    # ── Draft Support Email Button ─────────────────────────────────────────────
    if "email_draft" not in st.session_state:
        if st.button("📧 Draft Support Email", use_container_width=True, key="draft_email_btn"):
            with st.spinner("Writing email draft..."):
                from agents.graph import draft_support_email
                st.session_state.email_draft = draft_support_email(
                    user_message=agent_state.get("user_message", ""),
                    chat_reply=chat_reply
                )
            st.rerun()

    # Show Email Draft if generated
    if st.session_state.get("email_draft"):
        st.markdown(
            """
            <div style="background:#2f2f2f; border:1px solid #3f3f3f; border-radius:12px; padding:16px; margin-bottom:16px;">
                <div style="color:#ececec; font-weight:600; font-size:14px; margin-bottom:8px;">📧 Support Email Draft</div>
                <div style="color:#b4b4b4; font-size:12px; margin-bottom:12px;">You can copy this draft to send to support or keep as a reference.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            "Draft Content",
            value=st.session_state.email_draft,
            height=220,
            key="email_draft_area"
        )
        if st.button("🔄 Regenerate Email Draft", key="regen_email_btn"):
            del st.session_state.email_draft
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#94a3b8; font-size:14px; font-weight:600; margin-bottom:10px;">Select your next action:</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "✅ Resolved — Issue Fixed!",
            type="primary",
            use_container_width=True,
            key="resolved_btn",
        ):
            _add_chat(
                "user",
                "✅ The issue is resolved. No support ticket is needed.",
            )
            _add_chat(
                "assistant",
                "🎉 **Wonderful! I'm glad the troubleshooting steps helped resolve the issue.**\n\n"
                "No ticket was created. Let me know if you need help with anything else!",
                agent_step="✅ Resolved",
            )
            st.session_state.portal_stage = "done"
            st.rerun()

    with col2:
        if st.button(
            "🎫 Create Support Ticket",
            use_container_width=True,
            key="not_resolved_btn",
        ):
            _add_chat(
                "user",
                "🎫 I would like to create a support ticket for this issue.",
            )
            _add_chat(
                "assistant",
                "Certainly! I have generated a support ticket details review screen for you. Please approve it below.",
                agent_step="🎫 Ticket Creator",
            )
            st.session_state.portal_stage = "ticket_preview"
            st.rerun()


def _render_ticket_preview_stage(username: str):
    """Show full ticket preview. User can approve, edit, or cancel."""
    agent_state = st.session_state.agent_state

    st.markdown(
        """
        <div style="background:#171717; border:1px solid #2f2f2f;
             border-radius:12px; padding:20px; margin-bottom:16px;">
            <div style="font-size:18px; font-weight:700; color:#ececec; margin-bottom:8px;">
                🎫 Support Ticket Preview
            </div>
            <div style="color:#b4b4b4; font-size:13px;">
                Review the AI-generated ticket below. You can edit the summary before approving.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background:#171717; border:1px solid #2f2f2f; border-radius:10px; padding:14px;">
                <div style="color:#b4b4b4; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Category</div>
                <div style="color:#ececec; font-size:16px; font-weight:700; margin-top:4px;">
                    {CATEGORY_ICONS.get(agent_state.get('category','Other'), '❓')} {agent_state.get('category','Other')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        sev = agent_state.get("severity", "Medium")
        sev_color = SEVERITY_COLORS.get(sev, "#f59e0b")
        conf = agent_state.get("analysis_confidence", 0.0)
        st.markdown(
            f"""
            <div style="background:#171717; border:1px solid #2f2f2f; border-radius:10px; padding:14px;">
                <div style="color:#b4b4b4; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Severity · Confidence</div>
                <div style="font-size:16px; font-weight:700; margin-top:4px;">
                    <span style="color:{sev_color};">⚠ {sev}</span>
                    <span style="color:#10a37f; margin-left:12px;">{conf:.0%} confidence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # AI Analysis Summary
    with st.expander("🧪 AI Analysis Details", expanded=True):
        st.markdown(f"**Problem:** {agent_state.get('problem_title', '')}")
        st.markdown(f"**Probable Cause:** {agent_state.get('probable_cause', '')}")
        st.markdown(f"**Analysis:** {agent_state.get('analysis_text', '')}")
        articles = agent_state.get("articles_used", [])
        if articles:
            st.markdown(f"**Knowledge Articles Used:** {', '.join(articles)}")

    # Q&A Summary
    questions = agent_state.get("diagnostic_questions", [])
    answers = st.session_state.user_answers
    if questions and answers:
        with st.expander("💬 Diagnostic Q&A", expanded=False):
            for q, a in zip(questions, answers):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

    # Editable issue summary
    st.markdown(
        '<div style="color:#94a3b8; font-size:13px; font-weight:600; margin:12px 0 6px;">✏️ Edit Issue Summary (optional):</div>',
        unsafe_allow_html=True,
    )
    default_summary = agent_state.get("issue_summary", agent_state.get("user_message", ""))
    edited_summary = st.text_area(
        "Issue summary",
        value=st.session_state.get("edit_summary", default_summary),
        height=80,
        label_visibility="collapsed",
        key="ticket_summary_edit",
    )
    st.session_state.edit_summary = edited_summary

    st.markdown("<br>", unsafe_allow_html=True)

    # Action buttons
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ Approve & Submit Ticket", type="primary", use_container_width=True, key="approve_ticket"):
            _submit_ticket(username, agent_state, edited_summary)

    with col_b:
        if st.button("✏️ Edit Summary", use_container_width=True, key="edit_ticket"):
            st.info("Edit the summary text above, then click Approve.")

    with col_c:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_ticket"):
            _add_chat("user", "❌ Ticket cancelled.")
            _add_chat(
                "assistant",
                "Ticket has been cancelled. If you need further help, please start a new session.",
                agent_step="Cancelled",
            )
            st.session_state.portal_stage = "done"
            st.rerun()


def _submit_ticket(username: str, agent_state: dict, edited_summary: str):
    """Create the ticket in SQLite and update UI."""
    user = get_or_create_user(username, "user")

    articles = agent_state.get("articles_used", [])

    ticket = create_ticket(
        user_id=user.id,
        category=agent_state.get("category", "Other"),
        issue_summary=edited_summary or agent_state.get("issue_summary", ""),
        ai_analysis=agent_state.get("analysis_text", ""),
        probable_cause=agent_state.get("probable_cause", ""),
        severity=agent_state.get("severity", "Medium"),
        confidence=agent_state.get("analysis_confidence", 0.0),
        questions_asked=0,
        sop_articles_used=", ".join(articles),
        resolution_notes=st.session_state.get("email_draft", "No email drafted"),
    )

    st.session_state.ticket_data = ticket
    _add_chat(
        "assistant",
        f"🎫 **Ticket Created Successfully!**\n\n"
        f"**Ticket ID:** {ticket.ticket_id}\n"
        f"**Status:** Open\n\n"
        f"An IT engineer will review your ticket shortly. "
        f"You can track it in the **My Tickets** section.",
        agent_step="✅ Ticket Submitted",
    )
    st.session_state.portal_stage = "done"
    st.rerun()


def _render_done_stage():
    """Final state after ticket creation or self-resolution."""
    ticket = st.session_state.get("ticket_data")

    if ticket:
        st.success(f"✅ Ticket **{ticket.ticket_id}** has been submitted to the IT team.")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Start New Session", type="primary", use_container_width=True, key="new_session"):
            _reset_portal()
            st.rerun()
    with col2:
        if st.button("📋 View My Tickets", use_container_width=True, key="view_tickets"):
            st.session_state.app_page = "my_tickets"
            st.rerun()
