"""
User Portal — Chat-App Style Interface
Full multi-turn workflow:
  1. User types their IT issue (bottom input bar)
  2. AI classifies + searches RAG knowledge base
  3. AI asks diagnostic questions → shown as chat bubble
  4. User answers via bottom input bar
  5. AI analyzes root cause + generates resolution
  6. Contextual action buttons appear above bottom bar (Resolved / Create Ticket)
  7. Ticket preview → Approve / Cancel / Edit
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


@st.dialog("Architecture & Agent Information")
def _show_arch_dialog():
    st.markdown("""
    ### Architecture
    - **Streamlit Frontend**: User Portal (Chat) & IT Engineer Dashboard
    - **LangGraph**: State Machine Orchestration
    - **Intake Agent (Groq LLM)**: Classifies user issues
    - **Knowledge Agent (FAISS RAG)**: Searches Microsoft SOP docs
    - **Troubleshoot Agent (Groq LLM)**: Asks diagnostic questions & analyzes root cause
    - **Resolution Agent**: Suggests self-resolution & creates enriched tickets
    - **Database**: SQLite (Tickets, Convos)
    
    ### Agent Details
    - **Intake Agent**: User message -> Category + confidence score
    - **Knowledge Agent**: Category + message -> Relevant SOP excerpts (FAISS RAG)
    - **Troubleshoot Agent**: Category + SOP context -> Diagnostic questions -> Root cause analysis
    - **Resolution Agent**: Ticket + root cause -> Resolution steps + PowerShell script
    """)

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
        "console_logs": [],
        "attached_files_context": "",
        "extracted_files_cache": {},
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
        "extracted_files_cache", "full_uploaded_text", "user_answers_input",
        "bottom_input",
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
            f'<span style="color:#10a37f; font-weight:700;">{item.get("agent","System Agent")}</span>'
            f'<span style="color:#b4b4b4;">{item.get("time","")}</span>'
            f'</div>'
            f'<div style="color:#e2e8f0; font-size:12px; line-height:1.5;">{item.get("detail","")}</div>'
            f'</div>'
        )
    return "".join(log_parts)


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    agent = " System Agent"
    detail = msg

    if "Guardrail" in msg or "guardrail" in msg:
        agent = " Safety Guardrail Agent"
        if "scanning" in msg.lower():
            detail = f"<b>Doing:</b> Running Relevance Classifier, Safety Validation, Moderation, and Safety Classification checks..."
        elif "BLOCKED" in msg:
            detail = f"<b>Blocked:</b> {msg.split('BLOCKED')[-1].strip()}"
        elif "cleared" in msg.lower() or "passed" in msg.lower():
            detail = "<b>Found:</b> All guardrail checks passed. Content is safe to process."
    elif "FAISS" in msg or "knowledge base" in msg.lower():
        agent = " RAG Search Agent"
        if "Scanning" in msg:
            detail = "<b>Doing:</b> Performing semantic search across Microsoft Windows troubleshooting knowledge base..."
        elif "Matched" in msg:
            detail = f"<b>Found:</b> {msg.replace('', '').strip()}"
        else:
            detail = f"<b>Info:</b> {msg.replace('', '').strip()}"
    elif "Classified" in msg or "classification" in msg.lower():
        agent = " Triage Agent"
        detail = f"<b>Found:</b> {msg.replace('', '').strip()}"
    elif "Groq" in msg or "classification" in msg.lower() or "cross-questions" in msg.lower():
        agent = " Triage Agent"
        detail = f"<b>Doing:</b> Calling Groq API (LLaMA-3.1) to classify issue category and check for existing context..."
    elif "Cross-questions" in msg or "cross-question" in msg.lower():
        agent = " Cross-Question Agent"
        detail = "<b>Doing:</b> Clarifying: Generating diagnostic verification questions for user response..."
    elif "Final resolution" in msg or "step-by-step" in msg.lower():
        agent = " Resolution Agent"
        detail = "<b>Found:</b> Step-by-step resolution recommendations and scripts generated successfully!"
    elif "Writing session" in msg:
        agent = " Resolution Agent"
        detail = "<b>Doing:</b> Compiling findings and writing to session_finding.txt state file..."
    elif "inventory" in msg.lower() or "" in msg:
        agent = " System Agent"
        detail = f"<b>Doing:</b> {msg.replace('', '').strip()}"
    elif "Ticket" in msg and "created" in msg.lower():
        agent = " System Agent"
        detail = f"<b>Done:</b> {msg.replace('', '').replace('', '').strip()}"
    elif "Reading" in msg or "" in msg:
        agent = " Ingestion Agent"
        detail = "<b>Doing:</b> Accessing uploaded attachments to extract context..."
    elif "Vision AI" in msg:
        agent = " Ingestion Agent"
        filename = msg.split("processed:")[-1].strip() if "processed:" in msg else ""
        detail = f"<b>Converting:</b> Extracted visual elements of <b>{filename}</b> using Vision API."
    elif "File context" in msg:
        agent = " Ingestion Agent"
        detail = "<b>Found:</b> Ingested text context successfully loaded into working memory."
    elif "error" in msg.lower() or "failed" in msg.lower():
        agent = " System Diagnostics"
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
        <div style="background:#171717; border:1px solid #2f2f2f; border-radius:12px; padding:14px; height:180px; overflow-y:auto;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid #2f2f2f; padding-bottom:0px;">
                <span style="color:#ececec; font-weight:700; font-size:14px; font-family:monospace;"> AI Console</span>
                {badge}
            </div>
            <div style="line-height:1.5;">
                {log_html}
            </div>
        </div>
        """
        st.session_state.console_placeholder.markdown(console_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Resolution Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_structured_resolution(chat_reply: str):
    code_blocks = re.findall(r"```(powershell|bash|cmd|shell|python)?\n(.*?)\n```", chat_reply, re.DOTALL | re.IGNORECASE)
    cleaned_reply = re.sub(r"```(?:powershell|bash|cmd|shell|python)?\n.*?\n```", "", chat_reply, flags=re.DOTALL | re.IGNORECASE)

    steps_start_match = re.search(r"(?:^|\n)\*?\*?Step\s+\d+", cleaned_reply, re.IGNORECASE)
    if steps_start_match:
        intro_text = cleaned_reply[:steps_start_match.start()].strip()
        steps_part = cleaned_reply[steps_start_match.start():].strip()
    else:
        intro_text = cleaned_reply.strip()
        steps_part = ""

    if intro_text:
        intro_text = re.sub(r"(?:^|\n)\*?Here are the recommended.*?\*?:?", "", intro_text, flags=re.IGNORECASE)
        intro_text = intro_text.strip()
        if intro_text:
            st.markdown(f"<div style='color:#ececec; font-size:14px; line-height:1.6; margin-bottom:14px;'>{intro_text}</div>", unsafe_allow_html=True)

    if steps_part:
        step_matches = list(re.finditer(r"(?:^|\n)\*?\*?Step\s+(\d+)\s*[:.-]\s*(.*?)(?=\n\*?\*?Step\s+\d+|\Z)", steps_part, re.DOTALL | re.IGNORECASE))
        if step_matches:
            steps_list = []
            for match in step_matches:
                step_num = match.group(1)
                step_content = match.group(2).strip()
                first_line_end = step_content.find("\n")
                if first_line_end != -1:
                    first_line = step_content[:first_line_end].strip()
                    rest = step_content[first_line_end:].strip()
                else:
                    first_line = step_content.strip()
                    rest = ""
                if first_line.endswith("**"):
                    first_line = first_line[:-2].strip()
                formatted_content = f"<b>{first_line}</b>: {rest}" if rest else f"<b>{first_line}</b>"
                steps_list.append(formatted_content)
            if steps_list:
                render_resolution_steps(steps_list)
        else:
            st.markdown(f"<div style='color:#ececec; font-size:14px; line-height:1.6; white-space:pre-wrap;'>{steps_part}</div>", unsafe_allow_html=True)

    if code_blocks:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        for lang, code in code_blocks:
            lang_name = "PowerShell" if (lang and lang.lower() == "powershell") else "Bash"
            render_script_block(code.strip(), script_type=lang_name)


# ─────────────────────────────────────────────────────────────────────────────
# File Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_and_cache_files(uploaded_files):
    """Scan files reactively, cache descriptions, add vision AI results as chat messages."""
    if not uploaded_files:
        return

    from utils.file_reader import extract_text_from_file, strip_pii_patterns

    for f in uploaded_files:
        file_key = f"{f.name}_{f.size}"
        if file_key in st.session_state.extracted_files_cache:
            continue

        raw_text, method = extract_text_from_file(f)
        confidence = 100
        description = raw_text

        if method == "image":
            if "CONFIDENCE:" in raw_text:
                try:
                    conf_line = [l for l in raw_text.split("\n") if "CONFIDENCE:" in l][0]
                    confidence = int(conf_line.split(":")[-1].replace("%", "").strip())
                except Exception:
                    confidence = 90
                try:
                    if "DESCRIPTION:" in raw_text:
                        description = raw_text.split("DESCRIPTION:", 1)[-1].strip()
                except Exception:
                    pass

        clean_text = strip_pii_patterns(description)
        st.session_state.extracted_files_cache[file_key] = {
            "text": clean_text,
            "method": method,
            "confidence": confidence,
        }

        # Add vision description as a chat message so it appears inline
        if method == "image":
            conf_color = "" if confidence >= 50 else ""
            conf_label = f"{conf_color} Vision AI ({confidence}% confidence)"
            if confidence < 50:
                vision_msg = (
                    f" **Attached:** `{f.name}`\n\n"
                    f" **Low Confidence ({confidence}%)** — The image is unclear or blurry. "
                    f"Please attach a clearer screenshot of your issue for better analysis."
                )
            else:
                vision_msg = (
                    f" **Attached:** `{f.name}` — {conf_label}\n\n"
                    f"**What I see in the image:** {clean_text[:400]}{'...' if len(clean_text) > 400 else ''}"
                )
            _add_chat("assistant", vision_msg, agent_step=" Vision AI")
        else:
            _add_chat(
                "assistant",
                f" **Attached:** `{f.name}` — {len(clean_text)} characters extracted and ready for analysis.",
                agent_step=" Document Reader"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────────────────────────────────────

def render_user_portal(username: str):
    _init_session()

    # Create 3 individually scrollable containers
    col_left, col_middle, col_right = st.columns([4, 3, 3])

    # ── RIGHT COLUMN — Asset Request ─────────────────────────────────────────
    with col_right:
        asset_container = st.container(height=500, border=True)
        with asset_container:


            # ── IT Asset Request Form ─────────────────────────────────────────────
            st.markdown(
                """
                <div style="color:#ececec; font-weight:700; font-size:15px; margin-bottom:16px;">
                     Request IT Asset
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<p style='color:#b4b4b4; font-size:12px; margin-bottom:8px;'>Manager (for approval):</p>", unsafe_allow_html=True)
            manager_name = st.text_input(
                "Manager:",
                value=st.session_state.get("user_manager_name", ""),
                placeholder="Enter manager name...",
                key="asset_manager_name",
                label_visibility="collapsed"
            )
            
            st.markdown("<p style='color:#b4b4b4; font-size:12px; margin-bottom:8px; margin-top:12px;'>Asset Request:</p>", unsafe_allow_html=True)
            justification = st.text_area(
                "Asset Request:",
                placeholder="Describe the asset you need (e.g., Headphone for work calls)...",
                height=250,
                key="asset_justification",
                label_visibility="collapsed"
            )

            if st.button("Submit Asset Request", type="primary", use_container_width=True, key="submit_asset_req"):
                if not justification.strip():
                    st.error("Please describe the asset you need.")
                elif not manager_name.strip():
                    st.error("Please enter manager name for approval.")
                else:
                    _log(" Guardrail scanning asset request...")
                    from utils.guardrail import guardrail_check
                    guard = guardrail_check(
                        user_issue=f"Asset Request: {justification.strip()}",
                        attached_doc_context=f"Manager: {manager_name.strip()}"
                    )
                    if not guard.safe:
                        st.error(f" Guardrail blocked: {guard.reason}")
                    else:
                        _log(" Guardrail: Content cleared")
                        stock = "Pending Review"
                        _log(f" Asset requested — {stock}")
                        _log(" Creating asset request ticket...")
                        from database.crud import get_session, get_or_create_user as _gcu
                        from database.models import Ticket
                        import random
                        user = _gcu(username, "user")
                        tid = f"SD-{random.randint(100, 999)}"
                        st.session_state.user_manager_name = manager_name.strip()
                        with get_session() as session:
                            session.add(Ticket(
                                ticket_id=tid, user_id=user.id,
                                category="Asset Request",
                                issue_summary=f"Asset Request (Manager: {manager_name.strip()})",
                                ai_analysis=f"Request Details: {justification.strip()}",
                                probable_cause=f"Inventory: {stock}",
                                severity="Medium", confidence=1.0,
                                status="Pending Manager Approval",
                                questions_asked=0, resolution_notes="No email drafted"
                            ))
                        _log(f" Ticket {tid} created (Pending Manager Approval)")
                        st.success(f" Ticket **{tid}** created — Pending Manager Approval")
                        _add_chat("user", f"I submitted an asset request.")
                        _add_chat(
                            "assistant",
                            f" **Asset Request Submitted!**\n\n"
                            f"- **Request:** {justification.strip()[:100]}...\n"
                            f"- **Manager:** {manager_name.strip()}\n"
                            f"- **Stock Status:** {stock}\n"
                            f"- **Ticket:** {tid} — Pending Manager Approval",
                            agent_step=" Asset Request"
                        )
                        st.session_state.asset_manager_name = ""
                        st.session_state.asset_justification = ""
                        st.rerun()

    # ── MIDDLE COLUMN — AI Console ───────────────────────────────────────────
    with col_middle:
        console_container = st.container(height=500, border=True)
        with console_container:
            stage = st.session_state.portal_stage
            is_live = stage in ["classifying", "generating_resolution"]
            badge = "<span style='background:#10b981;color:white;font-size:10px;padding:2px 7px;border-radius:10px;font-weight:600;'> LIVE</span>" if is_live else "<span style='background:#3f3f3f;color:#b4b4b4;font-size:10px;padding:2px 7px;border-radius:10px;'> IDLE</span>"

            console_placeholder = st.empty()
            logs = st.session_state.get("console_logs", [])
            log_html = _render_console_logs_html(logs)
            console_html = f"""
            <div style="background:#171717; border:1px solid #2f2f2f; border-radius:12px; padding:14px; height:460px; overflow-y:auto;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid #2f2f2f; padding-bottom:0px;">
                    <span style="color:#ececec; font-weight:700; font-size:14px; font-family:monospace;"> AI Console</span>
                    {badge}
                </div>
                <div style="line-height:1.5;">
                    {log_html}
                </div>
            </div>
            """
            console_placeholder.markdown(console_html, unsafe_allow_html=True)
            st.session_state.console_placeholder = console_placeholder
            st.session_state.console_badge = badge

    # ── LEFT COLUMN — Chat Interface ──────────────────────────────────────────
    with col_left:
        chat_container = st.container(height=500, border=True)
        with chat_container:

            if not is_index_ready():
                st.warning(
                    " **Knowledge Base Not Built Yet** — Run `python rag/ingest.py` to build the FAISS index.",
                    icon=None,
                )
            # Chat messages
            if not st.session_state.chat_history:
                st.markdown(
                    """
                    <div style="display:flex; flex-direction:column; align-items:center;
                                justify-content:center; height:400px; color:#4f4f4f; text-align:center;">
                        <div style="font-size:40px; margin-bottom:12px;"></div>
                        <div style="font-size:16px; font-weight:600; color:#6f6f6f;">ITAssist AI — Service Desk Copilot</div>
                        <div style="font-size:13px; color:#4f4f4f; margin-top:6px;">Describe your IT issue below or pick a common issue shortcut</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            for msg in st.session_state.chat_history:
                if msg["role"] == "assistant" and msg.get("agent_step") == " Service Desk Copilot Response":
                    step_badge = f'<span style="font-size:10px; background:#171717; color:#b4b4b4; padding:2px 8px; border-radius:10px; margin-bottom:6px; display:inline-block; border:1px solid #2f2f2f;">{msg.get("agent_step")}</span><br>'
                    st.markdown(
                        f"""
                        <div style="display:flex; margin:12px 0; align-items:flex-start;">
                            <div style="margin-right:8px; font-size:20px; padding-top:4px;"></div>
                            <div style="background:transparent; color:#ececec; padding:4px 12px; max-width:90%; font-size:14px; line-height:1.6;">
                                {step_badge}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    _, col_content = st.columns([0.04, 0.96])
                    with col_content:
                        render_structured_resolution(msg["content"])
                else:
                    render_chat_message(msg["role"], msg["content"], msg.get("agent_step", ""))


        # ── Stage: Spinner states (auto-run, no UI) ───────────────────────────
        stage = st.session_state.portal_stage
        if stage == "classifying":
            _run_classifying_stage(username)
        elif stage == "generating_resolution":
            _run_generating_resolution_stage()

        # ── Ticket Preview (inline in chat column) ────────────────────────────
        if stage == "ticket_preview":
            _render_ticket_preview_inline(username)

        # ── Done stage ────────────────────────────────────────────────────────
        if stage == "done":
            _render_done_stage()
            return

    # == BOTTOM INPUT BAR (Pinned to bottom of screen) =======================
    with st.bottom:
        # -- Contextual action buttons (self_resolve stage) -------------------
        if stage == "self_resolve":
            _render_self_resolve_actions(username)

        # -- Quick issue shortcuts (only at idle) -----------------------------
        if stage == "idle":
            SHORTCUTS = [
                ("VPN Error 800",       "VPN won't connect - error 800"),
                ("Slow PC / Freezing",  "My laptop is very slow and freezing"),
                ("BitLocker Key",       "BitLocker asking for recovery key"),
                ("Wi-Fi Dropping",      "Wi-Fi keeps disconnecting randomly"),
                ("Outlook Not Opening", "Outlook not opening or crashing on start"),
                ("Printer Offline",     "Printer showing offline and won't print"),
            ]
            st.markdown("<p style='font-size:11px; color:#64748b; margin:0 0 4px 0;'>Common issues:</p>", unsafe_allow_html=True)
            chip_cols = st.columns(6)
            for idx, (label, full_text) in enumerate(SHORTCUTS):
                with chip_cols[idx]:
                    if st.button(label, key=f"shortcut_{idx}", use_container_width=True):
                        st.session_state.bottom_input = full_text
                        st.session_state.bottom_input_area = full_text
                        st.rerun()

        # -- Unified Bottom Input Bar -----------------------------------------
        if stage in ("idle", "answering_questions"):
            _render_bottom_input_bar(username, stage)

        st.markdown(
            """
            <style>
            /* Center the Architecture button */
            .st-key-arch_btn button {
                background: transparent !important;
                border: none !important;
                color: #64748b !important;
                font-size: 12px !important;
                box-shadow: none !important;
                padding: 0px !important;
                min-height: auto !important;
                line-height: 1.2 !important;
            }
            .st-key-arch_btn button:hover {
                color: #ececec !important;
                background: transparent !important;
            }
            .st-key-arch_btn div.stButton {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
            }
            .st-key-arch_btn {
                width: 100% !important;
                margin-top: 4px !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stBottom"], 
            div[data-testid="stBottom"] > div, 
            div[data-testid="stBottom"] div[class*="st-emotion-cache-"] {
                padding: 0px !important;
                padding-bottom: 0px !important;
                margin-bottom: 0px !important;
            }
            
            /* Make the bottom section controls more compact */
            div[data-testid="stBottom"] div[data-testid="column"] {
                gap: 6px !important;
            }
            div[data-testid="stBottom"] .stTextArea {
                margin-bottom: -5px !important;
            }
            div[data-testid="stBottom"] .streamlit-expanderHeader {
                padding: 3px 6px !important;
                font-size: 12px !important;
                min-height: 0px !important;
            }
            div[data-testid="stBottom"] .streamlit-expanderContent {
                padding: 6px !important;
            }
            div[data-testid="stBottom"] hr {
                margin: 4px 0 !important;
            }
            div[data-testid="stBottom"] .stButton button {
                padding-top: 0px !important;
                padding-bottom: 0px !important;
                font-size: 13px !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        if st.button("Architecture ⚙", key="arch_btn"):
            _show_arch_dialog()


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Bottom Input Bar
# ─────────────────────────────────────────────────────────────────────────────

def _on_bottom_submit():
    raw_val = st.session_state.get("bottom_input_area", "").strip()
    if raw_val:
        st.session_state.pending_bottom_msg = raw_val
    st.session_state.bottom_input = ""
    st.session_state.bottom_input_area = ""


def _render_bottom_input_bar(username: str, stage: str):
    send_label = " Analyze My Issue" if stage == "idle" else "→ Submit Answer"
    placeholder = (
        "Describe your IT issue here..." if stage == "idle"
        else "Type your diagnostic answers here..."
    )

    # File uploader in an expander-style toggle
    with st.expander(" Attach files (screenshots, PDFs, images)", expanded=False):
        uploaded_files = st.file_uploader(
            "Attach files",
            type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "txt", "log"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
            help="Files are read in memory only.",
        )
        if uploaded_files:
            new_files = [f for f in uploaded_files if f"{f.name}_{f.size}" not in st.session_state.extracted_files_cache]
            if new_files:
                with st.spinner(" Reading and scanning files..."):
                    _extract_and_cache_files(new_files)
                st.rerun()
            else:
                names = ", ".join(f.name for f in uploaded_files)
                st.caption(f" {len(uploaded_files)} file(s) ready: {names}")

    user_input = st.text_area(
        "Message",
        value=st.session_state.get("bottom_input", ""),
        height=48,
        placeholder=placeholder,
        label_visibility="collapsed",
        key="bottom_input_area",
    )

    col_send, col_reset = st.columns([4, 1])
    with col_send:
        send = st.button(
            send_label, 
            type="primary", 
            use_container_width=True, 
            key="bottom_send_btn",
            on_click=_on_bottom_submit
        )
    with col_reset:
        if st.button(" Reset", use_container_width=True, key="bottom_reset_btn"):
            _reset_portal()
            st.rerun()

    pending = st.session_state.get("pending_bottom_msg", "").strip()
    if pending:
        st.session_state.pending_bottom_msg = ""
        if stage == "idle":
            _handle_idle_submit(username, pending)
        elif stage == "answering_questions":
            _handle_answer_submit(pending)
    elif send:
        st.error("Please type a message before sending.")


# ─────────────────────────────────────────────────────────────────────────────
# Submit Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_idle_submit(username: str, text: str):
    # Greeting intercept
    greeting_pat = r"^\s*(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|yo|sup)\s*[.!?]*\s*$"
    if re.match(greeting_pat, text.lower()):
        _add_chat("user", text)
        _add_chat(
            "assistant",
            " **Hello! Welcome to the IT Support Portal.**\n\n"
            "I'm your IT Service Desk Copilot. Describe your IT issue below and I'll help you troubleshoot it step by step!",
            agent_step=" Greeting"
        )
        st.rerun()
        return

    # Extract file context from cache
    uploaded_files = st.session_state.get("file_uploader") or []
    if uploaded_files:
        _log(f" Reading {len(uploaded_files)} attached file(s) from cache...")
        from utils.file_reader import semantic_search_in_doc
        parts, full_parts = [], []
        for f in uploaded_files:
            cached = st.session_state.extracted_files_cache.get(f"{f.name}_{f.size}")
            if cached:
                clean_text = cached["text"]
                full_parts.append(f"[{f.name} - Full Content]:\n{clean_text}")
                relevant = semantic_search_in_doc(text, clean_text, top_k=3)
                if relevant:
                    parts.append(f"[{f.name}]:\n{relevant}")
        st.session_state.attached_files_context = "\n\n".join(parts)
        st.session_state.full_uploaded_text = "\n\n".join(full_parts)
        _log(" File context ready")

    _add_chat("user", text)
    st.session_state.user_issue = text
    st.session_state.portal_stage = "classifying"
    st.rerun()


def _handle_answer_submit(text: str):
    _add_chat("user", text)
    st.session_state.user_answers_input = text
    st.session_state.portal_stage = "generating_resolution"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Background Processing Stages (no UI — run then rerun)
# ─────────────────────────────────────────────────────────────────────────────

def _run_classifying_stage(username: str):
    user_message = st.session_state.get("user_issue", "")
    # Align truncation to avoid truncation bypass in safety scan vs main agent
    if len(user_message) > 3000:
        user_message = user_message[:3000] + "\n[...content truncated...]"
        st.session_state.user_issue = user_message

    # 1. Try local category classification first
    from agents.intake_agent import classify_issue_locally
    local_result = classify_issue_locally(user_message)
    
    # 2. Check if we can perform a direct local SOP resolution immediately (0 Groq calls)
    is_direct_local = False
    if local_result and local_result.get("confidence", 0.0) >= 0.80:
        category = local_result.get("category", "Other")
        if category != "Other":
            from rag.retriever import parse_sop_file_locally
            local_diag = parse_sop_file_locally(category, user_message)
            if local_diag and local_diag.get("self_resolution_steps"):
                is_direct_local = True
                cat = category
                conf = local_result["confidence"]
                icon = CATEGORY_ICONS.get(cat, "")
                steps = local_diag["self_resolution_steps"]

    if is_direct_local:
        # Bypasses all Groq calls! Run only the local Layer 1 guardrail checks
        from utils.guardrail import decode_obfuscated_text, check_credentials_deterministic
        decoded = decode_obfuscated_text(user_message)
        has_creds, cred_reason = check_credentials_deterministic(decoded)
        if has_creds:
            _log(f" Local Guardrail BLOCKED: {cred_reason}")
            st.error(f" **Guardrail blocked this request**\n\n**Reason:** {cred_reason}")
            st.session_state.portal_stage = "idle"
            st.rerun()
            return
            
        steps_str = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        
        # Build initial AgentState locally to seed the state machine
        agent_state = {
            "user_message": user_message,
            "session_id": st.session_state.session_id,
            "category": cat,
            "intake_confidence": conf,
            "issue_summary": user_message[:100],
            "sop_context": local_diag.get("analysis", "Local SOP Match"),
            "diagnostic_questions": "DIRECT_RESOLUTION",
            "user_answers": "",
            "analysis_text": steps_str,
            "problem_title": local_diag.get("problem", cat),
            "probable_cause": local_diag.get("probable_cause", "Matched local SOP"),
            "self_resolution_steps": [steps_str],
            "conversation_history": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"**Here are the recommended troubleshooting steps from the local SOP:**\n\n{steps_str}"}
            ],
            "ticket_id": "",
            "resolution_notes": "",
            "direct_local": True,
        }
        st.session_state.agent_state = agent_state
        
        _add_chat(
            "assistant",
            f"**Issue Classified Locally:** {icon} **{cat}** (confidence: {conf:.0%})\n\n"
            f"**Knowledge Base Match:** Direct SOP resolution loaded locally (0 API Calls).",
            agent_step=" Local Triage"
        )
        _log(" Direct local SOP resolution loaded (0 API Calls)")
        _add_chat(
            "assistant",
            f"**Here are the recommended troubleshooting steps from the local SOP:**\n\n{steps_str}",
            agent_step=" Service Desk Copilot Response"
        )
        st.session_state.portal_stage = "self_resolve"
        st.rerun()
        return

    # 3. Fallback to normal flow if not direct local: runs FAISS and full Groq Guardrails
    attached_context = st.session_state.get("attached_files_context", "")
    with st.spinner(" Checking safety & searching knowledge base..."):
        _log(" Scanning FAISS knowledge base...")
        from rag.retriever import search as rag_search
        try:
            local_docs = rag_search(user_message, k=3)
            local_context = "\n\n".join(d.get("content", "") for d in local_docs) if local_docs else ""
            _log(f' Matched {len(local_docs)} local SOP article(s)' if local_docs else " No exact local SOP match — using general knowledge")
        except Exception:
            local_context = ""
            _log(" FAISS search skipped (index not ready)")

        _log(" Guardrail scanning input + attachments...")
        from utils.guardrail import guardrail_check, build_enriched_context
        guard = guardrail_check(
            user_issue=user_message,
            local_rag_context=local_context,
            attached_doc_context=st.session_state.get("full_uploaded_text", ""),
        )

        if not guard.safe:
            _log(f" Guardrail BLOCKED: {guard.reason[:60]}")
            st.error(f" **Guardrail blocked this request**\n\n**Reason:** {guard.reason}")
            st.session_state.portal_stage = "idle"
            st.rerun()
            return

        _log(" Guardrail: Content cleared")
        enriched_context = build_enriched_context(
            user_issue=user_message,
            local_rag_context=local_context,
            attached_doc_context=attached_context,
        )
        augmented_message = f"{user_message}\n\n--- CONTEXT ---\n{enriched_context}"

    with st.spinner(" Classifying issue and generating diagnostic questions..."):
        try:
            _log(" Sending to Groq for classification + cross-questions...")
            agent_state = run_intake_and_knowledge(
                user_message=augmented_message,
                session_id=st.session_state.session_id,
            )
            st.session_state.agent_state = agent_state

            cat = agent_state.get("category", "Other")
            conf = agent_state.get("intake_confidence", 0.0)
            icon = CATEGORY_ICONS.get(cat, "")
            articles = agent_state.get("articles_used", [])
            _log(f" Classified: {cat} ({conf:.0%} confidence)")

            _add_chat(
                "assistant",
                f"**Issue Classified:** {icon} **{cat}** (confidence: {conf:.0%})\n\n"
                f"**Knowledge Base Searched:** {', '.join(articles) if articles else 'General SOP documentation'}"
                + (f"\n\n *Attached document context included in analysis.*" if attached_context else ""),
                agent_step=" Intake +  Knowledge Agents",
            )

            questions = agent_state.get("diagnostic_questions", "")
            direct_local = agent_state.get("direct_local", False)

            if questions == "DIRECT_RESOLUTION" or direct_local:
                # Direct local resolution path: fetch steps directly from SOP
                from rag.retriever import parse_sop_file_locally
                user_issue_text = st.session_state.get("user_issue", "")
                local_diag = parse_sop_file_locally(cat, user_issue_text)
                if local_diag:
                    steps = local_diag.get("self_resolution_steps", [])
                    steps_str = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
                    
                    # Store details in state for the dashboard and self-resolution views
                    agent_state["analysis_text"] = steps_str
                    agent_state["self_resolution_steps"] = [steps_str]
                    agent_state["problem_title"] = local_diag.get("problem", cat)
                    agent_state["probable_cause"] = local_diag.get("probable_cause", "Matched local SOP")
                    
                    _log(" Direct local SOP resolution loaded")
                    _add_chat(
                        "assistant",
                        f"**Here are the recommended troubleshooting steps from the local SOP:**\n\n{steps_str}",
                        agent_step=" Service Desk Copilot Response"
                    )
                else:
                    _add_chat(
                        "assistant",
                        "Matched category locally but failed to retrieve specific SOP resolution steps.",
                        agent_step=" Intake Agent"
                    )
                st.session_state.portal_stage = "self_resolve"
            else:
                _log(" Cross-questions generated — awaiting user answers")
                _add_chat(
                    "assistant",
                    f"**To verify the exact cause, please answer these diagnostic questions:**\n\n{questions}",
                    agent_step=" Diagnostic Questions",
                )
                st.session_state.portal_stage = "answering_questions"

        except Exception as e:
            _log(f" Agent error: {str(e)[:50]}")
            st.error(f"AI processing error: {str(e)}")
            st.session_state.portal_stage = "idle"

    st.rerun()


def _run_generating_resolution_stage():
    agent_state = st.session_state.agent_state
    if not agent_state:
        st.session_state.portal_stage = "idle"
        st.rerun()
        return

    user_answers = st.session_state.get("user_answers_input", "")
    # Align truncation limits to avoid truncation bypass
    if len(user_answers) > 3000:
        user_answers = user_answers[:3000] + "\n[...content truncated...]"
        st.session_state.user_answers_input = user_answers

    with st.spinner(" Checking answer safety..."):
        _log(" Guardrail scanning user answers...")
        from utils.guardrail import guardrail_check
        guard = guardrail_check(user_issue=user_answers)
        if not guard.safe:
            _log(f" Guardrail BLOCKED answers: {guard.reason[:60]}")
            st.error(f" **Guardrail blocked:** {guard.reason}")
            st.session_state.portal_stage = "answering_questions"
            st.rerun()
            return
        _log(" Guardrail: Answers cleared")

    with st.spinner(" Generating troubleshooting steps..."):
        try:
            _log(" Writing session_finding.txt...")
            from agents.graph import generate_final_resolution
            updated_state = generate_final_resolution(
                state=agent_state,
                user_answers_text=user_answers
            )
            st.session_state.agent_state = updated_state
            chat_reply = updated_state.get("analysis_text", "")

            # Output Moderation Check (Layer 3) on generated response text
            from utils.guardrail import guardrail_output_check
            output_guard = guardrail_output_check(chat_reply)
            if not output_guard.safe:
                _log(f" Output Guardrail BLOCKED: {output_guard.reason[:60]}")
                st.error(f" **Safety Violation Blocked:** {output_guard.reason}")
                st.session_state.portal_stage = "answering_questions"
                st.rerun()
                return

            _log(" Final resolution ready!")
            _add_chat(
                "assistant",
                f"**Here are the recommended troubleshooting steps:**\n\n{chat_reply}",
                agent_step=" Service Desk Copilot Response"
            )
            st.session_state.portal_stage = "self_resolve"
        except Exception as e:
            _log(f" Resolution error: {str(e)[:50]}")
            st.error(f"Resolution generation error: {str(e)}")
            st.session_state.portal_stage = "self_resolve"

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Self Resolve — contextual action buttons above the bottom bar
# ─────────────────────────────────────────────────────────────────────────────

def _render_self_resolve_actions(username: str):
    agent_state = st.session_state.agent_state
    if not agent_state:
        return

    chat_reply = agent_state.get("analysis_text", "")

    # Email draft
    if "email_draft" not in st.session_state:
        if st.button(" Draft Support Email", use_container_width=True, key="draft_email_btn"):
            with st.spinner("Writing email draft..."):
                from agents.graph import draft_support_email
                draft = draft_support_email(
                    user_message=agent_state.get("user_message", ""),
                    chat_reply=chat_reply
                )
                # Output Moderation Check (Layer 3) on email draft
                from utils.guardrail import guardrail_output_check
                output_guard = guardrail_output_check(draft)
                if not output_guard.safe:
                    _log(f" Output Guardrail BLOCKED email: {output_guard.reason[:60]}")
                    st.error(f" **Safety Warning:** Generated support email contains unsafe content and was blocked: {output_guard.reason}")
                else:
                    st.session_state.email_draft = draft
            st.rerun()

    if st.session_state.get("email_draft"):
        with st.expander(" Support Email Draft", expanded=True):
            st.text_area("Draft", value=st.session_state.email_draft, height=180, key="email_draft_area")
            if st.button(" Regenerate", key="regen_email_btn"):
                del st.session_state.email_draft
                st.rerun()

    st.markdown("<p style='font-size:12px; color:#64748b; font-weight:600; margin:8px 0 6px 0;'>Did the steps resolve your issue?</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Yes — Issue Fixed!", type="primary", use_container_width=True, key="resolved_btn"):
            _add_chat("user", " The issue is resolved. No support ticket needed.")
            _add_chat(
                "assistant",
                " **Wonderful! I'm glad the steps helped.**\n\nNo ticket was created. Let me know if anything else comes up!",
                agent_step=" Resolved",
            )
            st.session_state.portal_stage = "done"
            st.rerun()

    with col2:
        direct_local = agent_state.get("direct_local", False)
        btn_label = " No — Still Having Issues" if direct_local else " No — Create Ticket"
        
        if st.button(btn_label, use_container_width=True, key="not_resolved_btn"):
            if direct_local:
                _add_chat("user", "No, the local steps didn't resolve my issue.")
                st.session_state.user_answers_input = "Local SOP troubleshooting steps failed to resolve the issue. Please suggest standard dynamic diagnostics and provide alternative resolution."
                agent_state["direct_local"] = False
                st.session_state.portal_stage = "generating_resolution"
            else:
                _add_chat("user", " I'd like to create a support ticket.")
                _add_chat(
                    "assistant",
                    "Certainly! I've prepared a ticket preview for you. Please review and approve it below.",
                    agent_step=" Ticket Creator",
                )
                st.session_state.portal_stage = "ticket_preview"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Ticket Preview — inline in left column
# ─────────────────────────────────────────────────────────────────────────────

def _render_ticket_preview_inline(username: str):
    agent_state = st.session_state.agent_state
    if not agent_state:
        return

    st.markdown(
        """
        <div style="background:#171717; border:1px solid #2f2f2f; border-radius:12px; padding:0px; margin:12px 0 8px 0;">
            <div style="font-size:15px; font-weight:700; color:#ececec; margin-bottom:4px;"> Support Ticket Preview</div>
            <div style="color:#b4b4b4; font-size:12px;">Review the AI-generated ticket. You can edit the summary before approving.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        cat = agent_state.get("category", "Other")
        st.markdown(
            f"""<div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1px;">Category</div>
                <div style="color:#e2e8f0; font-size:15px; font-weight:700; margin-top:4px;">{CATEGORY_ICONS.get(cat, "")} {cat}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        sev = agent_state.get("severity", "Medium")
        sev_color = SEVERITY_COLORS.get(sev, "#f59e0b")
        conf = agent_state.get("analysis_confidence", 0.0)
        st.markdown(
            f"""<div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:1px;">Severity · Confidence</div>
                <div style="font-size:15px; font-weight:700; margin-top:4px;">
                    <span style="font-size:12px; color:#b4b4b4; margin-right:6px;">Signed in as <b>{username}</b></span>
                    <span style="color:{sev_color};"> {sev}</span>
                    <span style="color:#10a37f; margin-left:12px;">{conf:.0%}</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    with st.expander(" AI Analysis Details", expanded=False):
        st.markdown(f"**Problem:** {agent_state.get('problem_title', '')}")
        st.markdown(f"**Probable Cause:** {agent_state.get('probable_cause', '')}")
        st.markdown(f"**Analysis:** {agent_state.get('analysis_text', '')[:300]}...")

    default_summary = agent_state.get("issue_summary", agent_state.get("user_message", ""))
    edited_summary = st.text_area(
        " Edit Issue Summary (optional):",
        value=st.session_state.get("edit_summary", default_summary),
        height=72,
        key="ticket_summary_edit",
    )
    st.session_state.edit_summary = edited_summary

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button(" Approve & Submit", type="primary", use_container_width=True, key="approve_ticket"):
            _submit_ticket(username, agent_state, edited_summary)
    with col_b:
        if st.button(" Edit Note", use_container_width=True, key="edit_ticket"):
            st.info("Edit the summary above, then click Approve.")
    with col_c:
        if st.button(" Cancel", use_container_width=True, key="cancel_ticket"):
            _add_chat("user", " Ticket cancelled.")
            _add_chat("assistant", "Ticket cancelled. Let me know if you need further help.", agent_step="Cancelled")
            st.session_state.portal_stage = "done"
            st.rerun()
    with col_d:
        if st.button(" Reset", use_container_width=True, key="reset_ticket_btn"):
            _reset_portal()
            st.rerun()


def _submit_ticket(username: str, agent_state: dict, edited_summary: str):
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
        f" **Ticket Created Successfully!**\n\n"
        f"**Ticket ID:** {ticket.ticket_id}\n"
        f"**Status:** Open\n\n"
        f"An IT engineer will review your ticket shortly. Track it in the **My Tickets** section.",
        agent_step=" Ticket Submitted",
    )
    st.session_state.portal_stage = "done"
    st.rerun()


def _render_done_stage():
    ticket = st.session_state.get("ticket_data")
    if ticket:
        st.success(f" Ticket **{ticket.ticket_id}** has been submitted to the IT team.")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Start New Session", type="primary", use_container_width=True, key="new_session"):
            _reset_portal()
            st.rerun()
    with col2:
        if st.button(" View My Tickets", use_container_width=True, key="view_tickets"):
            st.session_state.app_page = "my_tickets"
            st.rerun()
