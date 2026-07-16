"""
ITAssist AI — Main Streamlit Application Entry Point
Streamlit + LangGraph + FAISS + SQLite + Groq

Run with:  streamlit run app.py
"""
# ITAssist AI Main Entrypoint - Hot reload triggered (Cached resolution update)
import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ITAssist AI — Service Desk Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/welcome-windows-client",
        "Report a bug": None,
        "About": "ITAssist AI — AI-Powered Service Desk Copilot | Built with Streamlit + LangGraph + FAISS + Groq",
    },
)

# ── Global CSS — Professional Enterprise Theme ───────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 14px; }
    .stApp { background: #212121; color: #ececec; }

    /* ── Remove default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    button[title="Collapse sidebar"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    /* Remove default top padding */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        padding: 8px 16px;
        transition: background 0.15s, transform 0.1s;
    }
    .stButton > button:hover { background: #1a7f64; color: white; }
    .stButton > button:active { transform: scale(0.98); }
    .stButton > button[kind="secondary"] {
        background: #2f2f2f;
        border: 1px solid #3f3f3f;
        color: #ececec;
    }
    .stButton > button[kind="secondary"]:hover { background: #3e3e3e; color: #ececec; }

    /* ── Text inputs & Textareas ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #2f2f2f !important;
        border: 1px solid #3f3f3f !important;
        color: #ececec !important;
        border-radius: 8px !important;
        font-size: 14px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16,163,127,0.2) !important;
        outline: none !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #2f2f2f !important;
        border: 1px solid #3f3f3f !important;
        color: #ececec !important;
        border-radius: 8px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #171717;
        border-radius: 8px;
        padding: 4px;
        border: none;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b4b4b4;
        font-weight: 500;
        font-size: 13px;
        border-radius: 6px;
        padding: 6px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #2f2f2f !important;
        color: #ececec !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #2f2f2f !important;
        border: 1px solid #3f3f3f !important;
        border-radius: 8px !important;
        color: #ececec !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #2f2f2f;
        border: 1px dashed #3f3f3f;
        border-radius: 8px;
        padding: 8px;
    }

    /* ── Alerts ── */
    .stAlert { border-radius: 8px; font-size: 13px; background: #2f2f2f; color: #ececec; border: 1px solid #3f3f3f; }

    /* ── Progress / Spinner ── */
    .stProgress > div > div > div { background: #10a37f; border-radius: 4px; }
    .stSpinner > div { border-top-color: #10a37f !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #171717; }
    ::-webkit-scrollbar-thumb { background: #4f4f4f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6f6f6f; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Initialize DB (cached — runs ONCE per server session, not on every rerun) ──
@st.cache_resource(show_spinner=False)
def _initialize_app():
    from database.models import init_db
    from database.crud import seed_knowledge_articles
    init_db()
    seed_knowledge_articles()
    # Pre-warm the embedding model so the first RAG search is instant
    try:
        from rag.retriever import is_index_ready, _get_embeddings
        if is_index_ready():
            _get_embeddings()  # Load sentence-transformers model into memory now
    except Exception:
        pass
    return True

_initialize_app()

# ── Session State ─────────────────────────────────────────────────────────────
if "app_username" not in st.session_state or not st.session_state.app_username:
    st.session_state.app_username = "Sachin"
if "app_role" not in st.session_state:
    st.session_state.app_role = "user"
if "app_page" not in st.session_state:
    st.session_state.app_page = "chat"


# ── Compact Top Nav Bar ──────────────────────────────────────────────────
# Inline nav: logo left, toggle right, no wasted space
st.markdown(
    """
    <div style="background:#171717; border-bottom:1px solid #2f2f2f;
                padding:12px 16px; margin-bottom:16px; border-radius:8px;
                display:flex; align-items:center;">
        <span style="font-size:18px; font-weight:700; color:#ececec;">🤖 ITAssist AI</span>
        <span style="font-size:12px; color:#b4b4b4; margin-left:10px; margin-top:2px;">Service Desk Copilot</span>
        <span style="flex:1;"></span>
        <span style="font-size:12px; color:#b4b4b4; margin-right:6px;">Signed in as <b>Sachin</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)
col_spacer, col_toggle = st.columns([8, 2])
with col_toggle:
    if st.session_state.app_page == "chat":
        if st.button("→ IT Portal", use_container_width=True, key="toggle_to_it"):
            st.session_state.app_page = "it_dashboard"
            st.session_state.app_role = "engineer"
            st.rerun()
    else:
        if st.button("← User Portal", use_container_width=True, key="toggle_to_user"):
            st.session_state.app_page = "chat"
            st.session_state.app_role = "user"
            st.rerun()

page = st.session_state.app_page
username = st.session_state.app_username
role = st.session_state.app_role

if page == "chat":
    from ui.user_portal import render_user_portal
    render_user_portal(username)

elif page == "my_tickets":
    from database.crud import get_all_tickets, get_or_create_user
    from ui.components import render_page_header, render_ticket_card, render_info_banner

    render_page_header("🎫 My Tickets", f"Support tickets submitted by {username}")

    user = get_or_create_user(username, role)
    all_tickets = get_all_tickets()
    # Filter by current user
    my_tickets = [t for t in all_tickets if t.get("username") == username]

    if not my_tickets:
        render_info_banner(
            "No Tickets Yet",
            "You haven't submitted any support tickets. Use the Chat with AI page to describe your issue.",
            icon="📭",
            color="#6366f1",
        )
    else:
        for ticket in my_tickets:
            render_ticket_card(ticket, compact=False)
            st.markdown("<br>", unsafe_allow_html=True)

elif page == "it_dashboard" and role == "engineer":
    from ui.it_dashboard import render_it_dashboard
    render_it_dashboard(username)

elif page == "all_tickets" and role == "engineer":
    from ui.it_dashboard import render_it_dashboard
    # Show dashboard with All Tickets tab pre-selected
    render_it_dashboard(username)

elif page == "knowledge_base" and role == "engineer":
    from ui.it_dashboard import render_it_dashboard
    render_it_dashboard(username)

else:
    # Fallback
    from ui.user_portal import render_user_portal
    render_user_portal(username)
