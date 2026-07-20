"""
ITAssist AI — Main Streamlit Application Entry Point
Streamlit + LangGraph + FAISS + SQLite + Groq

Run with:  streamlit run app.py
"""
# ITAssist AI Main Entrypoint - Updated UI layout with 3-column structure
import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ITAssist AI — Service Desk Copilot",
    page_icon=None,
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
    .stApp { 
        background: #212121; 
        color: #ececec;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ── Remove default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    button[title="Collapse sidebar"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    /* ── Maximize Width - 5px padding on all sides ── */
    .main .block-container {
        padding: 5px !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Target all possible container elements - remove padding */
    section[data-testid="stMain"],
    section.main,
    .stApp > section,
    div[data-testid="stAppViewContainer"],
    .appview-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Remove horizontal padding from main content divs */
    .st-emotion-cache-liupih,
    [data-testid="stMain"] > div,
    .main > div,
    section.main > div,
    .stApp > header + section > div {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Force horizontal blocks to use full width */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
        gap: 0.5rem !important;
        width: 100% !important;
    }
    
    div[data-testid="column"] {
        padding: 0 0.25rem !important;
    }

    /* ── Sticky bottom input bar ── */
    .sticky-bottom-bar {
        position: sticky;
        bottom: 0;
        background: #212121;
        border-top: 1px solid #2f2f2f;
        padding-top: 10px;
        z-index: 100;
    }

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
    
    # Try to load the vector store; if missing/corrupt, auto-rebuild it dynamically
    try:
        from rag.retriever import _get_vectorstore, _get_embeddings
        vs = _get_vectorstore()
        if vs is None:
            print("FAISS index is missing or corrupt on this environment. Rebuilding index from local documents...")
            from rag.ingest import ingest
            # Build without requiring the large external PDF if it's missing
            ingest()
        else:
            # Pre-warm embeddings
            _get_embeddings()
    except Exception as e:
        print(f"Failed to load or initialize FAISS index: {e}. Rebuilding index...")
        try:
            from rag.ingest import ingest
            ingest()
        except Exception as ie:
            print(f"Failed to ingest FAISS index: {ie}")
            
    return True

_initialize_app()

# ── Session State ─────────────────────────────────────────────────────────────
if "app_username" not in st.session_state or not st.session_state.app_username:
    st.session_state.app_username = "user"
if "app_role" not in st.session_state:
    st.session_state.app_role = "user"
if "app_page" not in st.session_state:
    st.session_state.app_page = "chat"


# ── Compact Top Nav Bar (with inline portal toggle) ──────────────────────
_toggle_label = "→ IT Portal" if st.session_state.app_page == "chat" else "← User Portal"
_toggle_key = "toggle_portal_nav"

_nav_col_main, _nav_col_btn = st.columns([7, 2])
with _nav_col_main:
    st.markdown(
        """
        <div style="display:flex; align-items:center; height:42px;">
            <span style="font-size:18px; font-weight:700; color:#ececec;">ITAssist AI</span>
            <span style="font-size:13px; color:#b4b4b4; margin-left:12px; margin-top:2px;">Service Desk Copilot</span>
            <span style="flex:1;"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with _nav_col_btn:
    if st.button(_toggle_label, use_container_width=True, key=_toggle_key):
        if st.session_state.app_page == "chat":
            st.session_state.app_page = "it_dashboard"
            st.session_state.app_role = "engineer"
        else:
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

    render_page_header(" My Tickets", f"Support tickets submitted by {username}")

    user = get_or_create_user(username, role)
    all_tickets = get_all_tickets()
    # Filter by current user
    my_tickets = [t for t in all_tickets if t.get("username") == username]

    if not my_tickets:
        render_info_banner(
            "No Tickets Yet",
            "You haven't submitted any support tickets. Use the Chat with AI page to describe your issue.",
            icon=None,
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
