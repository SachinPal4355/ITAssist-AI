"""
Reusable Streamlit UI components for ITAssist AI.
"""
import streamlit as st


# ── Color Map ─────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
    "Critical": "#7c3aed",
}

STATUS_COLORS = {
    "Open": "#3b82f6",
    "In Progress": "#f59e0b",
    "Resolved": "#22c55e",
    "Escalated": "#ef4444",
    "Cancelled": "#6b7280",
}

CATEGORY_ICONS = {
    "Performance": "",
    "VPN / Remote Access": "",
    "Network": "",
    "Email": "",
    "Access / Permissions": "",
    "Software": "",
    "Hardware": "",
    "Backup / Storage": "",
    "Printer": "",
    "Security / BitLocker": "",
    "Other": "",
}


# ── Chat Bubble ────────────────────────────────────────────────────────────────

def render_chat_message(role: str, content: str, agent_step: str = ""):
    """Render a styled chat bubble resembling ChatGPT."""
    if role == "user":
        st.markdown(
            f"""
            <div style="
                display: flex; justify-content: flex-end; margin: 12px 0; align-items: flex-start;
            ">
                <div style="
                    background: #2f2f2f;
                    color: #ececec; padding: 10px 16px; border-radius: 18px 18px 4px 18px;
                    max-width: 75%; font-size: 14px; line-height: 1.5;
                    border: 1px solid #3f3f3f;
                ">
                    {content}
                </div>
                <div style="margin-left: 8px; font-size: 20px; padding-top: 4px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        step_badge = ""
        if agent_step:
            step_badge = f'<span style="font-size:10px; background:#171717; color:#b4b4b4; padding:2px 8px; border-radius:10px; margin-bottom:6px; display:inline-block; border: 1px solid #2f2f2f;">{agent_step}</span><br>'
        st.markdown(
            f"""
            <div style="display: flex; margin: 12px 0; align-items: flex-start;">
                <div style="margin-right: 8px; font-size: 20px; padding-top: 4px;"></div>
                <div style="
                    background: transparent; color: #ececec;
                    padding: 4px 12px;
                    max-width: 85%; font-size: 14px; line-height: 1.6;
                ">
                    {step_badge}{content.replace(chr(10), "<br>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Metric Card ───────────────────────────────────────────────────────────────

def render_metric_card(label: str, value, icon: str = "", color: str = "#10a37f", delta: str = ""):
    """Render a clean minimal metric card."""
    delta_html = f'<div style="font-size:12px; color:#b4b4b4; margin-top:4px;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div style="
            background: #2f2f2f;
            border: 1px solid #3f3f3f;
            border-radius: 12px; padding: 20px; text-align: center;
        ">
            <div style="font-size: 28px;">{icon}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin: 6px 0;">{value}</div>
            <div style="font-size: 13px; color: #b4b4b4; font-weight: 500;">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Agent Step Badge ──────────────────────────────────────────────────────────

def render_agent_step(step_name: str, active: bool = False):
    """Show which agent is currently active."""
    steps = ["intake", "knowledge", "ask_questions", "analyze", "generate_resolution", "complete"]
    labels = {
        "intake": " Intake Agent",
        "knowledge": " Knowledge Agent",
        "ask_questions": " Troubleshoot Agent",
        "analyze": " Analysis",
        "generate_resolution": " Resolution Agent",
        "complete": " Complete",
    }
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        label = labels.get(step, step)
        is_current = (step == step_name)
        is_done = steps.index(step) < steps.index(step_name) if step_name in steps else False
        if is_current:
            color = "#10a37f"
            bg = "#1f352f"
            border = "1px solid #10a37f"
        elif is_done:
            color = "#2e7d32"
            bg = "#1b301c"
            border = "1px solid #2e7d32"
        else:
            color = "#b4b4b4"
            bg = "#171717"
            border = "1px solid #2f2f2f"
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center; font-size:11px; color:{color}; background:{bg}; padding:6px 2px; border-radius:6px; border:{border}; font-weight:500;">{label}</div>',
                unsafe_allow_html=True,
            )


# ── Ticket Card ───────────────────────────────────────────────────────────────

def render_ticket_card(ticket: dict, compact: bool = False):
    """Render a full or compact ticket card."""
    sev_color = SEVERITY_COLORS.get(ticket.get("severity", "Medium"), "#f59e0b")
    status_color = STATUS_COLORS.get(ticket.get("status", "Open"), "#3b82f6")
    cat_icon = CATEGORY_ICONS.get(ticket.get("category", "Other"), "")
    confidence = ticket.get("confidence", 0.0)

    if compact:
        st.markdown(
            f"""
            <div style="
                background: #2f2f2f; border: 1px solid #3f3f3f;
                border-left: 4px solid {sev_color};
                border-radius: 10px; padding: 14px; margin: 6px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight:700; color:#ececec; font-size:15px;">
                        {cat_icon} {ticket.get('ticket_id', '')}
                    </span>
                    <span style="background:{status_color}22; color:{status_color}; font-size:11px; padding:3px 10px; border-radius:20px; border:1px solid {status_color};">
                        {ticket.get('status', 'Open')}
                    </span>
                </div>
                <div style="color:#b4b4b4; font-size:13px; margin-top:6px;">
                    {ticket.get('issue_summary', '')[:80]}...
                </div>
                <div style="display:flex; gap:12px; margin-top:8px;">
                    <span style="color:{sev_color}; font-size:12px;"> {ticket.get('severity', 'Medium')}</span>
                    <span style="color:#b4b4b4; font-size:12px;"> {ticket.get('username', '')}</span>
                    <span style="color:#b4b4b4; font-size:12px;"> {ticket.get('created_at', '')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background: #2f2f2f;
                border: 1px solid #3f3f3f; border-radius: 14px;
                padding: 24px; margin: 12px 0;
            ">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="font-size:20px; font-weight:800; color:#ececec;">
                            {cat_icon} {ticket.get('ticket_id', '')}
                        </div>
                        <div style="color:#b4b4b4; font-size:13px; margin-top:4px;">
                            {ticket.get('category', '')} |  {ticket.get('username', '')} |  {ticket.get('created_at', '')}
                        </div>
                    </div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <span style="background:{sev_color}22; color:{sev_color}; padding:4px 12px; border-radius:20px; border:1px solid {sev_color}; font-size:12px; font-weight:600;">
                             {ticket.get('severity', 'Medium')}
                        </span>
                        <span style="background:{status_color}22; color:{status_color}; padding:4px 12px; border-radius:20px; border:1px solid {status_color}; font-size:12px; font-weight:600;">
                            {ticket.get('status', 'Open')}
                        </span>
                    </div>
                </div>

                <div style="margin-top:16px; padding:14px; background:#171717; border-radius:8px; border:1px solid #2f2f2f;">
                    <div style="color:#b4b4b4; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Issue Summary</div>
                    <div style="color:#ececec; font-size:14px; margin-top:6px;">{ticket.get('issue_summary', '')}</div>
                </div>

                <div style="margin-top:12px; display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                    <div style="padding:12px; background:#171717; border-radius:8px; border:1px solid #2f2f2f;">
                        <div style="color:#b4b4b4; font-size:11px; font-weight:600; text-transform:uppercase;">Probable Cause</div>
                        <div style="color:#fbbf24; font-size:13px; margin-top:4px;">{ticket.get('probable_cause', 'Under investigation')}</div>
                    </div>
                    <div style="padding:12px; background:#171717; border-radius:8px; border:1px solid #2f2f2f;">
                        <div style="color:#b4b4b4; font-size:11px; font-weight:600; text-transform:uppercase;">AI Confidence</div>
                        <div style="color:#10a37f; font-size:20px; font-weight:700; margin-top:4px;">{confidence:.0%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Info Banner ───────────────────────────────────────────────────────────────

def render_info_banner(title: str, content: str, color: str = "#10a37f", icon: str = ""):
    st.markdown(
        f"""
        <div style="
            background: #2f2f2f; border: 1px solid #3f3f3f;
            border-left: 4px solid {color}; border-radius: 10px;
            padding: 14px 18px; margin: 10px 0;
        ">
            <div style="font-weight:700; color:{color}; font-size:14px;">{icon} {title}</div>
            <div style="color:#ececec; font-size:13px; margin-top:6px; line-height:1.5;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Resolution Steps Card ─────────────────────────────────────────────────────

def render_resolution_steps(steps: list[str]):
    """Render numbered resolution steps."""
    st.markdown(
        '<div style="background:#2f2f2f; border:1px solid #3f3f3f; border-radius:12px; padding:16px;">',
        unsafe_allow_html=True,
    )
    for i, step in enumerate(steps, 1):
        st.markdown(
            f"""
            <div style="display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-bottom:1px solid #3f3f3f;">
                <div style="
                    min-width:26px; height:26px; background:#10a37f;
                    border-radius:50%; display:flex; align-items:center; justify-content:center;
                    font-weight:700; font-size:11px; color:white;
                ">{i}</div>
                <div style="color:#ececec; font-size:13px; line-height:1.6; padding-top:2px;">{step}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Script Block ──────────────────────────────────────────────────────────────

def render_script_block(script: str, script_type: str = "PowerShell"):
    """Render syntax-highlighted script with copy note."""
    lang = "powershell" if "PowerShell" in script_type else "bash"
    st.markdown(
        f'<div style="color:#b4b4b4; font-size:12px; margin-bottom:4px;"> {script_type} Script — Copy and run as Administrator</div>',
        unsafe_allow_html=True,
    )
    st.code(script, language=lang)


# ── Page Header ───────────────────────────────────────────────────────────────

def render_page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            background: #171717;
            border: 1px solid #2f2f2f; border-radius: 12px;
            padding: 24px 28px; margin-bottom: 20px;
        ">
            <h1 style="
                color: #ececec; font-size: 24px; font-weight: 700; margin: 0;
            ">{title}</h1>
            <p style="color: #b4b4b4; margin: 6px 0 0 0; font-size: 13px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
