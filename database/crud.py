"""
CRUD helpers for ITAssist AI database.
All functions use a context-managed session to ensure safety.
"""
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy.orm import Session
from database.models import (
    engine,
    init_db,
    User,
    Ticket,
    Conversation,
    KnowledgeArticle,
    TicketStatus,
)
import uuid


@contextmanager
def get_session():
    session = Session(bind=engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Users ─────────────────────────────────────────────────────────────────────

def get_or_create_user(username: str, role: str = "user") -> User:
    with get_session() as session:
        user = session.query(User).filter_by(username=username).first()
        if not user:
            user = User(username=username, role=role)
            session.add(user)
            session.flush()
        session.expunge(user)
        return user


# ── Tickets ───────────────────────────────────────────────────────────────────

def generate_ticket_id() -> str:
    with get_session() as session:
        count = session.query(Ticket).count()
        return f"SD-{count + 101:04d}"


def create_ticket(
    user_id: int,
    category: str,
    issue_summary: str,
    ai_analysis: str = "",
    probable_cause: str = "",
    severity: str = "Medium",
    confidence: float = 0.0,
    questions_asked: int = 0,
    sop_articles_used: str = "",
    resolution_notes: str = "",
) -> Ticket:
    ticket_id = generate_ticket_id()
    with get_session() as session:
        ticket = Ticket(
            ticket_id=ticket_id,
            user_id=user_id,
            category=category,
            issue_summary=issue_summary,
            ai_analysis=ai_analysis,
            probable_cause=probable_cause,
            severity=severity,
            confidence=confidence,
            status=TicketStatus.OPEN,
            questions_asked=questions_asked,
            sop_articles_used=sop_articles_used,
            resolution_notes=resolution_notes,
        )
        session.add(ticket)
        session.flush()
        session.expunge(ticket)
        return ticket


def get_all_tickets(status: str = None) -> list[dict]:
    with get_session() as session:
        q = session.query(Ticket)
        if status:
            q = q.filter(Ticket.status == status)
        tickets = q.order_by(Ticket.created_at.desc()).all()
        result = []
        for t in tickets:
            user = session.query(User).filter_by(id=t.user_id).first()
            result.append({
                "id": t.id,
                "ticket_id": t.ticket_id,
                "username": user.username if user else "Unknown",
                "category": t.category,
                "issue_summary": t.issue_summary,
                "ai_analysis": t.ai_analysis,
                "probable_cause": t.probable_cause,
                "severity": t.severity,
                "confidence": t.confidence,
                "status": t.status,
                "questions_asked": t.questions_asked,
                "sop_articles_used": t.sop_articles_used,
                "resolution_notes": t.resolution_notes,
                "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
            })
        return result


def get_ticket_by_id(ticket_id: str) -> dict | None:
    with get_session() as session:
        t = session.query(Ticket).filter_by(ticket_id=ticket_id).first()
        if not t:
            return None
        user = session.query(User).filter_by(id=t.user_id).first()
        convos = (
            session.query(Conversation)
            .filter_by(ticket_id=t.id)
            .order_by(Conversation.timestamp)
            .all()
        )
        return {
            "id": t.id,
            "ticket_id": t.ticket_id,
            "username": user.username if user else "Unknown",
            "category": t.category,
            "issue_summary": t.issue_summary,
            "ai_analysis": t.ai_analysis,
            "probable_cause": t.probable_cause,
            "severity": t.severity,
            "confidence": t.confidence,
            "status": t.status,
            "questions_asked": t.questions_asked,
            "sop_articles_used": t.sop_articles_used,
            "resolution_notes": t.resolution_notes,
            "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
            "conversations": [
                {"role": c.role, "message": c.message, "agent_step": c.agent_step}
                for c in convos
            ],
        }


def update_ticket_status(ticket_id: str, status: str, resolution_notes: str = "") -> bool:
    with get_session() as session:
        t = session.query(Ticket).filter_by(ticket_id=ticket_id).first()
        if not t:
            return False
        t.status = status
        t.updated_at = datetime.utcnow()
        if resolution_notes:
            t.resolution_notes = resolution_notes
        return True


def get_ticket_stats() -> dict:
    with get_session() as session:
        total = session.query(Ticket).count()
        open_ = session.query(Ticket).filter_by(status=TicketStatus.OPEN).count()
        in_progress = session.query(Ticket).filter_by(status="In Progress").count()
        resolved = session.query(Ticket).filter_by(status="Resolved").count()
        return {
            "total": total,
            "open": open_,
            "in_progress": in_progress,
            "resolved": resolved,
        }


# ── Conversations ──────────────────────────────────────────────────────────────

def log_message(
    session_id: str,
    role: str,
    message: str,
    agent_step: str = "",
    ticket_id: int | None = None,
):
    with get_session() as session:
        convo = Conversation(
            session_id=session_id,
            ticket_id=ticket_id,
            role=role,
            message=message,
            agent_step=agent_step,
        )
        session.add(convo)


def get_session_messages(session_id: str) -> list[dict]:
    with get_session() as session:
        convos = (
            session.query(Conversation)
            .filter_by(session_id=session_id)
            .order_by(Conversation.timestamp)
            .all()
        )
        return [
            {"role": c.role, "message": c.message, "agent_step": c.agent_step}
            for c in convos
        ]


# ── Knowledge Articles ────────────────────────────────────────────────────────

def seed_knowledge_articles():
    """Seed the knowledge_articles table from the SOP file list."""
    articles = [
        {
            "title": "Windows Performance Troubleshooting SOP",
            "filename": "performance_sop.txt",
            "category": "Performance",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/performance-overview",
        },
        {
            "title": "Windows Networking & VPN Troubleshooting SOP",
            "filename": "networking_vpn_sop.txt",
            "category": "VPN / Remote Access",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/networking-overview",
        },
        {
            "title": "Windows Printing Troubleshooting SOP",
            "filename": "printing_sop.txt",
            "category": "Printer",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/printing/printing-overview",
        },
        {
            "title": "BitLocker & Windows Security SOP",
            "filename": "security_bitlocker_sop.txt",
            "category": "Security / BitLocker",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/windows-security/bitlocker-recovery-known-issues",
        },
        {
            "title": "Backup, Storage & File Access SOP",
            "filename": "backup_storage_sop.txt",
            "category": "Backup / Storage",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/backup-and-storage/backup-and-storage-overview",
        },
        {
            "title": "Email & Outlook Troubleshooting SOP",
            "filename": "email_outlook_sop.txt",
            "category": "Email",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/application-management/application-management-overview",
        },
        {
            "title": "Active Directory, User Accounts & Access SOP",
            "filename": "active_directory_access_sop.txt",
            "category": "Access / Permissions",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/active-directory/active-directory-overview",
        },
        {
            "title": "Windows Update Troubleshooting SOP",
            "filename": "windows_update_sop.txt",
            "category": "Windows Update",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/performance-overview",
        },
        {
            "title": "Hardware & Device Driver Troubleshooting SOP",
            "filename": "hardware_device_sop.txt",
            "category": "Hardware",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/welcome-windows-client",
        },
        {
            "title": "Remote Desktop & Remote Access SOP",
            "filename": "remote_desktop_teams_sop.txt",
            "category": "Remote Desktop",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/remote/remote-desktop-services-overview",
        },
        {
            "title": "Software Installation & Application Management SOP",
            "filename": "software_application_sop.txt",
            "category": "Software",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/application-management/application-management-overview",
        },
        {
            "title": "Wireless Networking & 802.1X Authentication SOP",
            "filename": "wireless_networking_sop.txt",
            "category": "Wireless / WiFi",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/networking-overview",
        },
        {
            "title": "Windows System Recovery & Restore SOP",
            "filename": "system_recovery_sop.txt",
            "category": "System Recovery",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/welcome-windows-client",
        },
        {
            "title": "Windows Defender & Antivirus / Malware SOP",
            "filename": "windows_defender_security_sop.txt",
            "category": "Security / BitLocker",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/windows-security/windows-security-overview",
        },
        {
            "title": "OneDrive, SharePoint & Cloud Storage SOP",
            "filename": "onedrive_sharepoint_sop.txt",
            "category": "OneDrive / Cloud",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/backup-and-storage/backup-and-storage-overview",
        },
        {
            "title": "Windows Power Management & Sleep/Hibernate SOP",
            "filename": "power_management_sop.txt",
            "category": "Power / Sleep",
            "source_url": "https://learn.microsoft.com/en-us/troubleshoot/windows-client/welcome-windows-client",
        },
    ]
    with get_session() as session:
        # Check and seed only missing articles
        for a in articles:
            exists = session.query(KnowledgeArticle).filter_by(filename=a["filename"]).first()
            if not exists:
                article = KnowledgeArticle(**a)
                session.add(article)


def get_all_knowledge_articles() -> list[dict]:
    with get_session() as session:
        arts = session.query(KnowledgeArticle).all()
        return [
            {
                "id": a.id,
                "title": a.title,
                "filename": a.filename,
                "category": a.category,
                "source_url": a.source_url,
            }
            for a in arts
        ]
