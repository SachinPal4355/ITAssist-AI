"""
SQLAlchemy ORM models for ITAssist AI.
Tables: users, tickets, conversations, knowledge_articles
"""
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Enum,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum
from config.settings import DATABASE_URL


class Base(DeclarativeBase):
    pass


class TicketStatus(str, enum.Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    ESCALATED = "Escalated"
    CANCELLED = "Cancelled"


class TicketSeverity(str, enum.Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class UserRole(str, enum.Enum):
    USER = "user"
    ENGINEER = "engineer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    role = Column(String(20), default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow)

    tickets = relationship("Ticket", back_populates="user")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(20), unique=True, nullable=False)  # SD-101
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category = Column(String(50), nullable=False)
    issue_summary = Column(Text, nullable=False)
    ai_analysis = Column(Text, nullable=True)
    probable_cause = Column(Text, nullable=True)
    severity = Column(String(20), default=TicketSeverity.MEDIUM)
    confidence = Column(Float, default=0.0)
    status = Column(String(20), default=TicketStatus.OPEN)
    questions_asked = Column(Integer, default=0)
    resolution_notes = Column(Text, nullable=True)
    sop_articles_used = Column(Text, nullable=True)  # comma-separated titles
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="tickets")
    conversations = relationship("Conversation", back_populates="ticket")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=True)
    session_id = Column(String(100), nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    agent_step = Column(String(50), nullable=True)  # which agent produced this
    timestamp = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="conversations")


class KnowledgeArticle(Base):
    __tablename__ = "knowledge_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    filename = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    source_url = Column(String(500), nullable=True)
    content_preview = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Engine & init ─────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def init_db():
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
