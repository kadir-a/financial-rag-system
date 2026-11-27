from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

FAILURE_MODES = (
    "retrieval_miss", "parse_error", "wrong_period", "calc_error", "hallucination",
    "rewrite_error", "abstention_fail", "unit_error", "incomplete_answer",
)


class Base(DeclarativeBase):
    pass


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class Report(Timestamped, Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str]
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    company: Mapped[str | None]
    period: Mapped[str | None]
    page_count: Mapped[int | None]
    chroma_collection: Mapped[str] = mapped_column(unique=True)
    sessions: Mapped[list[ChatSession]] = relationship(back_populates="report")


class ChatSession(Timestamped, Base):
    __tablename__ = "chat_sessions"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), index=True)
    report: Mapped[Report] = relationship(back_populates="sessions")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class ChatMessage(Timestamped, Base):
    __tablename__ = "chat_messages"
    __table_args__ = (CheckConstraint("role IN ('user','assistant')", name="ck_message_role"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None]
    latency_ms: Mapped[float | None]
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    session: Mapped[ChatSession] = relationship(back_populates="messages")


class EvalRun(Timestamped, Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    dataset_name: Mapped[str]
    split: Mapped[str]
    model: Mapped[str | None]
    embedding_model: Mapped[str | None]
    k: Mapped[int]
    git_commit: Mapped[str | None]
    notes: Mapped[str]
    results: Mapped[list[EvalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint("run_id", "question_id", name="uq_run_question"),
        CheckConstraint(f"failure_mode IN {FAILURE_MODES!r}", name="ck_failure_mode"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[str]
    category: Mapped[str]
    answerable: Mapped[bool]
    answer: Mapped[str] = mapped_column(Text)
    correct: Mapped[bool | None]
    retrieval_hit: Mapped[bool | None]
    failure_mode: Mapped[str | None]
    latency_ms: Mapped[float | None]
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    run: Mapped[EvalRun] = relationship(back_populates="results")
