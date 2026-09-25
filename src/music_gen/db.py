"""Database tables (SQLAlchemy ORM) and session setup.

Schema changes go through Alembic migrations (`alembic/versions/`), never by editing
the live database. tests/test_db.py fails if these models and the migrations disagree.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

JOB_STATUSES = ("pending", "running", "done", "failed")


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    """A text description to generate from, plus the structured attributes behind it."""

    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text, unique=True)
    # e.g. {"genre": "r&b", "mood": "chill", "energy": "low", "instrument": "rhodes piano"}
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source: Mapped[str] = mapped_column(String(32))  # "pool_grid" now; "user" in Phase 4
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["GenerationJob"]] = relationship(back_populates="prompt")


class GenerationJob(Base):
    """One request to generate one clip. Tracks status so interrupted runs can resume."""

    __tablename__ = "generation_jobs"
    __table_args__ = (
        # Same prompt + settings + model = same job; re-running the grid adds nothing.
        # NULLS NOT DISTINCT: two jobs with bpm=NULL still count as duplicates.
        UniqueConstraint(
            "prompt_id",
            "seed",
            "duration_s",
            "bpm",
            "model_name",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(f"status IN {JOB_STATUSES}", name="valid_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"))
    seed: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float] = mapped_column(Float)
    bpm: Mapped[int | None] = mapped_column(Integer)
    keyscale: Mapped[str | None] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    prompt: Mapped[Prompt] = relationship(back_populates="jobs")
    song: Mapped["Song | None"] = relationship(back_populates="job")


class Song(Base):
    """A successfully generated clip. Generation settings live on its job."""

    __tablename__ = "songs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("generation_jobs.id"), unique=True)
    audio_path: Mapped[str] = mapped_column(Text)  # relative to settings.data_dir
    duration_s: Mapped[float] = mapped_column(Float)
    sample_rate: Mapped[int] = mapped_column(Integer)
    generation_time_s: Mapped[float] = mapped_column(Float)
    loudness_dbfs: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[GenerationJob] = relationship(back_populates="song")


def make_session_factory(database_url: str) -> sessionmaker:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)
