"""Database tables (SQLAlchemy ORM) and session setup.

Schema changes go through Alembic migrations (`alembic/versions/`), never by editing
the live database. tests/test_db.py fails if these models and the migrations disagree.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
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
        CheckConstraint("variant IS NULL OR variant IN ('steered', 'plain')", name="valid_variant"),
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
    # Set for live requests (Phase 4); NULL for pool jobs.
    request_id: Mapped[int | None] = mapped_column(
        ForeignKey("generation_requests.id", name="generation_jobs_request_id_fkey"), index=True
    )
    variant: Mapped[str | None] = mapped_column(String(16))  # "steered" | "plain"
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


EMBEDDING_DIM = 512  # both CLAP and MuQ-MuLan produce 512-d vectors


class SongEmbedding(Base):
    """One song's embedding from one model. Several models can coexist per song."""

    __tablename__ = "song_embeddings"

    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))  # L2-normalized
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SongFeatures(Base):
    """A named, versioned set of scalar features for one song, e.g. "signal_v1"."""

    __tablename__ = "song_features"

    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), primary_key=True)
    feature_set: Mapped[str] = mapped_column(String(160), primary_key=True)
    values: Mapped[dict[str, float]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


REQUEST_STATUSES = ("generating", "ready", "failed")


class ListenerRequest(Base):
    """A listener's request for new songs, e.g. "chill lofi for studying"."""

    __tablename__ = "generation_requests"
    __table_args__ = (CheckConstraint(f"status IN {REQUEST_STATUSES}", name="valid_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSONB)  # tags + energy found in the text
    # Each request gets its own rating session; its result slate lives in that session.
    session_id: Mapped[int] = mapped_column(ForeignKey("rating_sessions.id"), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="generating", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["GenerationJob"]] = relationship(order_by="GenerationJob.id")
    rating_session: Mapped["RatingSession"] = relationship()


class RatingSession(Base):
    """One sitting of rating, in one mood ("context")."""

    __tablename__ = "rating_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    context: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    slates: Mapped[list["Slate"]] = relationship(back_populates="session")


class Slate(Base):
    """A set of songs shown together, and the listener's answer."""

    __tablename__ = "slates"
    __table_args__ = (
        CheckConstraint("worst_song_id IS NULL OR worst_song_id <> chosen_song_id"),
        CheckConstraint("(chosen_song_id IS NULL) = (answered_at IS NULL)", name="answer_complete"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("rating_sessions.id"), index=True)
    policy: Mapped[str] = mapped_column(String(32))  # how songs were chosen
    model_name: Mapped[str] = mapped_column(String(32))  # which model scored them
    n_training_choices: Mapped[int] = mapped_column(Integer)  # answers the model had seen
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    chosen_song_id: Mapped[int | None] = mapped_column(ForeignKey("songs.id"))
    worst_song_id: Mapped[int | None] = mapped_column(ForeignKey("songs.id"))
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[RatingSession] = relationship(back_populates="slates")
    items: Mapped[list["SlateItem"]] = relationship(
        back_populates="slate", order_by="SlateItem.position"
    )


class SlateItem(Base):
    """One song in a slate. `source` and `score` are logged but never shown."""

    __tablename__ = "slate_items"
    __table_args__ = (
        UniqueConstraint("slate_id", "position"),
        CheckConstraint("source IN ('model', 'random')", name="valid_source"),
    )

    slate_id: Mapped[int] = mapped_column(ForeignKey("slates.id"), primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)  # display order (shuffled)
    source: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)

    slate: Mapped[Slate] = relationship(back_populates="items")


def make_session_factory(database_url: str) -> sessionmaker:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)
