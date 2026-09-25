"""Schema tests: migrations stay in sync with the models, and constraints hold."""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from conftest import alembic_config
from sqlalchemy.exc import IntegrityError

from music_gen.db import Base, GenerationJob, Prompt, Song


def test_migrations_match_models(migrated_engine):
    # If this fails you changed db.py without a migration. Fix with:
    #   uv run alembic revision --autogenerate -m "describe the change"
    with migrated_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_migrations_downgrade_and_upgrade_cleanly(migrated_engine, db_url):
    cfg = alembic_config(db_url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


def make_prompt(session, text="chill r&b") -> Prompt:
    prompt = Prompt(text=text, attributes={"genre": "r&b"}, source="pool_grid")
    session.add(prompt)
    session.flush()
    return prompt


def make_job(prompt, **overrides) -> GenerationJob:
    fields = dict(prompt_id=prompt.id, seed=0, duration_s=30.0, bpm=None, model_name="m")
    return GenerationJob(**(fields | overrides))


def test_prompt_and_job_roundtrip(session_factory):
    with session_factory() as s:
        prompt = make_prompt(s)
        s.add(make_job(prompt, bpm=90, keyscale="A minor"))
        s.commit()
    with session_factory() as s:
        job = s.query(GenerationJob).one()
        assert job.prompt.text == "chill r&b"
        assert job.prompt.attributes == {"genre": "r&b"}
        assert (job.status, job.attempts, job.bpm) == ("pending", 0, 90)
        assert job.created_at is not None


def test_duplicate_prompt_text_is_rejected(session_factory):
    with session_factory() as s:
        make_prompt(s, "same")
        with pytest.raises(IntegrityError):
            make_prompt(s, "same")


def test_duplicate_job_is_rejected_even_when_bpm_is_null(session_factory):
    # Postgres treats NULLs as distinct by default; our NULLS NOT DISTINCT fixes that.
    with session_factory() as s:
        prompt = make_prompt(s)
        s.add(make_job(prompt, bpm=None))
        s.commit()
        s.add(make_job(prompt, bpm=None))
        with pytest.raises(IntegrityError):
            s.commit()


def test_invalid_job_status_is_rejected(session_factory):
    with session_factory() as s:
        s.add(make_job(make_prompt(s), status="exploded"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_song_requires_an_existing_job(session_factory):
    with session_factory() as s:
        s.add(
            Song(
                job_id=999,
                audio_path="x.wav",
                duration_s=1,
                sample_rate=48_000,
                generation_time_s=1,
                loudness_dbfs=-20,
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()
