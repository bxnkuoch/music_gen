"""Shared test fixtures.

Database tests use a separate database (`music_gen_test`, created by
docker/init-test-db.sql) that is wiped and rebuilt from the Alembic migrations,
so they never touch your real data.
"""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from music_gen.config import Settings


def alembic_config(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False  # keep pytest's logging intact
    return cfg


@pytest.fixture(scope="session")
def db_url() -> str:
    return Settings(_env_file=None).test_database_url


@pytest.fixture(scope="session")
def migrated_engine(db_url):
    engine = create_engine(db_url)
    try:
        engine.connect().close()
    except OperationalError:
        pytest.skip("Postgres is not running. Start it with: docker compose up -d")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    command.upgrade(alembic_config(db_url), "head")
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(migrated_engine) -> sessionmaker:
    """A clean, empty, fully migrated database for each test."""
    with migrated_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE slate_items, slates, rating_sessions, song_embeddings, song_features, "
                "songs, generation_jobs, prompts "
                "RESTART IDENTITY CASCADE"
            )
        )
    return sessionmaker(migrated_engine, expire_on_commit=False)
