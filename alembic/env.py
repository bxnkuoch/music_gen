"""Alembic migration environment.

The database URL comes from music_gen settings (env var MUSIC_GEN_DATABASE_URL),
unless a caller (e.g. the test suite) sets `sqlalchemy.url` explicitly.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from music_gen.config import get_settings
from music_gen.db import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata
url = config.get_main_option("sqlalchemy.url") or get_settings().database_url

if context.is_offline_mode():
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    with create_engine(url, poolclass=pool.NullPool).connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
