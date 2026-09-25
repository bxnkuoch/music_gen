"""Build the song pool from configs/pool_grid.yaml. Safe to stop (Ctrl+C) and re-run.

Usage:
    uv run python scripts/generate_pool.py               # generate everything pending
    uv run python scripts/generate_pool.py --limit 5     # just a few (quick check)
    uv run python scripts/generate_pool.py --status      # show progress, generate nothing

Requires the database: `docker compose up -d && uv run alembic upgrade head`.
"""

import argparse
import logging
from pathlib import Path

from music_gen.config import get_settings, resolve_device
from music_gen.db import make_session_factory
from music_gen.generation import AceStepGenerator
from music_gen.generation.acestep import MODEL_NAME
from music_gen.logging_setup import configure_logging
from music_gen.pool.grid import load_grid
from music_gen.pool.runner import enqueue_grid, job_status_counts, run_pending

logger = logging.getLogger("generate_pool")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/pool_grid.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="max jobs to run this time")
    parser.add_argument("--status", action="store_true", help="only print job counts")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = make_session_factory(settings.database_url)

    grid = load_grid(args.config)
    with session_factory() as session:
        new_jobs = enqueue_grid(session, grid, MODEL_NAME)
        logger.info("Grid: %d prompts, %d new jobs queued", len(grid.prompts), new_jobs)
        logger.info("Job status: %s", job_status_counts(session))
    if args.status:
        return

    generator = AceStepGenerator(settings.acestep_model_path, resolve_device(settings.device))
    summary = run_pending(session_factory, generator, settings.data_dir, limit=args.limit)
    logger.info(
        "Finished: %d generated, %d failed%s",
        summary.generated,
        summary.failed,
        " (stopped early, see errors above)" if summary.stopped_early else "",
    )
    with session_factory() as session:
        logger.info("Job status: %s", job_status_counts(session))


if __name__ == "__main__":
    main()
