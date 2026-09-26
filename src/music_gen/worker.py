"""Background worker for live requests: generate -> featurize -> rank.

Runs as its own process (scripts/worker.py) because it holds the ~10 GB music
model; the API stays light and responsive. They communicate only through the
database (the job queue), so either can restart without losing work.
"""

import logging
from pathlib import Path

import numpy as np
from sqlalchemy.orm import sessionmaker

from music_gen.api.requests import RequestService
from music_gen.db import Song
from music_gen.embeddings import AudioTextEmbedder
from music_gen.features.pipeline import featurize_song
from music_gen.features.tags import TagVocabulary
from music_gen.pool.runner import Generator, process_next_job

logger = logging.getLogger(__name__)


def work_once(
    session_factory: sessionmaker,
    generator: Generator,
    data_dir: Path,
    embedder: AudioTextEmbedder,
    vocab: TagVocabulary,
    tag_text_vecs: np.ndarray,
    requests: RequestService,
) -> bool:
    """Do one unit of work. Returns False when there was nothing to do."""
    outcome = process_next_job(session_factory, generator, data_dir)
    if outcome is not None:
        if outcome.song_id is None:
            logger.warning("job %d failed: %s", outcome.job_id, outcome.error)
        else:
            logger.info("job %d generated in %.1fs", outcome.job_id, outcome.elapsed_s)
            with session_factory() as session:
                try:
                    song = session.get_one(Song, outcome.song_id)
                    featurize_song(session, song, data_dir, embedder, vocab, tag_text_vecs)
                    session.commit()
                except Exception:  # noqa: BLE001 - an unrankable song mustn't stop the worker
                    session.rollback()
                    logger.exception("featurizing song %d failed", outcome.song_id)

    with session_factory() as session:
        for request_id in requests.finalize_ready(session):
            logger.info("request %d ranked and ready", request_id)
    return outcome is not None
