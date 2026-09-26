"""Run the live-request worker (started by scripts/start_app.sh).

Usage:  uv run python scripts/worker.py
"""

import logging
import time
from pathlib import Path

import numpy as np

from music_gen.api.requests import RequestService
from music_gen.api.service import Recommender
from music_gen.config import get_settings, resolve_device
from music_gen.db import make_session_factory
from music_gen.embeddings import ClapEmbedder
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.generation import AceStepGenerator
from music_gen.generation.acestep import MODEL_NAME
from music_gen.logging_setup import configure_logging
from music_gen.personalization.data import DbSongSource, load_contexts
from music_gen.personalization.steering import load_request_words
from music_gen.pool.runner import requeue
from music_gen.worker import work_once

logger = logging.getLogger("worker")
IDLE_SLEEP_S = 1.0


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    device = resolve_device(settings.device)
    session_factory = make_session_factory(settings.database_url)
    vocab = load_tags(Path("configs/tags.yaml"))
    space = FeatureSpace.load(settings.data_dir / "models" / "feature_space_v1.npz")
    source = DbSongSource(space, vocab)

    with session_factory() as session:
        requeue(session, max_attempts=3)  # pick up anything interrupted last time
        recommender = Recommender(
            source.load(session), load_contexts(Path("configs/contexts.yaml")), source=source
        )
    requests = RequestService(
        recommender,
        vocab,
        load_request_words(Path("configs/request_words.yaml"), vocab),
        MODEL_NAME,
    )
    generator = AceStepGenerator(settings.acestep_model_path, device)
    embedder = ClapEmbedder(space.embedding_model.split(":", 1)[1], device)
    tag_text_vecs: np.ndarray = embedder.embed_text(vocab.texts)
    _ = generator.pipeline  # load the music model now, not on the first request

    logger.info("Worker ready; waiting for requests")
    while True:
        if not work_once(
            session_factory, generator, settings.data_dir, embedder, vocab, tag_text_vecs, requests
        ):
            time.sleep(IDLE_SLEEP_S)


if __name__ == "__main__":
    main()
