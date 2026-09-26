"""Compute embeddings + features for every song that doesn't have them yet, then fit
the feature space. Safe to re-run (e.g. after the pool grows).

Usage:  uv run python scripts/compute_features.py
Writes: song_embeddings / song_features rows, data/models/feature_space_v1.npz
"""

import logging
from pathlib import Path

import numpy as np
from sqlalchemy.orm import sessionmaker

from music_gen import audio
from music_gen.config import Settings, get_settings, resolve_device
from music_gen.db import make_session_factory
from music_gen.embeddings import AudioTextEmbedder, ClapEmbedder
from music_gen.features import signal, store
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import TagVocabulary, load_tags, score_tags
from music_gen.logging_setup import configure_logging

logger = logging.getLogger("compute_features")

TAGS_PATH = Path("configs/tags.yaml")
FEATURE_SPACE_VERSION = "v1"
N_COMPONENTS = 32
COMMIT_EVERY = 25


def compute_signal(session_factory: sessionmaker, settings: Settings) -> None:
    with session_factory() as session:
        songs = store.songs_missing_features(session, signal.FEATURE_SET)
        logger.info("Signal features: %d songs to process", len(songs))
        for i, song in enumerate(songs, 1):
            clip, sr = audio.load_audio(settings.data_dir / song.audio_path)
            values = signal.compute_signal_features(clip, sr)
            store.save_features(session, song.id, signal.FEATURE_SET, values)
            if i % COMMIT_EVERY == 0:
                session.commit()
                logger.info("  signal %d/%d", i, len(songs))
        session.commit()


def compute_embeddings(
    session_factory: sessionmaker, settings: Settings, embedder: AudioTextEmbedder
) -> None:
    with session_factory() as session:
        songs = store.songs_missing_embedding(session, embedder.name)
        logger.info("%s: %d songs to embed", embedder.name, len(songs))
        for i, song in enumerate(songs, 1):
            clip, sr = audio.load_audio(settings.data_dir / song.audio_path)
            store.save_embedding(session, song.id, embedder.name, embedder.embed_audio(clip, sr))
            if i % COMMIT_EVERY == 0:
                session.commit()
                logger.info("  embedded %d/%d", i, len(songs))
        session.commit()


def compute_tags(
    session_factory: sessionmaker, embedder: AudioTextEmbedder, vocab: TagVocabulary
) -> None:
    """Tag scores come from the stored embeddings, so no audio is re-read."""
    feature_set = vocab.feature_set(embedder.name)
    with session_factory() as session:
        missing = {s.id for s in store.songs_missing_features(session, feature_set)}
        ids, vecs = store.load_embeddings(session, embedder.name)
        rows = [i for i, sid in enumerate(ids) if sid in missing]
        logger.info("%s: %d songs to tag", feature_set, len(rows))
        if not rows:
            return
        scores = score_tags(vecs[rows], embedder.embed_text(vocab.texts))
        for row, song_scores in zip(rows, scores, strict=True):
            values = dict(zip(vocab.names, map(float, song_scores), strict=True))
            store.save_features(session, ids[row], feature_set, values)
        session.commit()


def fit_feature_space(
    session_factory: sessionmaker, settings: Settings, embedder_name: str, vocab: TagVocabulary
) -> None:
    with session_factory() as session:
        song_ids, emb, tabular, names = store.load_feature_table(
            session,
            embedder_name,
            {vocab.feature_set(embedder_name): "tag", signal.FEATURE_SET: "signal"},
        )
    space = FeatureSpace.fit(
        emb,
        tabular,
        names,
        n_components=N_COMPONENTS,
        version=FEATURE_SPACE_VERSION,
        embedding_model=embedder_name,
    )
    path = settings.data_dir / "models" / f"feature_space_{FEATURE_SPACE_VERSION}.npz"
    space.save(path)
    logger.info(
        "Feature space %s: %d songs x %d features (%d PCs keep %.0f%% of embedding variance) -> %s",
        FEATURE_SPACE_VERSION,
        len(song_ids),
        len(space.names),
        N_COMPONENTS,
        100 * float(np.sum(space.explained_variance_ratio)),
        path,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    device = resolve_device(settings.device)
    session_factory = make_session_factory(settings.database_url)
    vocab = load_tags(TAGS_PATH)

    compute_signal(session_factory, settings)

    embedders: list[AudioTextEmbedder] = [ClapEmbedder(settings.clap_model_id, device)]
    try:
        from music_gen.embeddings.muq_mulan import MuQMuLanEmbedder

        embedders.append(MuQMuLanEmbedder(settings.muq_model_id, device))
    except ImportError:
        logger.warning("muq not installed (uv sync --extra muq); skipping MuQ-MuLan")

    for embedder in embedders:
        compute_embeddings(session_factory, settings, embedder)
        compute_tags(session_factory, embedder, vocab)

    # The preference model's features use CLAP (the permissively licensed default).
    fit_feature_space(session_factory, settings, embedders[0].name, vocab)


if __name__ == "__main__":
    main()
