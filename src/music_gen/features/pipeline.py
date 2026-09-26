"""Featurize one song right after it's generated (used by the live-request worker).

scripts/compute_features.py does the same in bulk for the pool (plus MuQ-MuLan).
"""

from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from music_gen import audio
from music_gen.db import Song
from music_gen.embeddings import AudioTextEmbedder
from music_gen.features import signal, store
from music_gen.features.tags import TagVocabulary, score_tags


def featurize_song(
    session: Session,
    song: Song,
    data_dir: Path,
    embedder: AudioTextEmbedder,
    vocab: TagVocabulary,
    tag_text_vecs: np.ndarray,
) -> None:
    """Signal features + embedding + tag scores: everything the feature space needs.

    Doesn't commit; the caller decides when.
    """
    clip, sr = audio.load_audio(data_dir / song.audio_path)
    store.save_features(
        session, song.id, signal.FEATURE_SET, signal.compute_signal_features(clip, sr)
    )
    vec = embedder.embed_audio(clip, sr)
    store.save_embedding(session, song.id, embedder.name, vec)
    scores = score_tags(vec[None, :], tag_text_vecs)[0]
    store.save_features(
        session,
        song.id,
        vocab.feature_set(embedder.name),
        dict(zip(vocab.names, map(float, scores), strict=True)),
    )
