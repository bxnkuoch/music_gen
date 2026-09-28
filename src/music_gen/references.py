"""Reference songs (Phase 6): upload a real song you like, then
  * find the most similar songs in the library (pgvector search), and
  * generate new songs that imitate its sound (ACE-Step reference audio).

The API only checks and stores the upload, so it stays light. The worker embeds it
with the serving embedding model, the same way it embeds every generated song.
"""

import io
import logging
import os
from pathlib import Path

import soundfile as sf
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from music_gen import audio
from music_gen.db import ReferenceSong
from music_gen.embeddings import AudioTextEmbedder
from music_gen.features import store

logger = logging.getLogger(__name__)

REFERENCE_AUDIO_DIR = Path("audio/references")  # relative to settings.data_dir
ALLOWED_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg")  # what libsndfile can decode
MAX_BYTES = 50 * 2**20
MIN_DURATION_S = 5.0


class InvalidReference(ValueError):
    pass


def save_reference(session: Session, data_dir: Path, filename: str, data: bytes) -> ReferenceSong:
    """Check an upload is usable audio, store it, and queue it for embedding."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InvalidReference(f"unsupported file type; use one of {', '.join(ALLOWED_SUFFIXES)}")
    if len(data) > MAX_BYTES:
        raise InvalidReference(f"file is larger than {MAX_BYTES // 2**20} MB")
    try:
        info = sf.info(io.BytesIO(data))
    except RuntimeError as e:  # soundfile's LibsndfileError
        raise InvalidReference("couldn't read this file as audio") from e
    if info.duration < MIN_DURATION_S:
        raise InvalidReference(f"song must be at least {MIN_DURATION_S:.0f} s long")

    reference = ReferenceSong(
        filename=Path(filename).name, audio_path="", duration_s=info.duration, status="pending"
    )
    session.add(reference)
    session.flush()  # get the id for the file name
    rel_path = REFERENCE_AUDIO_DIR / f"ref_{reference.id:06d}{suffix}"
    path = data_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)  # atomic: the worker never reads a half-written file
    reference.audio_path = str(rel_path)
    session.commit()
    return reference


def embed_pending_references(
    session_factory: sessionmaker, data_dir: Path, embedder: AudioTextEmbedder
) -> int:
    """Embed every uploaded reference that isn't yet. Returns how many were processed."""
    with session_factory() as session:
        pending = session.scalars(
            select(ReferenceSong)
            .where(ReferenceSong.status == "pending")
            .order_by(ReferenceSong.id)
        ).all()
        for reference in pending:
            try:
                clip, sr = audio.load_audio(data_dir / reference.audio_path)
                vec = embedder.embed_audio(clip, sr)
            except Exception as e:  # noqa: BLE001 - one bad upload mustn't stop the worker
                logger.warning("embedding reference %d failed: %s", reference.id, e)
                reference.status = "failed"
                reference.error = f"{type(e).__name__}: {e}"[:2000]
            else:
                reference.embedding = vec
                reference.embedding_model = embedder.name
                reference.status = "ready"
            session.commit()
        return len(pending)


def similar_songs(
    session: Session, reference: ReferenceSong, k: int = 5
) -> list[tuple[int, float]]:
    """(song id, cosine distance) of the k library songs that sound most like it."""
    if reference.status != "ready":
        return []
    return store.nearest_to_vector(session, reference.embedding, reference.embedding_model, k)
