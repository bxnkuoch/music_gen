"""Read/write song embeddings and features in the database, and similarity search."""

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from music_gen.db import Song, SongEmbedding, SongFeatures


def songs_missing_embedding(session: Session, model_name: str) -> list[Song]:
    return list(
        session.scalars(
            select(Song)
            .outerjoin(
                SongEmbedding,
                and_(SongEmbedding.song_id == Song.id, SongEmbedding.model_name == model_name),
            )
            .where(SongEmbedding.song_id.is_(None))
            .order_by(Song.id)
        )
    )


def songs_missing_features(session: Session, feature_set: str) -> list[Song]:
    return list(
        session.scalars(
            select(Song)
            .outerjoin(
                SongFeatures,
                and_(SongFeatures.song_id == Song.id, SongFeatures.feature_set == feature_set),
            )
            .where(SongFeatures.song_id.is_(None))
            .order_by(Song.id)
        )
    )


def save_embedding(session: Session, song_id: int, model_name: str, vector: np.ndarray) -> None:
    if not np.isfinite(vector).all():
        raise ValueError(f"embedding for song {song_id} contains NaN or inf")
    session.execute(
        insert(SongEmbedding)
        .values(song_id=song_id, model_name=model_name, embedding=vector.astype(np.float32))
        .on_conflict_do_nothing()
    )


def save_features(
    session: Session, song_id: int, feature_set: str, values: dict[str, float]
) -> None:
    bad = [k for k, v in values.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"song {song_id}: non-finite {feature_set} features {bad}")
    session.execute(
        insert(SongFeatures)
        .values(song_id=song_id, feature_set=feature_set, values=values)
        .on_conflict_do_nothing()
    )


def load_embeddings(session: Session, model_name: str) -> tuple[list[int], np.ndarray]:
    rows = session.execute(
        select(SongEmbedding.song_id, SongEmbedding.embedding)
        .where(SongEmbedding.model_name == model_name)
        .order_by(SongEmbedding.song_id)
    ).all()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [r[0] for r in rows], np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])


def load_features(session: Session, feature_set: str) -> tuple[list[int], list[str], np.ndarray]:
    """All songs' values for one feature set as (song_ids, column names, matrix)."""
    rows = session.execute(
        select(SongFeatures.song_id, SongFeatures.values)
        .where(SongFeatures.feature_set == feature_set)
        .order_by(SongFeatures.song_id)
    ).all()
    if not rows:
        return [], [], np.zeros((0, 0))
    names = sorted(rows[0][1])  # JSONB doesn't keep key order, so sort for stability
    for song_id, values in rows:
        if sorted(values) != names:
            raise ValueError(f"song {song_id} has different {feature_set} keys than others")
    return [r[0] for r in rows], names, np.array([[r[1][n] for n in names] for r in rows])


def load_feature_table(
    session: Session, embedding_model: str, feature_sets: dict[str, str]
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    """Join an embedding with several feature sets, keeping songs that have all of them.

    `feature_sets` maps feature_set -> column prefix, e.g. {"signal_v1": "signal"}.
    Returns (song_ids, embeddings, tabular, tabular_names).
    """
    ids, emb = load_embeddings(session, embedding_model)
    by_song = {sid: [] for sid in ids}
    names: list[str] = []
    for feature_set, prefix in feature_sets.items():
        f_ids, f_names, f_values = load_features(session, feature_set)
        names += [f"{prefix}:{n}" for n in f_names]
        lookup = dict(zip(f_ids, f_values, strict=True))
        for sid in list(by_song):
            if sid in lookup:
                by_song[sid].append(lookup[sid])
            else:
                del by_song[sid]  # missing this feature set: leave the song out
    keep = [i for i, sid in enumerate(ids) if sid in by_song]
    tabular = np.array([np.concatenate(by_song[ids[i]]) for i in keep]).reshape(len(keep), -1)
    return [ids[i] for i in keep], emb[keep], tabular, names


def nearest_songs(
    session: Session, song_id: int, model_name: str, k: int = 5
) -> list[tuple[int, float]]:
    """The k most similar songs by cosine distance (0 = identical), via pgvector."""
    target = session.scalar(
        select(SongEmbedding.embedding).where(
            SongEmbedding.song_id == song_id, SongEmbedding.model_name == model_name
        )
    )
    if target is None:
        raise LookupError(f"song {song_id} has no {model_name} embedding")
    return nearest_to_vector(session, target, model_name, k, exclude_song_id=song_id)


def nearest_to_vector(
    session: Session,
    target: np.ndarray,
    model_name: str,
    k: int = 5,
    exclude_song_id: int | None = None,
) -> list[tuple[int, float]]:
    """The k songs closest to any embedding (e.g. an uploaded reference song)."""
    distance = SongEmbedding.embedding.cosine_distance(target)
    rows = session.execute(
        select(SongEmbedding.song_id, distance)
        .where(SongEmbedding.model_name == model_name, SongEmbedding.song_id != exclude_song_id)
        .order_by(distance)
        .limit(k)
    ).all()
    return [(r[0], float(r[1])) for r in rows]
