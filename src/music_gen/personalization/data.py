"""Load everything the personalization code needs from the database, as arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from music_gen.db import GenerationJob, Prompt, Song
from music_gen.features import signal, store
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import TagVocabulary


def load_contexts(path: Path) -> dict[str, str]:
    contexts = yaml.safe_load(path.read_text())["contexts"]
    if not contexts:
        raise ValueError("at least one context (mood) is required")
    return {str(k): str(v) for k, v in contexts.items()}


@dataclass(frozen=True)
class SongMatrix:
    """Songs as rows. `x` is the model's input; the rest is metadata."""

    song_ids: np.ndarray  # database ids, row-aligned with x
    x: np.ndarray  # (n, d) feature-space vectors
    names: list[str]  # feature names, (d,)
    groups: np.ndarray  # prompt id per song (to avoid near-duplicates in a slate)
    genres: np.ndarray
    energies: np.ndarray

    def row_of(self, song_id: int) -> int:
        rows = np.flatnonzero(self.song_ids == song_id)
        if not len(rows):
            raise KeyError(f"song {song_id} has no features")
        return int(rows[0])


def load_song_matrix(session: Session, space: FeatureSpace, vocab: TagVocabulary) -> SongMatrix:
    ids, emb, tabular, names = store.load_feature_table(
        session,
        space.embedding_model,
        {vocab.feature_set(space.embedding_model): "tag", signal.FEATURE_SET: "signal"},
    )
    if names != space.names[len(space.components) :]:
        raise ValueError("stored features don't match the feature space; rerun compute_features")
    meta = dict(
        (sid, (pid, attrs))
        for sid, pid, attrs in session.execute(
            select(Song.id, Prompt.id, Prompt.attributes)
            .join(GenerationJob, Song.job_id == GenerationJob.id)
            .join(Prompt, GenerationJob.prompt_id == Prompt.id)
        )
    )
    return SongMatrix(
        song_ids=np.array(ids),
        x=space.transform(emb, tabular),
        names=space.names,
        groups=np.array([meta[s][0] for s in ids]),
        genres=np.array([meta[s][1].get("genre", "?") for s in ids]),
        energies=np.array([meta[s][1].get("energy", "?") for s in ids]),
    )
