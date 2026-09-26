"""The tag vocabulary (configs/tags.yaml) and zero-shot tag scoring."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Tag:
    name: str
    category: str  # genre | mood | instrument | character
    text: str  # what the text encoder sees


@dataclass(frozen=True)
class TagVocabulary:
    version: int
    tags: list[Tag]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tags]

    @property
    def texts(self) -> list[str]:
        return [t.text for t in self.tags]

    def indices(self, category: str) -> list[int]:
        return [i for i, t in enumerate(self.tags) if t.category == category]

    def feature_set(self, embedder_name: str) -> str:
        """Name under which scores are stored, e.g. "tags_v1:clap:laion/...".

        Includes the model because scores from different models aren't comparable.
        """
        return f"tags_v{self.version}:{embedder_name}"


def load_tags(path: Path) -> TagVocabulary:
    raw = yaml.safe_load(path.read_text())
    tags = [
        Tag(name=str(name), category=category, text=str(text))
        for category, entries in raw["categories"].items()
        for name, text in (entries or {}).items()
    ]
    if not tags:
        raise ValueError("tag vocabulary is empty")
    names = [t.name for t in tags]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate tag names: {sorted(duplicates)}")
    if any(not t.text.strip() for t in tags):
        raise ValueError("every tag needs non-empty text")
    return TagVocabulary(version=int(raw["version"]), tags=tags)


def score_tags(audio_vecs: np.ndarray, tag_text_vecs: np.ndarray) -> np.ndarray:
    """(songs, dim) x (tags, dim) -> (songs, tags) cosine similarities.

    Raw scores sit in a narrow band (embeddings are anisotropic), so compare them
    within a song (which tag fits best?) or standardize them across songs.
    """
    return audio_vecs @ tag_text_vecs.T
