from typing import Protocol

import numpy as np


class AudioTextEmbedder(Protocol):
    """A model that maps audio and text into one shared, L2-normalized vector space.

    Because vectors are unit length, `a @ b` is their cosine similarity.
    """

    name: str  # stored next to every embedding so different models never get mixed
    dim: int

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """One clip -> (dim,) vector."""
        ...

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """N texts -> (N, dim) matrix."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def split_windows(mono: np.ndarray, window: int) -> list[np.ndarray]:
    """Cut audio into non-overlapping windows of `window` samples.

    Deterministic by design (unlike CLAP's default random crop), so the same song
    always gets the same embedding. A trailing remainder shorter than half a window
    is dropped unless it is the whole clip.
    """
    windows = [mono[i : i + window] for i in range(0, len(mono), window)]
    if len(windows) > 1 and len(windows[-1]) < window // 2:
        windows.pop()
    return windows
