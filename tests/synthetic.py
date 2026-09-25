"""Deterministic synthetic audio shared by tests and by the fixture-recording script."""

import numpy as np


def chord_with_noise(seconds: float, sample_rate: int, seed: int = 0) -> np.ndarray:
    """An A-minor triad plus a little seeded noise, stereo (2, samples), float32."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63)) / 3
    noise = np.random.default_rng(seed).normal(0, 0.05, size=(2, t.size))
    return (0.5 * tone + noise).astype(np.float32)
