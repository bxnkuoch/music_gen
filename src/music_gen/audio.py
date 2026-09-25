"""Small, model-agnostic audio helpers.

Convention used everywhere in this project: audio arrays are float32 with shape
(channels, samples) for multi-channel audio, or (samples,) for mono.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioError(ValueError):
    """Raised when audio is empty, corrupt (NaN/inf), or otherwise unusable."""


def validate(audio: np.ndarray, sample_rate: int, min_seconds: float = 0.1) -> None:
    if sample_rate <= 0:
        raise AudioError(f"sample_rate must be positive, got {sample_rate}")
    if audio.ndim not in (1, 2):
        raise AudioError(f"expected 1-D or 2-D audio, got shape {audio.shape}")
    n_samples = audio.shape[-1]
    if n_samples < min_seconds * sample_rate:
        raise AudioError(
            f"audio too short: {n_samples / sample_rate:.3f}s < {min_seconds}s minimum"
        )
    if not np.isfinite(audio).all():
        raise AudioError("audio contains NaN or inf values")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) -> (samples,) by averaging channels. Mono input passes through."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=0).astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def prepare_mono(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    """Validate, downmix to mono, and resample: the standard input for embedding models."""
    validate(audio, sample_rate)
    return resample(to_mono(audio), sample_rate, target_sr)


def rms_dbfs(audio: np.ndarray) -> float:
    """Loudness in dB relative to full scale. Digital silence returns -inf."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    validate(audio, sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (samples, channels).
    sf.write(path, audio.T if audio.ndim == 2 else audio, sample_rate, subtype="PCM_16")
    return path


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a file as (channels, samples) float32 (or (samples,) if mono)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.T
    return (audio[0] if audio.shape[0] == 1 else audio), sr
