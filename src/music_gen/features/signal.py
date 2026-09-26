"""Hand-crafted signal features: interpretable, model-free descriptions of a clip."""

import librosa
import numpy as np

from music_gen import audio

FEATURE_SET = "signal_v1"  # bump if the computation below changes
ANALYSIS_SR = 22_050  # librosa's standard analysis rate; plenty for these features


def compute_signal_features(audio_array: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Tempo, loudness, brightness, and note density for one clip."""
    y = audio.prepare_mono(audio_array, sample_rate, ANALYSIS_SR)
    duration_s = len(y) / ANALYSIS_SR
    onsets = librosa.onset.onset_detect(y=y, sr=ANALYSIS_SR, units="time")
    return {
        # Estimated beats per minute (may be off by 2x on some music: "octave errors").
        "tempo_bpm": float(librosa.feature.tempo(y=y, sr=ANALYSIS_SR)[0]),
        # Average loudness; the rating UI normalizes playback, this keeps it measurable.
        "loudness_dbfs": audio.rms_dbfs(y),
        # Spectral centroid: higher = brighter/harsher, lower = darker/warmer.
        "brightness_hz": float(librosa.feature.spectral_centroid(y=y, sr=ANALYSIS_SR).mean()),
        # Detected note/drum onsets per second: a rough "busyness" measure.
        "onset_rate": len(onsets) / duration_s,
    }
