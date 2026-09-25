import numpy as np
import pytest
from synthetic import chord_with_noise

from music_gen import audio


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.zeros(100)]).astype(np.float32)
    np.testing.assert_allclose(audio.to_mono(stereo), 0.5)


def test_to_mono_passes_mono_through():
    mono = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(audio.to_mono(mono), mono)


def test_resample_changes_length_proportionally():
    x = chord_with_noise(1.0, 48_000)[0]
    y = audio.resample(x, 48_000, 24_000)
    assert abs(len(y) - 24_000) <= 1
    assert y.dtype == np.float32


def test_resample_same_rate_is_noop():
    x = np.ones(100, dtype=np.float32)
    assert audio.resample(x, 16_000, 16_000) is x


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros(0, dtype=np.float32), "too short"),
        (np.zeros(10, dtype=np.float32), "too short"),  # 10 samples at 48 kHz
        (np.full(48_000, np.nan, dtype=np.float32), "NaN"),
        (np.full(48_000, np.inf, dtype=np.float32), "NaN or inf"),
        (np.zeros((2, 2, 48_000), dtype=np.float32), "1-D or 2-D"),
    ],
)
def test_validate_rejects_bad_audio(bad, match):
    with pytest.raises(audio.AudioError, match=match):
        audio.validate(bad, 48_000)


def test_validate_rejects_bad_sample_rate():
    with pytest.raises(audio.AudioError, match="sample_rate"):
        audio.validate(np.ones(100, dtype=np.float32), 0)


def test_rms_dbfs():
    assert audio.rms_dbfs(np.zeros(100)) == float("-inf")
    assert audio.rms_dbfs(np.ones(100)) == pytest.approx(0.0)
    assert audio.rms_dbfs(np.full(100, 0.1)) == pytest.approx(-20.0)


def test_wav_roundtrip_preserves_shape_and_rate(tmp_path):
    x = chord_with_noise(1.5, 48_000)
    path = audio.save_wav(x, 48_000, tmp_path / "nested" / "clip.wav")
    y, sr = audio.load_audio(path)
    assert sr == 48_000
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-3)  # 16-bit quantization


def test_save_wav_refuses_corrupt_audio(tmp_path):
    with pytest.raises(audio.AudioError):
        audio.save_wav(np.full(48_000, np.nan, dtype=np.float32), 48_000, tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()
