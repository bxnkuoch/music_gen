"""Embedding-wrapper tests with fake models: fast, no downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from synthetic import chord_with_noise

from music_gen.audio import AudioError
from music_gen.embeddings import ClapEmbedder, l2_normalize, split_windows

SR = 48_000
WINDOW = 10 * SR

# --- helpers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected_lengths",
    [
        (3.0, [3 * SR]),  # shorter than one window: keep it (it's the whole clip)
        (10.0, [WINDOW]),
        (24.0, [WINDOW, WINDOW]),  # 4 s remainder < half window: dropped
        (25.0, [WINDOW, WINDOW, 5 * SR]),  # 5 s remainder == half window: kept
        (30.0, [WINDOW] * 3),
    ],
)
def test_split_windows(seconds, expected_lengths):
    windows = split_windows(np.zeros(int(seconds * SR), dtype=np.float32), WINDOW)
    assert [len(w) for w in windows] == expected_lengths


def test_l2_normalize_rows_have_unit_length_and_zero_is_safe():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    y = l2_normalize(x)
    np.testing.assert_allclose(y[0], [0.6, 0.8])
    np.testing.assert_array_equal(y[1], [0.0, 0.0])  # no division-by-zero NaNs


# --- CLAP wrapper with fakes ------------------------------------------------------------


class FakeClapProcessor:
    def __init__(self):
        self.window_counts: list[int] = []

    def __call__(self, audio=None, text=None, sampling_rate=None, **_):
        if audio is not None:
            assert sampling_rate == SR
            self.window_counts.append(len(audio))
            # One 2-number "feature" per window: its mean and std.
            feats = torch.tensor([[w.mean(), w.std()] for w in audio], dtype=torch.float32)
            return {"input_features": feats}
        ids = torch.tensor([[len(t), t.count(" ")] for t in text], dtype=torch.float32)
        return {"input_ids": ids}


class FakeClapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=3)
        self.proj = torch.nn.Linear(2, 3)
        torch.nn.init.normal_(self.proj.weight, generator=torch.Generator().manual_seed(0))

    def get_audio_features(self, input_features):
        return SimpleNamespace(pooler_output=self.proj(input_features))  # transformers>=5 style

    def get_text_features(self, input_ids):
        return self.proj(input_ids)  # transformers<5 style: plain tensor


@pytest.fixture
def clap() -> ClapEmbedder:
    return ClapEmbedder("fake", "cpu", model=FakeClapModel(), processor=FakeClapProcessor())


def test_clap_audio_embedding_is_unit_vector_of_right_dim(clap):
    v = clap.embed_audio(chord_with_noise(12.0, SR), SR)
    assert v.shape == (3,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_clap_audio_embedding_is_deterministic(clap):
    x = chord_with_noise(25.0, SR)
    np.testing.assert_array_equal(clap.embed_audio(x, SR), clap.embed_audio(x, SR))


def test_clap_uses_one_window_per_10_seconds(clap):
    clap.embed_audio(chord_with_noise(30.0, SR), SR)
    assert clap.processor.window_counts == [3]


def test_clap_resamples_other_rates(clap):
    v = clap.embed_audio(chord_with_noise(5.0, 44_100), 44_100)
    assert v.shape == (3,)


def test_clap_rejects_corrupt_audio(clap):
    bad = chord_with_noise(5.0, SR)
    bad[0, 100] = np.nan
    with pytest.raises(AudioError):
        clap.embed_audio(bad, SR)


def test_clap_text_embeddings_handle_both_output_styles(clap):
    t = clap.embed_text(["a", "two words"])
    assert t.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)


def test_clap_empty_text_list_raises(clap):
    with pytest.raises(ValueError):
        clap.embed_text([])
