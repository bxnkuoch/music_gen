"""Signal features, tag vocabulary, and feature space: pure functions, no DB/models."""

from pathlib import Path

import librosa
import numpy as np
import pytest
import yaml

from music_gen.features.signal import compute_signal_features
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags, score_tags

ROOT = Path(__file__).parents[1]
SR = 44_100


def click_track(bpm: float, seconds: float = 30.0) -> np.ndarray:
    times = np.arange(0, seconds, 60 / bpm)
    return librosa.clicks(times=times, sr=SR, length=int(seconds * SR)).astype(np.float32)


def sine(freq: float, seconds: float = 5.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- signal features ------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [70, 100, 128, 145])
def test_tempo_is_recovered_from_a_click_track(bpm):
    tempo = compute_signal_features(click_track(bpm), SR)["tempo_bpm"]
    assert tempo == pytest.approx(bpm, rel=0.03)


def test_onset_rate_counts_clicks_per_second():
    # 120 BPM = 2 clicks per second
    assert compute_signal_features(click_track(120), SR)["onset_rate"] == pytest.approx(2, abs=0.2)


def test_loudness_matches_sine_rms():
    # RMS of a sine with amplitude a is a/sqrt(2): 0.5 -> -9.03 dBFS
    assert compute_signal_features(sine(440), SR)["loudness_dbfs"] == pytest.approx(-9.03, abs=0.1)


def test_higher_pitch_is_brighter():
    low = compute_signal_features(sine(200), SR)["brightness_hz"]
    high = compute_signal_features(sine(3000), SR)["brightness_hz"]
    assert high > 5 * low


def test_signal_features_accept_stereo_and_return_all_keys():
    stereo = np.stack([click_track(100), click_track(100)])
    values = compute_signal_features(stereo, SR)
    assert set(values) == {"tempo_bpm", "loudness_dbfs", "brightness_hz", "onset_rate"}
    assert all(isinstance(v, float) for v in values.values())


# --- tag vocabulary ---------------------------------------------------------------------


def test_real_tag_vocabulary_loads():
    vocab = load_tags(ROOT / "configs/tags.yaml")
    assert len(vocab.tags) >= 40
    assert len(set(vocab.names)) == len(vocab.names)
    assert {t.category for t in vocab.tags} == {"genre", "mood", "instrument", "character"}
    assert "melodic" in vocab.names and "plucked guitar" in vocab.names


def test_genre_and_mood_tags_cover_the_pool_grid():
    # Needed to measure how well CLAP recognises the genres/moods we generated.
    vocab = load_tags(ROOT / "configs/tags.yaml")
    grid = yaml.safe_load((ROOT / "configs/pool_grid.yaml").read_text())
    genre_tags = {vocab.names[i] for i in vocab.indices("genre")}
    mood_tags = {vocab.names[i] for i in vocab.indices("mood")}
    assert set(grid["genres"]) <= genre_tags
    assert set(grid["moods"]) <= mood_tags


def test_feature_set_name_includes_version_and_model():
    vocab = load_tags(ROOT / "configs/tags.yaml")
    assert vocab.feature_set("clap:x") == f"tags_v{vocab.version}:clap:x"


@pytest.mark.parametrize(
    "categories, match",
    [
        ({"genre": {"pop": "pop music"}, "mood": {"pop": "poppy"}}, "duplicate"),
        ({"genre": {"pop": "  "}}, "non-empty"),
        ({"genre": {}}, "empty"),
    ],
)
def test_bad_vocabularies_are_rejected(tmp_path, categories, match):
    path = tmp_path / "tags.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "categories": categories}))
    with pytest.raises(ValueError, match=match):
        load_tags(path)


def test_score_tags_is_cosine_similarity():
    audio_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    text_vecs = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, -1.0]])
    np.testing.assert_allclose(
        score_tags(audio_vecs, text_vecs), [[1.0, 0.6, 0.0], [0.0, 0.8, -1.0]]
    )


# --- feature space ----------------------------------------------------------------------


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 16)) + 3.0  # shared offset, like real (anisotropic) embeddings
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tabular = np.column_stack([rng.normal(100, 20, 50), rng.normal(0.1, 0.01, 50)])
    return emb, tabular, ["signal:tempo_bpm", "tag:melodic"]


def fit(data, k=4):
    emb, tabular, names = data
    return FeatureSpace.fit(emb, tabular, names, n_components=k, version="t", embedding_model="m")


def test_fit_produces_standardized_named_columns(data):
    space = fit(data)
    x = space.transform(data[0], data[1])
    assert x.shape == (50, 6)
    assert space.names == ["pc1", "pc2", "pc3", "pc4", "signal:tempo_bpm", "tag:melodic"]
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(x.std(axis=0), 1, atol=1e-9)


def test_explained_variance_is_sorted_and_bounded(data):
    ratio = fit(data).explained_variance_ratio
    assert np.all(np.diff(ratio) <= 1e-12)
    assert 0 < ratio.sum() <= 1


def test_refitting_is_deterministic(data):
    np.testing.assert_array_equal(fit(data).components, fit(data).components)


def test_save_and_load_give_identical_transforms(data, tmp_path):
    space = fit(data)
    space.save(tmp_path / "space.npz")
    loaded = FeatureSpace.load(tmp_path / "space.npz")
    assert (loaded.version, loaded.embedding_model, loaded.names) == ("t", "m", space.names)
    np.testing.assert_array_equal(loaded.transform(*data[:2]), space.transform(*data[:2]))


def test_constant_column_becomes_zero_not_nan(data):
    emb, tabular, names = data
    tabular = tabular.copy()
    tabular[:, 1] = 0.5
    x = fit((emb, tabular, names)).transform(emb, tabular)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, -1], 0)


def test_new_songs_use_the_fitted_statistics(data):
    space = fit(data)
    one = space.transform(data[0][:1], data[1][:1])
    np.testing.assert_allclose(one, space.transform(*data[:2])[:1])


@pytest.mark.parametrize("k", [0, 17, 51])
def test_invalid_component_count_is_rejected(data, k):
    with pytest.raises(ValueError, match="n_components"):
        fit(data, k=k)


def test_mismatched_or_nonfinite_inputs_are_rejected(data):
    emb, tabular, names = data
    with pytest.raises(ValueError, match="tabular shape"):
        FeatureSpace.fit(emb, tabular, names[:1], n_components=2, version="t", embedding_model="m")
    bad = emb.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        FeatureSpace.fit(bad, tabular, names, n_components=2, version="t", embedding_model="m")
    space = fit(data)
    with pytest.raises(ValueError, match="embedding dimension"):
        space.transform(emb[:, :8], tabular)
    with pytest.raises(ValueError, match="tabular features"):
        space.transform(emb, tabular[:, :1])
