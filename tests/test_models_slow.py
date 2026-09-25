"""Integration tests against the real pretrained models.

Skipped by default (multi-GB downloads, slow). Run with: `uv run pytest -m slow`
"""

from pathlib import Path

import numpy as np
import pytest
from synthetic import chord_with_noise

from music_gen.config import Settings, resolve_device

pytestmark = pytest.mark.slow

SETTINGS = Settings(_env_file=None)
FIXTURES = Path(__file__).parent / "fixtures"
DISSIMILAR_TEXTS = [
    "heavy metal with distorted guitars",
    "soft classical piano",
    "a dog barking",
]


@pytest.fixture(scope="module")
def clap():
    from music_gen.embeddings import ClapEmbedder

    return ClapEmbedder(SETTINGS.clap_model_id, "cpu")


@pytest.fixture(scope="module")
def muq():
    pytest.importorskip("muq")
    from music_gen.embeddings.muq_mulan import MuQMuLanEmbedder

    return MuQMuLanEmbedder(SETTINGS.muq_model_id, "cpu")


def test_clap_text_embeddings_are_not_collapsed(clap):
    # Regression test for the broken laion/larger_clap_music checkpoint, whose text
    # tower maps every text to (nearly) the same vector.
    t = clap.embed_text(DISSIMILAR_TEXTS)
    off_diagonal = (t @ t.T)[~np.eye(len(t), dtype=bool)]
    assert off_diagonal.max() < 0.9


def test_clap_audio_embedding_is_deterministic(clap):
    x = chord_with_noise(25.0, 48_000)
    a, b = clap.embed_audio(x, 48_000), clap.embed_audio(x, 48_000)
    assert a.shape == (512,)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_muq_matches_transformers_v4_reference(muq):
    # Proves our transformers-5 compatibility patch reproduces the original model.
    ref = np.load(FIXTURES / "muq_mulan_reference.npz")
    audio_vec = muq.embed_audio(chord_with_noise(12.0, 48_000), 48_000)
    text_vec = muq.embed_text(["a calm piano chord"])[0]
    np.testing.assert_allclose(audio_vec, ref["audio"] / np.linalg.norm(ref["audio"]), atol=1e-3)
    np.testing.assert_allclose(text_vec, ref["text"] / np.linalg.norm(ref["text"]), atol=1e-3)


@pytest.mark.skipif(
    not SETTINGS.acestep_model_path.exists(), reason="run scripts/download_acestep.sh first"
)
def test_acestep_generates_reproducible_audio():
    from music_gen.generation import AceStepGenerator, GenerationRequest

    gen = AceStepGenerator(SETTINGS.acestep_model_path, resolve_device("auto"))
    request = GenerationRequest("upbeat funk with slap bass", duration_s=10, seed=123)
    a, b = gen.generate(request), gen.generate(request)
    assert a.audio.shape[0] == 2
    assert a.duration_s == pytest.approx(10.0, abs=0.1)
    np.testing.assert_allclose(a.audio, b.audio, atol=1e-2)
