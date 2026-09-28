"""Generator tests with a fake pipeline: fast, no model download needed."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from music_gen import audio
from music_gen.generation import (
    AceStepGenerator,
    GenerationError,
    GenerationRequest,
)
from music_gen.generation.acestep import INSTRUMENTAL_LYRICS


class FakePipeline:
    """Mimics AceStepPipeline: returns seeded noise shaped (batch, 2, samples)."""

    sample_rate = 48_000

    def __init__(self, output: str = "noise") -> None:
        self.output = output
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        n = int(kwargs["audio_duration"] * self.sample_rate)
        audio = 0.1 * torch.randn(1, 2, n, generator=kwargs["generator"])
        if self.output == "nan":
            audio[0, 0, 0] = float("nan")
        elif self.output == "silent":
            audio = torch.zeros(1, 2, n)
        return SimpleNamespace(audios=audio)


def make_generator(output: str = "noise") -> tuple[AceStepGenerator, FakePipeline]:
    pipe = FakePipeline(output)
    return AceStepGenerator(Path("unused"), "cpu", pipeline=pipe), pipe


# --- request validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"prompt": "   \n"},
        {"prompt": "ok", "duration_s": 9.99},
        {"prompt": "ok", "duration_s": 600.01},
        {"prompt": "ok", "seed": -1},
        {"prompt": "ok", "bpm": 29},
        {"prompt": "ok", "bpm": 301},
    ],
)
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GenerationRequest(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "ok", "duration_s": 10.0},
        {"prompt": "ok", "duration_s": 600.0},
        {"prompt": "ok", "bpm": 30},
        {"prompt": "ok", "bpm": 300},
        {"prompt": "ok", "seed": 0},
    ],
)
def test_boundary_requests_are_accepted(kwargs):
    GenerationRequest(**kwargs)


def test_instrumental_flag():
    assert GenerationRequest("ok").is_instrumental
    assert GenerationRequest("ok", lyrics="  ").is_instrumental
    assert not GenerationRequest("ok", lyrics="[verse]\nhello").is_instrumental


# --- generation -------------------------------------------------------------------


def test_generate_returns_stereo_clip_of_requested_length():
    gen, _ = make_generator()
    clip = gen.generate(GenerationRequest("lofi beat", duration_s=12.0))
    assert clip.audio.shape == (2, 12 * 48_000)
    assert clip.audio.dtype == np.float32
    assert clip.sample_rate == 48_000
    assert clip.duration_s == pytest.approx(12.0)
    assert clip.elapsed_s >= 0


def test_instrumental_request_sends_instrumental_tag():
    gen, pipe = make_generator()
    gen.generate(GenerationRequest("lofi beat", duration_s=10))
    assert pipe.calls[0]["lyrics"] == INSTRUMENTAL_LYRICS


def test_lyrics_and_musical_params_are_forwarded():
    gen, pipe = make_generator()
    gen.generate(
        GenerationRequest("pop", duration_s=10, lyrics="[verse]\nhi", bpm=92, keyscale="A minor")
    )
    call = pipe.calls[0]
    assert call["lyrics"] == "[verse]\nhi"
    assert call["bpm"] == 92
    assert call["keyscale"] == "A minor"
    assert call["prompt"] == "pop"


def test_no_reference_by_default():
    gen, pipe = make_generator()
    gen.generate(GenerationRequest("pop", duration_s=10))
    assert pipe.calls[0]["reference_audio"] is None


def test_reference_audio_is_sent_as_stereo_at_the_pipeline_rate(tmp_path):
    mono_24k = (0.1 * np.random.default_rng(0).standard_normal(24_000 * 3)).astype(np.float32)
    path = audio.save_wav(mono_24k, 24_000, tmp_path / "ref.wav")
    gen, pipe = make_generator()
    gen.generate(GenerationRequest("pop", duration_s=10, reference_audio=path))
    reference = pipe.calls[0]["reference_audio"]
    assert isinstance(reference, torch.Tensor)
    assert reference.shape == (2, 3 * 48_000)
    assert torch.equal(reference[0], reference[1])


def test_same_seed_is_reproducible_and_different_seeds_differ():
    gen, _ = make_generator()
    a = gen.generate(GenerationRequest("x", duration_s=10, seed=7)).audio
    b = gen.generate(GenerationRequest("x", duration_s=10, seed=7)).audio
    c = gen.generate(GenerationRequest("x", duration_s=10, seed=8)).audio
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_nan_output_raises_generation_error():
    gen, _ = make_generator("nan")
    with pytest.raises(GenerationError, match="invalid audio"):
        gen.generate(GenerationRequest("x", duration_s=10))


def test_silent_output_raises_generation_error():
    gen, _ = make_generator("silent")
    with pytest.raises(GenerationError, match="silent"):
        gen.generate(GenerationRequest("x", duration_s=10))


def test_missing_weights_give_actionable_error(tmp_path):
    gen = AceStepGenerator(tmp_path / "does-not-exist", "cpu")
    with pytest.raises(FileNotFoundError, match="download_acestep.sh"):
        gen.generate(GenerationRequest("x", duration_s=10))
