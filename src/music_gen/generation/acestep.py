"""Thin wrapper around the ACE-Step 1.5 diffusers pipeline.

Its job is to (1) validate requests, (2) make generation reproducible via seeds,
and (3) reject broken outputs before they enter the song pool. It deliberately
exposes only the knobs the personalization system will steer.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from music_gen import audio

logger = logging.getLogger(__name__)

MODEL_NAME = "ace-step-1.5-turbo-2b"
INSTRUMENTAL_LYRICS = "[Instrumental]"  # ACE-Step's convention for "no vocals"
MIN_DURATION_S = 10.0  # ACE-Step's supported range is 10-600 s
MAX_DURATION_S = 600.0
SILENCE_THRESHOLD_DBFS = -60.0


class GenerationError(RuntimeError):
    """The model ran but produced unusable audio."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    duration_s: float = 30.0
    seed: int = 0
    bpm: int | None = None  # None lets the model choose
    keyscale: str | None = None  # e.g. "A minor"
    lyrics: str = ""  # empty -> instrumental
    # A song whose sound (timbre, production) to imitate. ACE-Step uses 3 x 10 s of it.
    reference_audio: Path | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not MIN_DURATION_S <= self.duration_s <= MAX_DURATION_S:
            raise ValueError(
                f"duration_s must be in [{MIN_DURATION_S}, {MAX_DURATION_S}], got {self.duration_s}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.bpm is not None and not 30 <= self.bpm <= 300:
            raise ValueError(f"bpm must be in [30, 300], got {self.bpm}")

    @property
    def is_instrumental(self) -> bool:
        return not self.lyrics.strip()


@dataclass
class GeneratedClip:
    request: GenerationRequest
    audio: np.ndarray  # (channels, samples), float32
    sample_rate: int
    elapsed_s: float
    model_name: str = MODEL_NAME

    @property
    def duration_s(self) -> float:
        return self.audio.shape[-1] / self.sample_rate


class AceStepGenerator:
    """Loads the pipeline lazily on first use so constructing it is cheap."""

    def __init__(self, model_path: Path, device: str, pipeline: Any | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline = pipeline  # injectable for tests

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ACE-Step weights not found at {self.model_path}. "
                    "Run scripts/download_acestep.sh first."
                )
            from diffusers import AceStepPipeline  # heavy import, keep it lazy

            logger.info("Loading ACE-Step from %s on %s", self.model_path, self.device)
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self._pipeline = AceStepPipeline.from_pretrained(self.model_path, dtype=dtype).to(
                self.device
            )
        return self._pipeline

    def generate(self, request: GenerationRequest) -> GeneratedClip:
        pipe = self.pipeline
        reference = (
            None
            if request.reference_audio is None
            else _load_reference(request.reference_audio, pipe.sample_rate)
        )
        start = time.perf_counter()
        output = pipe(
            prompt=request.prompt,
            lyrics=INSTRUMENTAL_LYRICS if request.is_instrumental else request.lyrics,
            audio_duration=request.duration_s,
            bpm=request.bpm,
            keyscale=request.keyscale,
            reference_audio=reference,
            # A CPU generator makes the same seed give the same noise on any device.
            generator=torch.Generator("cpu").manual_seed(request.seed),
        )
        if self.device == "mps":
            torch.mps.synchronize()  # otherwise timing stops before the GPU finishes
        elapsed = time.perf_counter() - start

        clip_audio = output.audios[0].float().cpu().numpy()
        self._check_output(clip_audio, pipe.sample_rate)
        logger.info(
            "Generated %.1fs clip in %.1fs (seed=%d): %r",
            clip_audio.shape[-1] / pipe.sample_rate,
            elapsed,
            request.seed,
            request.prompt,
        )
        return GeneratedClip(request, clip_audio, pipe.sample_rate, elapsed)

    @staticmethod
    def _check_output(clip_audio: np.ndarray, sample_rate: int) -> None:
        try:
            audio.validate(clip_audio, sample_rate)
        except audio.AudioError as e:
            raise GenerationError(f"model produced invalid audio: {e}") from e
        loudness = audio.rms_dbfs(clip_audio)
        if loudness < SILENCE_THRESHOLD_DBFS:
            raise GenerationError(f"model produced near-silent audio ({loudness:.1f} dBFS)")


def _load_reference(path: Path, sample_rate: int) -> torch.Tensor:
    """The pipeline wants a (2, samples) tensor at its own sample rate."""
    clip, sr = audio.load_audio(path)
    audio.validate(clip, sr)
    stereo = clip if clip.ndim == 2 else np.stack([clip, clip])
    if stereo.shape[0] != 2:
        stereo = np.stack([audio.to_mono(stereo)] * 2)
    return torch.from_numpy(audio.resample(stereo, sr, sample_rate))
