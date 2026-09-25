"""LAION CLAP (music checkpoint): the project's default embedding. Apache-2.0."""

import logging
from typing import Any

import numpy as np
import torch

from music_gen import audio
from music_gen.embeddings.base import l2_normalize, split_windows

logger = logging.getLogger(__name__)


class ClapEmbedder:
    sample_rate = 48_000
    window_s = 10  # CLAP's native input length

    def __init__(
        self,
        model_id: str,
        device: str,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.name = f"clap:{model_id}"
        self.device = device
        if model is None or processor is None:
            from transformers import ClapModel, ClapProcessor

            logger.info("Loading %s on %s", model_id, device)
            model = ClapModel.from_pretrained(model_id)
            processor = ClapProcessor.from_pretrained(model_id)
        self.model = model.to(device).eval()
        self.processor = processor
        self.dim: int = self.model.config.projection_dim

    @torch.inference_mode()
    def embed_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = audio.prepare_mono(audio_array, sample_rate, self.sample_rate)
        windows = split_windows(mono, self.window_s * self.sample_rate)
        inputs = self.processor(audio=windows, sampling_rate=self.sample_rate, return_tensors="pt")
        features = self.model.get_audio_features(
            input_features=inputs["input_features"].to(self.device)
        )
        per_window = _as_tensor(features).float().cpu().numpy()
        return l2_normalize(per_window.mean(axis=0))

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return l2_normalize(_as_tensor(features).float().cpu().numpy())


def _as_tensor(features: Any) -> torch.Tensor:
    # transformers>=5 returns a model output object; older versions returned a tensor.
    return features if isinstance(features, torch.Tensor) else features.pooler_output
