"""MuQ-MuLan: second embedding, used for comparison experiments and simulated users.

License: weights are CC BY-NC 4.0 (non-commercial). Install with `uv sync --extra muq`.
"""

import logging
from typing import Any

import numpy as np
import torch

from music_gen import audio
from music_gen.embeddings.base import l2_normalize

logger = logging.getLogger(__name__)


class MuQMuLanEmbedder:
    sample_rate = 24_000

    def __init__(self, model_id: str, device: str, model: Any | None = None) -> None:
        self.name = f"muq-mulan:{model_id}"
        self.device = device
        if model is None:
            from muq import MuQMuLan

            logger.info("Loading %s on %s", model_id, device)
            model = MuQMuLan.from_pretrained(model_id)
            _patch_for_transformers_v5(model)
        self.model = model.to(device).eval()
        self.dim: int | None = None  # known after the first call

    @torch.inference_mode()
    def embed_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = audio.prepare_mono(audio_array, sample_rate, self.sample_rate)
        # MuQ-MuLan itself splits into deterministic 10 s windows and averages them.
        wav = torch.from_numpy(mono).unsqueeze(0).to(self.device)
        vec = l2_normalize(self.model(wavs=wav)[0].float().cpu().numpy())
        self.dim = vec.shape[-1]
        return vec

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        vecs = l2_normalize(self.model(texts=texts).float().cpu().numpy())
        self.dim = vecs.shape[-1]
        return vecs


def _patch_for_transformers_v5(model: torch.nn.Module) -> None:
    """Make the `muq` package (written for transformers 4.x) run on transformers 5.

    We can't downgrade transformers: diffusers' ACE-Step pipeline needs
    huggingface-hub>=1, which transformers<5 forbids. Two things changed in v5:
      1. The conformer config must have `_attn_implementation` (muq uses a plain
         EasyDict), so we set the classic "eager" attention.
      2. `Wav2Vec2ConformerEncoder` no longer returns per-layer `hidden_states`, and
         MuQ-MuLan reads an intermediate layer. We rebuild the v4 tuple with hooks.
    tests/test_muq_mulan.py checks outputs against values recorded on transformers 4.57.
    """
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerEncoder,
    )

    for module in model.modules():
        config = getattr(module, "config", None)
        if isinstance(config, dict) and "_attn_implementation" not in config:
            config["_attn_implementation"] = "eager"
        if isinstance(module, Wav2Vec2ConformerEncoder):
            _restore_hidden_states(module)


def _restore_hidden_states(encoder: torch.nn.Module) -> None:
    # transformers 4.x returned (input to each layer..., final layer-normed output).
    original_forward = encoder.forward

    def forward(hidden_states, attention_mask=None, output_hidden_states=False, **kwargs):
        captured: list[torch.Tensor] = []
        hooks = []
        if output_hidden_states:
            hooks = [
                layer.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
                for layer in encoder.layers
            ]
        try:
            out = original_forward(hidden_states, attention_mask=attention_mask, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if output_hidden_states:
            out["hidden_states"] = (*captured, out.last_hidden_state)
        return out

    encoder.forward = forward
