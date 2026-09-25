from pathlib import Path

import pytest
import torch

from music_gen.config import Settings, resolve_device


def test_defaults_point_inside_data_dir():
    s = Settings(_env_file=None)
    assert s.audio_dir == Path("data/audio")
    assert s.results_dir == Path("data/results")


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("MUSIC_GEN_DATA_DIR", "/tmp/elsewhere")
    monkeypatch.setenv("MUSIC_GEN_DEVICE", "cpu")
    s = Settings(_env_file=None)
    assert s.audio_dir == Path("/tmp/elsewhere/audio")
    assert s.device == "cpu"


def test_invalid_device_is_rejected(monkeypatch):
    monkeypatch.setenv("MUSIC_GEN_DEVICE", "tpu")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_default_clap_is_not_the_broken_checkpoint():
    # laion/larger_clap_music embeds every text to the same vector (Phase 0 finding).
    assert Settings(_env_file=None).clap_model_id != "laion/larger_clap_music"


def test_resolve_device_auto_returns_available_backend():
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
