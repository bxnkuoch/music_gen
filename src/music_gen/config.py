"""Project settings, loaded from environment variables and an optional `.env` file.

Every setting can be overridden with an env var prefixed `MUSIC_GEN_`,
e.g. `MUSIC_GEN_DEVICE=cpu`. See `.env.example`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "mps", "cuda"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MUSIC_GEN_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    device: Device = "auto"
    log_level: str = "INFO"

    # Local development database from docker-compose.yml (bound to localhost only).
    database_url: str = "postgresql+psycopg://music_gen:music_gen@localhost:5432/music_gen"
    test_database_url: str = (
        "postgresql+psycopg://music_gen:music_gen@localhost:5432/music_gen_test"
    )

    # Generation: ACE-Step 1.5 turbo (2B), converted locally to diffusers format.
    # Produced by scripts/download_acestep.sh.
    acestep_model_path: Path = Path("data/models/acestep-v15-turbo-diffusers")

    # Embeddings. CLAP is Apache-2.0; MuQ-MuLan weights are CC BY-NC 4.0 (research only).
    # Not "laion/larger_clap_music": its text tower is broken (every text embeds to the
    # same vector; see docs/results/phase0_benchmark.md). The music+speech sibling works.
    clap_model_id: str = "laion/larger_clap_music_and_speech"
    muq_model_id: str = "OpenMuQ/MuQ-MuLan-large"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(device: Device) -> str:
    """Turn "auto" into the best available backend; validate explicit choices."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MUSIC_GEN_DEVICE=cuda but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MUSIC_GEN_DEVICE=mps but Apple MPS is not available")
    return device
