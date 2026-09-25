"""Expand configs/pool_grid.yaml into a deterministic list of prompts."""

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptSpec:
    text: str
    bpm: int
    attributes: dict[str, Any]


@dataclass(frozen=True)
class GridConfig:
    prompts: list[PromptSpec]
    seeds_per_prompt: int
    duration_s: float


def load_grid(path: Path) -> GridConfig:
    raw = yaml.safe_load(path.read_text())
    return GridConfig(
        prompts=build_prompts(raw),
        seeds_per_prompt=int(raw["seeds_per_prompt"]),
        duration_s=float(raw["duration_s"]),
    )


def build_prompts(raw: dict[str, Any]) -> list[PromptSpec]:
    genres: dict[str, list[str]] = raw["genres"]
    moods: list[str] = raw["moods"]
    energy: dict[str, dict[str, Any]] = raw["energy"]
    if not genres or not moods or not energy:
        raise ValueError("pool grid needs at least one genre, mood, and energy level")
    if int(raw["seeds_per_prompt"]) < 1:
        raise ValueError("seeds_per_prompt must be >= 1")

    rng = random.Random(raw["random_seed"])  # same config -> same prompts, always
    prompts = []
    for genre, mood, level in itertools.product(genres, moods, energy):
        instruments = genres[genre]
        if not instruments:
            raise ValueError(f"genre {genre!r} has no instruments")
        low, high = energy[level]["bpm"]
        if not low <= high:
            raise ValueError(f"energy {level!r}: bpm range [{low}, {high}] is reversed")
        instrument = rng.choice(instruments)
        bpm = rng.randint(low, high)
        prompts.append(
            PromptSpec(
                text=f"{mood} {genre}, {energy[level]['words']}, {instrument}",
                bpm=bpm,
                attributes={
                    "genre": genre,
                    "mood": mood,
                    "energy": level,
                    "instrument": instrument,
                },
            )
        )
    return prompts
