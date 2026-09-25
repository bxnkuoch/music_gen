"""Phase 0 benchmark: can this machine run the chosen stack, and do the models behave?

Measures:
  1. ACE-Step generation speed + memory at several clip lengths.
  2. Embedding speed for CLAP and MuQ-MuLan.
  3. Prompt->clip retrieval accuracy per embedding model (does the embedding "hear"
     what the prompt asked for? chance = 1/N).
  4. Seed diversity: how similar are clips from the SAME prompt with different seeds,
     compared with clips from different prompts? This decides whether "N candidates
     from one prompt" gives the ranker enough variety to choose from.

Usage:  uv run python scripts/phase0_benchmark.py
Writes: data/results/phase0_benchmark.json (+ WAVs in data/audio/phase0/)
"""

import json
import logging
import platform
import time
from itertools import combinations

import numpy as np
import torch

from music_gen import audio
from music_gen.config import get_settings, resolve_device
from music_gen.embeddings import ClapEmbedder
from music_gen.embeddings.muq_mulan import MuQMuLanEmbedder
from music_gen.generation import AceStepGenerator, GenerationRequest
from music_gen.logging_setup import configure_logging

logger = logging.getLogger("phase0")

PROMPTS = [
    "chill late-night R&B, warm rhodes, soft drums",
    "aggressive heavy metal, distorted guitars, double kick drums",
    "gentle solo piano, classical, slow and melancholic",
    "upbeat 80s synthwave, driving bassline, retro synths",
    "lo-fi hip hop beat, dusty vinyl, jazzy chords",
    "energetic EDM festival anthem, big synth drops",
    "acoustic folk, fingerpicked guitar, warm and intimate",
    "dark cinematic orchestral score, tense strings, brass hits",
    "reggae groove, offbeat guitar skank, deep bass",
    "bossa nova, nylon guitar, soft percussion",
]
SEED_VARIANT_PROMPTS = [0, 4]  # extra seeds for these prompt indices
EXTRA_SEEDS = [1, 2]
CLIP_S = 30.0
SCALING_DURATIONS = [10.0, 60.0]


def peak_memory_gb(device: str) -> float | None:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def retrieval_stats(text_vecs: np.ndarray, audio_vecs: np.ndarray) -> dict:
    """Row i of each matrix belongs to prompt i. Rank of the true clip per prompt."""
    sims = text_vecs @ audio_vecs.T
    ranks = [int((sims[i] > sims[i, i]).sum()) + 1 for i in range(len(sims))]
    return {
        "top1_accuracy": float(np.mean([r == 1 for r in ranks])),
        "top3_accuracy": float(np.mean([r <= 3 for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "chance_top1": 1 / len(sims),
        "ranks": ranks,
    }


def seed_diversity(vecs: dict[tuple[int, int], np.ndarray]) -> dict:
    """Same prompt, different seeds vs. different prompts (seed 0 of all 10 prompts)."""
    same = [float(va @ vb) for (a, va), (b, vb) in combinations(vecs.items(), 2) if a[0] == b[0]]
    seed0 = [v for (_, s), v in vecs.items() if s == 0]
    different = [float(a @ b) for a, b in combinations(seed0, 2)]
    return {
        "same_prompt_mean_cos": float(np.mean(same)),
        "different_prompt_mean_cos": float(np.mean(different)),
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    device = resolve_device(settings.device)
    out_dir = settings.audio_dir / "phase0"
    results: dict = {
        "machine": {"platform": platform.platform(), "device": device},
        "versions": {"torch": torch.__version__},
    }

    # 1. Generation --------------------------------------------------------------------
    gen = AceStepGenerator(settings.acestep_model_path, device)
    t0 = time.perf_counter()
    _ = gen.pipeline  # load once, time separately
    results["generation"] = {"model": "ace-step-1.5-turbo-2b", "load_s": time.perf_counter() - t0}

    jobs = [(i, 0) for i in range(len(PROMPTS))]
    jobs += [(i, s) for i in SEED_VARIANT_PROMPTS for s in EXTRA_SEEDS]
    clips = {}
    for i, seed in jobs:
        clip = gen.generate(GenerationRequest(PROMPTS[i], duration_s=CLIP_S, seed=seed))
        audio.save_wav(clip.audio, clip.sample_rate, out_dir / f"p{i:02d}_s{seed}.wav")
        clips[(i, seed)] = clip

    # The first call includes one-off warm-up cost; report it separately.
    times = [c.elapsed_s for c in clips.values()]
    results["generation"].update(
        {
            "clip_s": CLIP_S,
            "n_clips": len(times),
            "first_call_s": times[0],
            "median_s": float(np.median(times[1:])),
            "max_s": float(np.max(times[1:])),
            "realtime_factor": CLIP_S / float(np.median(times[1:])),
            "peak_memory_gb": peak_memory_gb(device),
        }
    )
    scaling = {}
    for dur in SCALING_DURATIONS:
        clip = gen.generate(GenerationRequest(PROMPTS[0], duration_s=dur, seed=0))
        scaling[str(dur)] = clip.elapsed_s
    scaling[str(CLIP_S)] = results["generation"]["median_s"]
    results["generation"]["seconds_by_duration"] = scaling
    results["generation"]["peak_memory_gb"] = peak_memory_gb(device)
    del gen
    if device == "mps":
        torch.mps.empty_cache()

    # 2-4. Embeddings -------------------------------------------------------------------
    results["embeddings"] = {}
    for embedder in (
        ClapEmbedder(settings.clap_model_id, device),
        MuQMuLanEmbedder(settings.muq_model_id, device),
    ):
        vecs, elapsed = {}, []
        for key, clip in clips.items():
            t0 = time.perf_counter()
            vecs[key] = embedder.embed_audio(clip.audio, clip.sample_rate)
            elapsed.append(time.perf_counter() - t0)
        text_vecs = embedder.embed_text(PROMPTS)
        audio_vecs = np.stack([vecs[(i, 0)] for i in range(len(PROMPTS))])
        results["embeddings"][embedder.name] = {
            "dim": int(audio_vecs.shape[1]),
            "median_embed_s_per_30s_clip": float(np.median(elapsed[1:])),
            "retrieval": retrieval_stats(text_vecs, audio_vecs),
            "seed_diversity": seed_diversity(vecs),
        }
        logger.info("%s: %s", embedder.name, results["embeddings"][embedder.name]["retrieval"])

    out = settings.results_dir / "phase0_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
