# Adaptive AI Music: personalized generation from pairwise feedback

A music system that **learns one listener's taste** from "A or B?" feedback and uses that learned taste to **steer generation** and **rank candidate songs**. Open-source models generate and embed the audio. The preference learning, exploration strategy and evaluation are built from scratch.

> **Status:** Phase 0 of 8 is complete (model stack chosen and benchmarked). See **[docs/ROADMAP.md](docs/ROADMAP.md)** for progress and next steps.

## Why this is more than "call a music API"

| Typical project | This project |
|---|---|
| Prompt → one song | Prompt → 8 candidates → personalized ranking → you pick → the model updates |
| Star ratings | Pairwise preferences with a Bayesian Bradley-Terry model |
| "It works" | Held-out learning curves vs. baselines, with confidence intervals |
| Recommends more of the same | Thompson-sampling exploration + diversity-aware slates |

## Stack

| Layer | Choice | Why |
|---|---|---|
| Generation | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) turbo 2B (MIT) | Best open model; about 9 s per 30 s clip on an M3 |
| Embeddings | [CLAP music+speech](https://huggingface.co/laion/larger_clap_music_and_speech) (Apache-2.0); [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) for comparison | Shared text-audio space: zero-shot tags + prompt matching |
| ML | PyTorch, NumPy, scikit-learn | |
| Backend (Phase 1+) | FastAPI, PostgreSQL + pgvector | |
| Frontend (Phase 3+) | Next.js + TypeScript | |

Model comparison: [docs/MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Benchmark numbers: [docs/results/phase0_benchmark.md](docs/results/phase0_benchmark.md).

## Getting started

Requirements: macOS on Apple Silicon (or Linux with NVIDIA), [uv](https://docs.astral.sh/uv/), about 15 GB free disk.

```bash
# 1. Install Python 3.12 + all dependencies into .venv
uv sync --extra muq

# 2. Download and convert the generation model (~6 GB, one time)
bash scripts/download_acestep.sh

# 3. Run the fast tests (no downloads, ~2 s)
uv run pytest

# 4. Run the tests against the real models (downloads CLAP + MuQ, ~1 min)
uv run pytest -m slow

# 5. Reproduce the Phase 0 benchmark (~5 min; writes WAVs to data/audio/phase0/)
uv run python scripts/phase0_benchmark.py
```

Settings are read from environment variables or a `.env` file. Copy `.env.example` to get started. Nothing secret is ever committed.

## Repository layout

```
src/music_gen/
  config.py            settings (env vars / .env), device selection
  audio.py             audio validation, mono/resample, WAV I/O
  generation/          ACE-Step wrapper: seeded, validated generation
  embeddings/          CLAP + MuQ-MuLan wrappers (deterministic, L2-normalized)
tests/                 fast unit tests (fakes) + @slow tests (real models)
scripts/               benchmark, model download, fixture recording
docs/
  ROADMAP.md           ← what's done / what's next (start here)
  ARCHITECTURE.md      design decisions and reasons
  CONCEPTS.md          plain-language glossary (embeddings, Bradley-Terry, ...)
  MODEL_RESEARCH.md    why these models (Phase 0 comparison)
  results/             measured results, one file per experiment
data/                  (git-ignored) model weights, audio, raw results
```

## Testing

- `uv run pytest`: **52 fast tests**. They use fake models to check validation and edge cases (empty prompts, out-of-range durations and BPM, NaN or silent audio, seed reproducibility, deterministic windowing, resampling, missing weights).
- `uv run pytest -m slow`: **4 integration tests** with the real models. They include a regression test for a broken CLAP checkpoint, and a check that our MuQ compatibility patch reproduces the original model's outputs.
- `uv run ruff check . && uv run ruff format --check .`: lint and formatting.

## Licenses

Code license: not chosen yet (add a `LICENSE` file before making the repo public; MIT is the common choice for portfolios). Model licenses are listed in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-licenses). MuQ-MuLan weights are **non-commercial** and are used only for research and evaluation.
