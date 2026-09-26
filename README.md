# Adaptive AI Music: personalized generation from pairwise feedback

A music system that **learns one listener's taste** from "A or B?" feedback and uses that learned taste to **steer generation** and **rank candidate songs**. Open-source models generate and embed the audio. The preference learning, exploration strategy and evaluation are built from scratch.

> **Status:** Phases 0–2 of 8 are complete: model stack benchmarked, a 540-song pool in PostgreSQL, and validated song features plus similar-song search (pgvector). See **[docs/ROADMAP.md](docs/ROADMAP.md)** for progress and next steps.

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

Requirements: macOS on Apple Silicon (or Linux with NVIDIA), [uv](https://docs.astral.sh/uv/), [Docker](https://www.docker.com/), about 15 GB free disk.

```bash
# 1. Install Python 3.12 + all dependencies into .venv
uv sync --extra muq

# 2. Download and convert the generation model (~6 GB, one time)
bash scripts/download_acestep.sh

# 3. Start the database and create the tables
docker compose up -d
uv run alembic upgrade head

# 4. Run the fast tests (~2 s; database tests need step 3)
uv run pytest

# 5. Run the tests against the real models (downloads CLAP + MuQ, ~1 min)
uv run pytest -m slow

# 6. Generate the song pool (540 clips, ~1.5 hours; safe to stop and re-run)
uv run python scripts/generate_pool.py

# 7. Compute embeddings + features, then validate them (~10 min)
uv run python scripts/compute_features.py
uv run python scripts/phase2_validation.py

# Find songs similar to song 42
uv run python scripts/similar_songs.py 42

# Optional: reproduce the Phase 0 benchmark (~5 min)
uv run python scripts/phase0_benchmark.py
```

Settings are read from environment variables or a `.env` file. Copy `.env.example` to get started. Nothing secret is ever committed.

## Repository layout

```
src/music_gen/
  config.py            settings (env vars / .env), device selection
  db.py                database tables: prompts, generation_jobs, songs
  audio.py             audio validation, mono/resample, WAV I/O
  pool/                prompt grid + resumable pool generation
  features/            signal features, tag scores, feature space, pgvector search
  generation/          ACE-Step wrapper: seeded, validated generation
  embeddings/          CLAP + MuQ-MuLan wrappers (deterministic, L2-normalized)
tests/                 fast unit tests (fakes) + @slow tests (real models)
scripts/               pool generation, benchmark, model download
configs/               pool_grid.yaml (what the pool contains), tags.yaml (tag vocabulary)
alembic/               database migrations (schema history)
docker-compose.yml     PostgreSQL + pgvector
docs/
  ROADMAP.md           ← what's done / what's next (start here)
  ARCHITECTURE.md      design decisions and reasons
  CONCEPTS.md          plain-language glossary (embeddings, Bradley-Terry, ...)
  MODEL_RESEARCH.md    why these models (Phase 0 comparison)
  results/             measured results, one file per experiment
data/                  (git-ignored) model weights, audio, raw results
```

## Testing

- `uv run pytest`: **120 fast tests**. They use fake models to check validation and edge cases (empty prompts, out-of-range durations and BPM, NaN or silent audio, seed reproducibility, deterministic windowing, missing weights). Feature tests use signals with known answers (click tracks at a known BPM, sines of known loudness). Database tests cover migrations matching the models, constraints, crash and resume, no duplicates, and pgvector search ordering, using a separate test database. They're skipped, with a message, if Postgres isn't running.
- `uv run pytest -m slow`: **4 integration tests** with the real models. They include a regression test for a broken CLAP checkpoint, and a check that our MuQ compatibility patch reproduces the original model's outputs.
- `uv run ruff check . && uv run ruff format --check .`: lint and formatting.

## Licenses

Code license: not chosen yet (add a `LICENSE` file before making the repo public; MIT is the common choice for portfolios). Model licenses are listed in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#8-licenses). MuQ-MuLan weights are **non-commercial** and are used only for research and evaluation.
