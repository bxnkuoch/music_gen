# Roadmap and progress

This file is the single source of truth for **what's done and what's next**. Update the checkboxes as you go.

Status key: ✅ done · 🔄 in progress · ⬜ not started

| Phase | Goal | Status |
|---|---|---|
| 0 | Research + benchmark the model stack | ✅ |
| 1 | Generation pipeline + song pool in a database | ✅ |
| 2 | Song representations (embeddings + interpretable features) | ⬜ **next** |
| 3 | Rating UI + Bayesian Bradley-Terry preference model + simulated users | ⬜ |
| 4 | Live generation + personalized candidate ranking | ⬜ |
| 5 | Exploration vs exploitation (Thompson sampling slates) | ⬜ |
| 6 | Reference songs + similarity search | ⬜ |
| 7 | Evaluation experiments vs baselines | ⬜ |
| 8 | Polished frontend + deployment | ⬜ |
| — | Vocals (after Phase 4) | ⬜ |

**The MVP is Phases 1–3.** They're enough to show that the system learns *your* taste, with numbers to back it up.

---

## Phase 0: research and benchmark ✅

- [x] Compare generation models (ACE-Step, Stable Audio 3/Open Small, MusicGen, HeartMuLa, YuE)
- [x] Compare embedding models (CLAP, MuQ-MuLan, MuQ, MERT, CLaMP 3, Essentia)
- [x] Project scaffold: uv, config, logging, tests, lint
- [x] ACE-Step wrapper (`src/music_gen/generation/acestep.py`), with reproducible seeds and rejection of NaN or silent output
- [x] CLAP and MuQ-MuLan wrappers (`src/music_gen/embeddings/`), deterministic
- [x] Benchmark on the M3 → [results](results/phase0_benchmark.md)

Key findings: 30 s clip in about 9 s; the broken CLAP checkpoint replaced; MuQ patched for transformers 5; embeddings need mean-centering.

## Phase 1: generation pipeline and song pool ✅

**Goal:** a reproducible pool of 30 s clips, with every clip's metadata in PostgreSQL.

- [x] `docker-compose.yml`: PostgreSQL 17 + pgvector, bound to localhost, plus a separate `music_gen_test` database for tests
- [x] Tables (`src/music_gen/db.py`) + Alembic migration: `prompts`, `generation_jobs`, `songs`
- [x] Prompt grid (`configs/pool_grid.yaml`): 10 genres × 4 moods × 3 energy levels = **120 prompts × 3 seeds = 360 clips**. The instrument and BPM for each prompt are picked with a fixed seed, so the grid is identical on every machine.
- [x] `scripts/generate_pool.py`: resumable, keeps going past individual failures, stops after 5 failures in a row, retries failed jobs up to 3 times across runs, writes files atomically
- [x] 34 new tests: grid validation, schema constraints, migrations matching the models, crash and resume, retries, no duplicates
- [x] Full pool generated → see [results/phase1_pool.md](results/phase1_pool.md)

**How the pieces fit:** a *prompt* is text plus structured attributes (genre, mood, energy, instrument). A *job* is "generate this prompt with this seed/BPM/length using this model" and tracks its status (`pending → running → done/failed`). A *song* exists only once its WAV file is safely on disk.

**Useful commands**
```bash
docker compose up -d                                  # start the database
uv run alembic upgrade head                           # create/update tables
uv run python scripts/generate_pool.py --status       # progress
uv run python scripts/generate_pool.py                # generate everything pending (Ctrl+C is safe)
docker compose exec db psql -U music_gen              # poke around in SQL
```

**Worth knowing:** the ACE-Step seed is deterministic, so retrying a job that failed because of its *content* (e.g. silent output) gives the same result. Retries only help with *transient* errors (e.g. running out of memory). If a job fails 3 times it stays `failed`. That's recorded, not hidden.

## Phase 2: song representations ⬜

- [ ] `song_embeddings` table (pgvector), keyed by (song, model name), so CLAP and MuQ can coexist
- [ ] Interpretable features: librosa tempo, loudness, brightness (spectral centroid), onset density
- [ ] Fixed **tag vocabulary** (about 40 tags: moods, genres, instruments, vocal type) + CLAP zero-shot tag scores
- [ ] Feature builder: mean-center (Phase 0 finding) → PCA to 32 dimensions → combine with tags and librosa features
- [ ] Tests: features are deterministic; PCA is fit once and versioned; nearest neighbors look sensible

## Phase 3: ratings and preference learning (core of the MVP) ⬜

- [ ] Minimal rating page: 4 clips per round, pick your favorite (and optionally your least favorite), in shuffled order, **loudness-normalized playback** (Phase 1 found about 16 dB spread; louder clips tend to win comparisons)
- [ ] Log every **slate**: what was shown, in what order, which strategy chose it
- [ ] Turn feedback into pairwise preferences ("the favorite beat each other clip")
- [ ] **Bayesian Bradley-Terry model** (`personalization/`): pure NumPy, online Laplace updates
- [ ] Simulated users whose hidden taste is defined on MuQ features (the model only sees CLAP): check that the model recovers them
- [ ] First learning curve: pairwise accuracy vs. amount of feedback

## Phases 4–8

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design. Each phase gets a detailed checklist here when it starts.

---

## Ideas parked for later (don't build yet)
- GitHub Actions CI running the fast tests on every push
- Batch generation (ACE-Step supports up to 8 clips per call) if pool generation gets slow
- Per-user LoRA fine-tuning of ACE-Step (supported from about 8 songs), a stretch goal
- Cloud GPU (Modal) for the deployed demo in Phase 8
- Store audio as FLAC instead of WAV (about half the disk space; WAV is 5.8 MB per 30 s clip)
