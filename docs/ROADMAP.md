# Roadmap and progress

This file is the single source of truth for **what's done and what's next**. Update the checkboxes as you go.

Status key: ✅ done · 🔄 in progress · ⬜ not started

| Phase | Goal | Status |
|---|---|---|
| 0 | Research + benchmark the model stack | ✅ |
| 1 | Generation pipeline + song pool in a database | ✅ |
| 2 | Song representations (embeddings + interpretable features) | ✅ |
| 3 | Rating UI + Bayesian Bradley-Terry preference model + simulated users | ✅ (now: collect ratings) |
| 4 | Live generation + personalized candidate ranking | ✅ (now: collect A/B data) |
| 5 | Exploration vs exploitation (Thompson sampling slates) | ⬜ **next** |
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

## Phase 2: song representations ✅

- [x] Pool extended with the listener's favorite styles: lo-fi guitar, pop, hyperpop, anime soundtrack, fantasy adventure video game (**540 songs**, 15 genres). The grid is append-only, and a test guarantees existing prompts never change.
- [x] `song_embeddings` (pgvector) and `song_features` (JSONB) tables + migration
- [x] CLAP + MuQ-MuLan embeddings for every song
- [x] Signal features (`features/signal.py`): tempo, loudness, brightness, onset rate
- [x] Tag vocabulary (`configs/tags.yaml`): 45 tags in 4 categories, including "melodic" and "plucked guitar"
- [x] Feature space (`features/space.py`): mean-center → 32 PCs (94% variance) + tags + signal → standardized, 81 features, saved and versioned
- [x] Similar-song search: `scripts/similar_songs.py <song_id>`
- [x] Validation against ground truth: [results/phase2_features.md](results/phase2_features.md)
- [x] Tests: known-tempo click tracks, tag vocabulary, feature space math, storage and search

**Key findings:** genre and energy are clearly present in the features; mood (as ACE-Step renders it) is not. Raw zero-shot tags are biased and need calibrating. ACE-Step matches the requested BPM about 60% of the time. → Mood becomes a *context the listener provides*, not something detected from audio (ARCHITECTURE.md §7).

**Useful commands**
```bash
uv run python scripts/compute_features.py        # embed/featurize new songs, refit the space
uv run python scripts/phase2_validation.py       # re-measure feature quality
uv run python scripts/similar_songs.py 42        # songs similar to song 42
```

## Phase 3: ratings and preference learning ✅

- [x] Rating page (`web/`, Next.js + TypeScript): pick a mood → 4 songs → favorite (+ optional least favorite). Songs play at **equal loudness** (−26 dBFS; 99.4% of the pool within 1 dB). You must hear ≥5 s of each song before answering.
- [x] **Blind by design:** the page never learns which songs were "model" vs "random" picks; that's only logged in the database. Display order is shuffled.
- [x] API (`src/music_gen/api/`, FastAPI): sessions, slates, answers, normalized audio, "what it learned" profile, stats. Interactive docs at http://localhost:8000/docs
- [x] Tables: `rating_sessions`, `slates`, `slate_items` (+ migration)
- [x] Feedback → pairs (`personalization/pairs.py`): favorite beats the other 3; least favorite loses to the other 2 → up to 5 pairs per slate
- [x] **Bayesian Bradley-Terry** (`personalization/bradley_terry.py`): Laplace approximation, damped Newton; global and **per-mood** variants
- [x] Baselines: random, average-of-liked-songs
- [x] Slate policy: 2 Thompson-sampling picks + 2 random picks, one song per prompt, unseen songs first
- [x] Simulated listeners on the real pool → [results/phase3_simulation.md](results/phase3_simulation.md)
- [x] Real-data learning curve: `scripts/evaluate_ratings.py`
- [x] 45 new tests (165 fast total): math, models, policy, simulation, API end to end

**Key findings (simulated):** the Bayesian model beats "average of liked songs" from about 10 ratings on. Per-mood modeling shows a small, not-yet-significant benefit when taste really is mood-dependent, so the app serves the global model until your real data says otherwise.

### ▶️ What to do now: rate songs
```bash
bash scripts/start_app.sh          # then open http://localhost:3000
```
- Pick the mood you're actually in. Rate honestly; skipping the least favorite is fine.
- **Targets:** ~20 slates to see early signal, **~100+ for a solid learning curve**. Spread them over several sessions and moods (about 10 minutes = 10 slates).
- After ~20 slates, check progress: `uv run python scripts/evaluate_ratings.py`

## Phase 4: live generation + personalized ranking ✅

- [x] **"Create new" tab**: type a request + pick a mood → 8 new songs are generated → the best 4 are shown blind → your pick trains the model
- [x] Request parsing (`personalization/steering.py`, `configs/request_words.yaml`): keyword/alias matching onto the tag vocabulary, plus energy words → BPM range. No LLM.
- [x] **Steering:** 4 of the 8 candidates add 2 tags picked from one Thompson sample of your learned taste each (never overriding the genre you asked for). The other 4 are plain (request only) = the non-personalized baseline
- [x] **Ranking:** the model scores all 8; the slate shows the top 2 steered + top 2 plain, shuffled (random tie-break when the model knows nothing yet)
- [x] **Worker process** (`scripts/worker.py`, `music_gen/worker.py`): generate → featurize → rank, communicating with the API only through the database job queue. Live requests jump ahead of pool jobs.
- [x] `generation_requests` table; jobs record `request_id` + `variant` (steered/plain)
- [x] The API picks up new songs automatically (`DbSongSource` reloads the song matrix when the song count changes)
- [x] **Steering A/B metric** in `scripts/evaluate_ratings.py`: how often you pick a steered song (50% = no effect), with a Wilson 95% CI
- [x] Real end-to-end run: 8 songs generated, featurized and ranked in **81 s** → [results/phase4_live_generation.md](results/phase4_live_generation.md)
- [x] 11 new tests (198 fast total), including the full request → worker → slate flow, failure cases, and job priority

### ▶️ What to do now
```bash
bash scripts/start_app.sh          # now also starts the worker; open http://localhost:3000
```
- Keep rating in **Rate library** (the model's taste gets sharper → better steering)
- Use **Create new** whenever you want: each answered request is one data point for "does personalization help?"
- **Targets:** ~30 answered requests for a first read on the A/B, ~100 for a confident one.

## Phases 5–8

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design. Each phase gets a detailed checklist here when it starts.

---

## Ideas parked for later (don't build yet)
- GitHub Actions CI running the fast tests on every push
- Batch generation (ACE-Step supports up to 8 clips per call) if pool generation gets slow
- Per-user LoRA fine-tuning of ACE-Step (supported from about 8 songs), a stretch goal
- Cloud GPU (Modal) for the deployed demo in Phase 8
- Store audio as FLAC instead of WAV (about half the disk space; WAV is 5.8 MB per 30 s clip)
