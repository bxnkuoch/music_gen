# Phase 1 results: the song pool

**Date:** 2026-09-25 · **Machine:** Apple M3, 24 GB · **Command:** `uv run python scripts/generate_pool.py`
**Grid:** [configs/pool_grid.yaml](../../configs/pool_grid.yaml): 10 genres × 4 moods × 3 energy levels = 120 prompts × 3 seeds

| | |
|---|---|
| Songs generated | **360 / 360** |
| Failed jobs | 0 (no retries needed) |
| Clip length | exactly 30.0 s each (48 kHz stereo WAV) |
| Wall time | 70 min |
| Generation time per clip | median 11.7 s, max 15.5 s |
| Disk | 1.9 GB (`data/audio/pool/`) |
| Re-run after completion | 0 new jobs (idempotent ✅) |

**Slower than the Phase 0 benchmark** (11.7 s vs. 8.9 s median). The likely causes are thermal throttling during a 70-minute sustained run and other apps sharing the GPU. For planning, use about 12 s per 30 s clip for long runs on this machine.

## Loudness by genre (dBFS, higher = louder)

| Genre | Clips | Mean | Quietest |
|---|---|---|---|
| synthwave | 36 | −18.3 | −25.6 |
| heavy metal | 36 | −18.7 | −27.8 |
| jazz | 36 | −18.8 | −27.4 |
| house | 36 | −19.6 | −26.5 |
| reggae | 36 | −19.6 | −25.8 |
| classical | 36 | −19.8 | −27.5 |
| acoustic folk | 36 | −19.9 | −24.8 |
| ambient | 36 | −20.3 | −26.9 |
| r&b | 36 | −20.5 | −27.1 |
| lo-fi hip hop | 36 | −21.2 | −31.1 |

All clips are well above the −60 dBFS silence threshold. Loudness varies by about 3 dB between genre averages and by about 16 dB between individual clips.

**Why this matters for Phase 3:** people tend to prefer louder audio in side-by-side comparisons. If we don't **loudness-normalize** clips for playback, the preference model could learn "likes loud" instead of your actual taste. We'll normalize playback volume in the rating UI and keep loudness as an explicit feature, so we can check whether it predicts your choices.
