# Phase 0 benchmark results

**Date:** 2026-09-25 · **Machine:** Apple M3 (base), 24 GB unified memory, macOS 15.7, PyTorch 2.14 on MPS
**Reproduce:** `uv run python scripts/phase0_benchmark.py` (raw output: `data/results/phase0_benchmark.json`)

All numbers below were measured on this machine. Seeds are fixed, and a second full run gave identical retrieval ranks.

## 1. Generation speed: ACE-Step 1.5 turbo (2B), 8 steps, bf16

| Clip length | Time to generate | Faster than real time by |
|---|---|---|
| 10 s | 3.5 s | 2.9× |
| 30 s | **8.9 s** (median of 13; max 9.6 s) | 3.4× |
| 60 s | 18.6 s | 3.2× |

- Model load: about 8 s (once per process).
- Peak GPU memory: **13.6 GB** (driver-allocated, including the 60 s clip). Fits comfortably in 24 GB.
- **Implication:** a pool of 400 clips of 30 s takes about **1 hour** locally, and 8 live candidates take about **72 s**. We don't need a cloud GPU for the pool. (Before measuring, I had estimated 30–50 s per clip; it's about 4× faster.)

## 2. Embedding speed (per 30 s clip, MPS)

| Model | Dim | Time |
|---|---|---|
| CLAP `laion/larger_clap_music_and_speech` | 512 | 0.11 s |
| MuQ-MuLan-large | 512 | 0.79 s |

Both are negligible next to generation time.

## 3. Does each embedding "hear" the prompt? (text → audio retrieval)

For each of 10 genre-distinct prompts we generated one clip. Then, for each prompt's text embedding, we ranked all 10 clips by cosine similarity. A random ranking gets top-1 right 10% of the time.

| Model | Top-1 | Top-3 | Mean rank (1 = best) |
|---|---|---|---|
| CLAP (music+speech) | **60%** | 80% | 2.3 |
| MuQ-MuLan | 50% | 80% | 2.3 |

- Both are far above chance, which also means ACE-Step follows prompts reasonably well.
- **With only 10 prompts these numbers are noisy:** one prompt more or less changes top-1 by 10 points. Don't read CLAP vs. MuQ as a real difference yet. Phase 7 compares them properly, on how well each *predicts your preferences*, which is the question that actually matters.

## 4. Do different seeds of one prompt give different songs?

Average cosine similarity between embeddings:

| Model | Same prompt, different seed | Different prompts |
|---|---|---|
| CLAP | 0.775 | 0.718 |
| MuQ-MuLan | 0.750 | 0.684 |

- Clips from the same prompt are only slightly more similar than clips from completely different genres. Seed alone creates real variation, which is good news for "generate N candidates and rank them".
- All similarities are high (about 0.7 even between metal and bossa nova). The embedding spaces are **anisotropic**: every vector shares a large common direction. **Phase 2 must subtract the mean embedding before using these as features**, otherwise that shared direction swamps the differences we care about.
- Caveat: this is 2 prompts × 3 seeds, so treat it as directional.

## 5. Problems found and fixed (these matter more than the numbers)

1. **`laion/larger_clap_music` is broken.** Its text tower maps every text to nearly the same vector (text-to-text cosine 0.999 for "heavy metal" vs. "a dog barking"). This happens with both transformers 4.57 and 5.17, so the problem is in the checkpoint. It's also the checkpoint most tutorials recommend. **Fix:** use the sibling `laion/larger_clap_music_and_speech` (also Apache-2.0), which works. `test_clap_text_embeddings_are_not_collapsed` guards against regressions and fails on the broken checkpoint.
2. **CLAP's default preprocessing randomly crops audio** (`truncation="rand_trunc"`), so the same song gets a different embedding on each run. **Fix:** embed fixed, non-overlapping 10 s windows and average them. This is also what MuQ-MuLan does internally, so the two models are comparable.
3. **MuQ-MuLan's code doesn't run on transformers 5**, and we can't downgrade: diffusers' ACE-Step pipeline needs `huggingface-hub>=1`, which transformers 4.x forbids. **Fix:** a small, contained compatibility patch (`_patch_for_transformers_v5` in `embeddings/muq_mulan.py`). A test proves it reproduces the original model's outputs, recorded under transformers 4.57 with the unpatched code (`scripts/record_muq_reference.py`).
4. **No diffusers-format version of the 2B turbo model is published** (only the 4B XL). **Fix:** download the original weights and convert them locally with diffusers' own conversion script (`scripts/download_acestep.sh`).

## Decision

The stack is confirmed: **ACE-Step 1.5 turbo 2B + CLAP (music+speech) as the default embedding + MuQ-MuLan for comparison.** Everything runs locally.
