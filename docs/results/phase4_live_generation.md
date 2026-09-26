# Phase 4 results: live generation

**Date:** 2026-09-26 · **Machine:** Apple M3, 24 GB

## End-to-end run (real models)

Request: *"chill lofi with guitar plucking for studying"*, mood *chill*, with 5 library ratings so far.

- **Parsed as:** chill · lo-fi hip hop · plucked guitar · low energy → BPM 68 for all candidates
- **81 s** from request to ranked slate: 8 × ~9 s generation + featurization + ranking; 0 failures

| Variant | Prompt | Model score | Shown |
|---|---|---|---|
| plain | chill lofi with guitar plucking for studying | 0.822 | ✅ |
| plain | (same, other seed) | 0.747 | ✅ |
| plain | (same, 2 more seeds) | lower | – |
| steered | …, harp, dark | 0.649 | ✅ |
| steered | …, strings, harp | 0.363 | ✅ |
| steered | …, distorted guitar, glitchy | lower | – |
| steered | …, uplifting, catchy | lower | – |

**Reading it:** with only 5 ratings the model's taste is very uncertain, so the four taste samples pointed in very different directions (that's intended: early on, steering explores). The ranking then filtered out the least likely ideas ("distorted guitar, glitchy"). As ratings accumulate, the samples should agree more and the steered modifiers should become consistent.

## The A/B test: results pending

Whether steered songs are preferred can only be answered with real choices. `uv run python scripts/evaluate_ratings.py` reports the win rate with a 95% Wilson interval. About 30 answered requests gives a first read; about 100 gives a confident one. No result is claimed until then.
