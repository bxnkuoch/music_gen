# Phase 3 results: simulated listeners

> ⚠️ **Synthetic data.** These are simulated listeners, used to check that the learning code works and to choose defaults. Real results come from `scripts/evaluate_ratings.py` once real ratings exist.

**Date:** 2026-09-25 · **Reproduce:** `uv run python scripts/phase3_simulation.py --users 30` · **Raw:** `data/results/phase3_simulation.json`

## Setup
- **30 simulated listeners per population** rate random 4-song slates of the **real 540-song pool**, 150 slates each, in random moods (5 moods), with sessions of 10 slates.
- **Hidden taste** = genre preference + energy preference + a small quirk on MuQ-MuLan features. The models only see the CLAP feature space, so they must *generalize*; they can't look the answer up.
- **Choices are noisy**, as with real people: Gumbel noise, consistent with the Bradley-Terry model.
- Two populations: **fixed taste**, and **mood-dependent taste** (energy preference changes with mood).
- **Metric:** accuracy at predicting each listener's *later* pairwise choices after training on their first N slates. 95% CI across listeners.
- **Oracle** = knows the true hidden taste. Because choices are noisy, even it can't be right every time; it marks the **ceiling**.

## Learning curves

**Fixed taste**

| Slates rated | 5 | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|
| Oracle (ceiling) | 78% | 78% | 78% | 78% | 78% | 78% |
| **Bayesian BT (global)** | **55%** | **58%** | **59%** | **61%** | **63%** | **64%** |
| Bayesian BT (per mood) | 55% | 57% | 59% | 60% | 62% | 62% |
| Average-of-liked baseline | 51% | 52% | 54% | 55% | 57% | 59% |
| Random | 50% | 50% | 50% | 50% | 50% | 50% |

**Mood-dependent taste**

| Slates rated | 5 | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|
| Oracle (ceiling) | 77% | 77% | 77% | 77% | 77% | 77% |
| Bayesian BT (global) | 53% | 54% | 55% | 56% | 59% | 59% |
| **Bayesian BT (per mood)** | **53%** | **54%** | **55%** | **57%** | **60%** | **60%** |
| Average-of-liked baseline | 51% | 52% | 52% | 53% | 54% | 55% |

**Top-pick quality after 120 slates** (true-taste percentile of the model's #1 song; 50% = random pick, 100% = the listener's favorite):

| | Fixed taste | Mood-dependent |
|---|---|---|
| Bayesian BT (global) | **83%** [78–89] | 72% [64–81] |
| Bayesian BT (per mood) | 82% [77–87] | **76%** [71–80] |
| Average-of-liked | 59% [51–68] | 57% [49–65] |
| Random | 41% [32–51] | 54% [46–62] |

## Findings
1. **The Bayesian model beats the common baseline everywhere.** For fixed taste the gap is clear from about 10 slates on (58% [56–59] vs 52% [51–54]), and it reaches 64% vs 59% at 120. Its top pick lands in the listener's top ~17% vs. ~41% for the baseline.
2. **Much of the gap to the ceiling comes from the features, not the model.** Part of each listener's taste lives in features the model can't see (by design), and genre is only ~50% linearly recoverable (Phase 2). Better song representations are the main lever for improvement.
3. **Per-mood modeling: a small, not-yet-significant benefit where taste really is mood-dependent** (60% vs 59%; top pick 76% vs 72%, CIs overlap), and a small cost where it isn't. With five moods, 120 slates is only about 24 per mood. → The app **serves the global model** and `evaluate_ratings.py` compares both on real data. Switch once the real curve says so.
4. **Prior variance barely matters** in the 0.01–1.0 range (≤1.5 points at 60 slates), with a consistent edge for **0.01**, now the default. The model isn't fragile to this choice.

## What this does and doesn't show
- ✅ The pipeline learns a hidden taste from noisy pairwise feedback, through the real feature space, and does it faster than the standard baseline.
- ❌ It doesn't show how well it'll do for a *real* listener, whose taste may depend on things neither the simulation nor the features capture. That's what the real learning curve is for.
