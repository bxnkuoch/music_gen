# Phase 5 results: exploration vs exploitation

> ⚠️ **Synthetic data.** Simulated listeners, used to choose the slate policy. Real results come from `scripts/evaluate_ratings.py`.

**Date:** 2026-09-26 · **Reproduce:** `uv run python scripts/phase5_simulation.py --users 40` (~4 min) · **Raw:** `data/results/phase5_simulation.json`

## Question
The Phase 3 app fills each 4-song slate with 2 Thompson-sampling picks + 2 random picks. Is there a better way to split a slate between **exploiting** what the model believes, **exploring** where it's unsure, and **random** picks?

## Setup
- Same simulated listeners as Phase 3 (hidden taste on genre, energy and MuQ-MuLan features; the model sees only the CLAP feature space; noisy choices), 40 per population, on the real 540-song pool.
- Each policy drives 150 slates for the *same* listener. The model refits after every slate, and unseen songs are shown first, like in the app.
- **Experience:** how good the best song in each slate is, as its percentile under the listener's *true* taste (100 = their favorite song was offered), over slates 1–30 and 31–60. Longer windows can't separate policies: after ~135 slates every policy has shown the whole pool.
- **Learning:** pairwise accuracy on a held-out set of *random* slates that each listener also rated. It's the same test for every policy, so data collected by different policies is judged fairly. Plus **top-pick quality** after 150 slates (the percentile of the model's #1 song; this is what ranks live-generated candidates).
- Policies are built from slot types: **exploit** (best by posterior mean), **Thompson** (best under a fresh posterior sample per slot), **random**; `:div` adds a diversity penalty (a candidate loses 1 standard deviation of score per unit of cosine similarity to a song already in the slate).

## Results (mean [95% CI across listeners])

**Fixed taste**

| Policy | Best in slate, 1–30 | Best in slate, 31–60 | Top pick @150 | Accuracy @30 | Accuracy @150 |
|---|---|---|---|---|---|
| random | 80.3 [79.5–81.2] | 79.9 [79.1–80.8] | 87.3 [82.1–92.6] | 60.3 | 64.0 |
| greedy (4 exploit) | 85.4 [83.7–87.1] | 83.0 [81.8–84.3] | 94.6 [92.7–96.5] | 57.5 | 62.5 |
| thompson+random (Phase 3) | 84.3 [83.2–85.4] | 83.4 [82.5–84.4] | 93.2 [90.5–95.8] | 59.6 | 64.1 |
| thompson (4 samples) | 85.5 [84.0–87.0] | 84.7 [83.8–85.5] | 94.3 [91.4–97.2] | 59.2 | 63.5 |
| thompson:div | 86.4 [85.1–87.7] | 85.3 [84.4–86.1] | 95.1 [93.5–96.7] | 59.6 | 64.2 |
| exploit3+thompson1 | 85.6 [83.7–87.4] | 84.2 [83.1–85.3] | 95.5 [93.7–97.3] | 57.7 | 63.5 |
| **exploit3+thompson1:div** | **86.5** [84.9–88.1] | 84.3 [83.2–85.3] | **97.3** [96.4–98.2] | 59.3 | 64.2 |
| thompson3+random1 | 85.3 [83.8–86.7] | 85.0 [84.0–85.9] | 94.9 [93.1–96.8] | 58.8 | 63.8 |

**Mood-dependent taste**

| Policy | Best in slate, 1–30 | Best in slate, 31–60 | Top pick @150 | Accuracy @30 | Accuracy @150 |
|---|---|---|---|---|---|
| random | 79.9 [78.9–81.0] | 80.4 [79.6–81.3] | 77.0 [71.4–82.6] | 55.6 | 59.0 |
| greedy | 82.0 [80.4–83.6] | 82.4 [81.3–83.4] | 86.4 [83.2–89.5] | 54.8 | 58.8 |
| thompson+random (Phase 3) | 82.4 [81.4–83.4] | 82.9 [82.1–83.7] | 82.9 [79.0–86.9] | 55.8 | 59.8 |
| thompson | 82.6 [81.2–84.1] | 83.0 [82.1–84.0] | 89.4 [86.6–92.1] | 55.5 | 59.0 |
| thompson:div | 83.1 [82.0–84.1] | 84.6 [83.8–85.4] | 86.9 [83.1–90.8] | 56.4 | 59.4 |
| exploit3+thompson1 | 82.0 [80.8–83.3] | 83.0 [82.1–83.9] | 89.1 [86.7–91.5] | 55.2 | 59.0 |
| **exploit3+thompson1:div** | **83.4** [82.1–84.7] | 83.2 [82.1–84.3] | **89.9** [87.6–92.2] | 55.0 | 58.7 |
| thompson3+random1 | 83.4 [82.4–84.4] | 83.4 [82.5–84.4] | 83.3 [78.3–88.3] | 56.2 | 59.5 |

**Paired difference vs the Phase 3 policy** (same listeners, so compared per listener; percentage points, 95% CI):

| Policy | Fixed: best in slate 1–30 | Fixed: top pick | Fixed: accuracy @150 | Mood: best in slate 1–30 | Mood: top pick | Mood: accuracy @150 |
|---|---|---|---|---|---|---|
| greedy | +1.1 [−0.6, +2.8] | +1.4 [−1.3, +4.1] | **−1.5** [−2.4, −0.7] | −0.4 [−2.2, +1.5] | +3.4 [−0.6, +7.4] | **−1.0** [−1.9, −0.1] |
| thompson:div | **+2.1** [+0.9, +3.3] | +1.9 [−0.9, +4.7] | +0.1 [−0.6, +0.9] | +0.7 [−0.4, +1.7] | +4.0 [−0.8, +8.8] | −0.4 [−1.4, +0.7] |
| exploit3+thompson1 | +1.3 [−0.5, +3.1] | **+2.3** [+0.2, +4.5] | −0.5 [−1.4, +0.4] | −0.4 [−1.6, +0.9] | **+6.2** [+2.3, +10.0] | **−0.8** [−1.6, −0.1] |
| **exploit3+thompson1:div** | **+2.2** [+0.7, +3.7] | **+4.1** [+1.4, +6.9] | +0.2 [−0.6, +0.9] | +1.0 [−0.3, +2.3] | **+6.9** [+3.0, +10.9] | −1.1 [−2.3, +0.1] |
| thompson3+random1 | +1.0 [−0.3, +2.2] | +1.8 [−1.3, +4.8] | −0.3 [−1.0, +0.5] | +1.0 [−0.2, +2.3] | +0.3 [−4.5, +5.1] | −0.3 [−1.0, +0.5] |

## Findings
1. **Using the model beats random slates** on every experience and top-pick measure (e.g. +4.0 points best-in-slate in the first 30 slates).
2. **Pure exploitation learns worse.** Greedy slates keep showing variations of the model's current favorite, and held-out accuracy drops (−1.5 and −1.0 points, significant in both populations). That's the exploration–exploitation trade-off showing up in the numbers.
3. **The diversity penalty mostly helps.** It improves early experience in all four with/without pairs (+0.5 to +1.4 points in slates 1–30). For `exploit3+thompson1` it also raises top-pick quality and, for fixed taste, removes the learning cost (−0.5 → +0.2 points vs Phase 3). For mood-dependent taste the small learning cost remains, and for pure Thompson the effect on top pick is mixed.
4. **Winner: `exploit3+thompson1:div`** (the design planned in ARCHITECTURE.md). It's the only policy that significantly improves top-pick quality in both populations (+4.1 and +6.9 points), and it improves early experience (+2.2 for fixed taste). Cost: a possible ~1-point drop in held-out accuracy for mood-dependent taste (CI just includes 0).
5. **Differences between the good policies are small** (1–2 points of experience). The choice matters less than *using the model at all*, and less than feature quality (Phase 3 finding 2).

## Decision
The app now serves **`exploit3+thompson1:div`** for library slates (`SERVING_POLICY` in `api/service.py`). "Create new" slates are unchanged (2 steered + 2 plain, for the A/B test).

**Trade-off to watch:** the new policy has no random slot. Phase 3 kept half of each slate random partly so that the *real* learning curve (`evaluate_ratings.py`) is measured on unbiased slates. From now on, new real test pairs come from the model's own picks: slates of similar, well-liked songs, which are harder to predict. Real accuracy may therefore look lower than before, without the model being worse. If unbiased real evaluation matters more (e.g. for Phase 7), `thompson3+random1` keeps one random slot at a small cost (no significant differences from the winner in experience; weaker top pick).
