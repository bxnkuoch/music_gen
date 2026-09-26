# Phase 2 results: how trustworthy are the song features?

**Date:** 2026-09-25 · **Songs:** 540 (15 genres × 4 moods × 3 energy levels × 3 seeds)
**Reproduce:** `uv run python scripts/compute_features.py && uv run python scripts/phase2_validation.py`
**Raw numbers:** `data/results/phase2_features.json`

Every song was generated from known attributes, so each feature can be checked against ground truth. Numbers below are for all 540 songs.

## 1. Tempo: does ACE-Step follow the requested BPM?

| Outcome | Share of songs |
|---|---|
| Within 4% of requested BPM | **60%** |
| Half or double (an "octave error"; could be the generator or the estimator) | 18% |
| Neither | 22% |

- Best genres (hit or octave): anime soundtrack 94%, fantasy adventure game 89%, house/jazz/pop/hyperpop 83%. Worst: synthwave and ambient at 58%. Ambient has no clear beat, so tempo there is hard to define anyway.
- **Implication:** BPM is a *usable but unreliable* control knob. The measured tempo, not the requested one, is what goes into the features.

## 2. Zero-shot tags: can the embeddings *name* the genre and mood?

Top-1 accuracy, picking the best-matching tag text per song:

| | CLAP raw | CLAP calibrated | MuQ raw | MuQ calibrated | Chance |
|---|---|---|---|---|---|
| Genre (15) | 21% | 26% | 21% | **30%** | 7% |
| Mood (4) | 32% | 31% | 27% | 35% | 25% |

- **Raw zero-shot scores are biased.** One label, "chill", wins for most songs, because some texts are closer to *all* audio. Calibrating (standardizing each tag's scores across songs, which uses no labels) fixes much of this. For example, CLAP's heavy-metal recognition goes from 6% to 64%. The feature space standardizes every column, so the preference model always gets calibrated tag scores.
- **Genre naming is modest.** Tags are useful features, but individual tag scores shouldn't be treated as reliable labels.

## 3. What's *linearly* in the features? (the number that matters for Phase 3)

The Phase 3 preference model is linear, so the right question is whether a linear classifier can recover each attribute. This uses logistic regression with 5-fold cross-validation, **holding out whole prompts** so seed siblings never leak between training and test data.

| Feature block | Genre | Mood | Energy |
|---|---|---|---|
| CLAP embedding, 32 PCs | 50% | 32% | 73% |
| MuQ-MuLan embedding, 32 PCs | 51% | 31% | **91%** |
| CLAP tag scores (45) | 49% | 31% | 76% |
| Signal features only (4) | 12% | 31% | 74% |
| **Full feature space (81)** | **50%** | 29% | 79% |
| *Chance (majority class)* | *7%* | *25%* | *33%* |

Genre recall with the full feature space, **the listener's favorite styles in bold**:

| **anime soundtrack** | **lo-fi guitar** | **lo-fi hip hop** | **fantasy adventure game** | **hyperpop** | **pop** |
|---|---|---|---|---|---|
| 58% | 58% | 56% | 53% | 53% | 33% |

(Others: heavy metal 75%, synthwave 67%, jazz 61%, classical 50%, acoustic folk 47%, reggae 47%, ambient 42%, house 25%, r&b 22%.)

**What this means:**
- **Genre and energy are clearly encoded.** Genre is 7× above chance, and energy is up to 91%. A linear taste model can learn "likes lo-fi guitar, dislikes metal, likes energetic".
- **Mood is barely above chance, for every feature block and both embedding models.** Since two independent models agree, the most likely explanation is that ACE-Step doesn't audibly render mood words ("dark", "melancholic"). The alternative, that neither model can hear mood, is less likely. → Mood becomes **context the listener provides** (ARCHITECTURE.md §7). Controlling mood through major/minor key is a future lever.
- **Pop is the least distinctive of the listener's genres** (33%). It overlaps with r&b, house and synthwave. Expect the model to learn "likes pop" more slowly than "likes anime soundtracks".
- **MuQ-MuLan captures energy much better than CLAP** (91% vs. 73%). If CLAP-based preferences plateau in Phase 7, trying MuQ is the first experiment. It's already computed for every song.
- The signal features add little beyond the embeddings for genre, but they're fully interpretable (e.g. "prefers slower tempos") and model-free, so they stay in.

## 4. Similar-song search

Share of each song's 5 nearest neighbors (excluding its own prompt's seed siblings) that have the same genre:

| | Raw | Mean-centered | Chance |
|---|---|---|---|
| CLAP | 30.9% | 31.6% | 6.5% |
| MuQ-MuLan | 28.6% | 31.4% | 6.5% |

Nearest neighbors share the genre about 5× more often than chance. Centering helps slightly. Neighbors often share *energy and texture* across genres (e.g. a chill R&B song's neighbors are chill ambient and house songs), which is reasonable behavior for "sounds like".

## 5. Feature space

- 32 principal components keep **94%** of CLAP embedding variance (8 keep 72%, 16 keep 86%).
- Total: 81 standardized features = 32 PCs + 45 tag scores + 4 signal features.

## Caveats
- Ground truth is *what was requested*, not what was actually produced. Where the generator ignored a request, the features aren't necessarily wrong. That's the ambiguity behind the mood result.
- One pool from one generator. The numbers describe ACE-Step output, not music in general.
