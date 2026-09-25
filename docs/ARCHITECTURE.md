# Architecture and design decisions

This records **what** we're building and **why**, so later decisions stay consistent. If you change a decision, update this file and note the reason.

## The one-sentence version

An adaptive music system that learns one listener's taste from **pairwise feedback** and uses the learned preferences to **steer generation** and **rank candidates**. Pretrained models only generate audio and embed it; the learning is ours.

## Data flow (target, end of Phase 5)

```
"chill late-night R&B"
        │
        ▼
Prompt parser ──► tags from a FIXED vocabulary  {mood: chill, genre: r&b, ...}
        │
        ▼
Attribute steering (ours): for each of N=8 candidates, sample a plausible
taste vector from the user model and add the modifier tags / BPM it favours
        │
        ▼
ACE-Step 1.5 (pretrained) ──► 8 clips (different seeds + modifiers)
        │
        ▼
Feature extraction ──► CLAP embedding + tag scores + tempo/loudness/brightness
        │
        ▼
Slate policy (ours): score with the user model → 3 "exploit" + 1 "explore",
with a diversity penalty so the slate isn't 4 near-duplicates
        │
        ▼
You listen (order shuffled, strategy hidden) → pick favourite
        │
        ▼
Pairwise preferences → Bayesian Bradley-Terry update (ours) → next round
```

## Components: pretrained vs built

| Pretrained (we integrate) | Built by us |
|---|---|
| ACE-Step 1.5 turbo: audio generation | Tag vocabulary + prompt parser |
| CLAP: text/audio embeddings, zero-shot tags | Song feature builder (centering, PCA, tags, signal features) |
| MuQ-MuLan: 2nd embedding for experiments | Bayesian Bradley-Terry preference model |
| librosa: tempo, loudness, etc. | Feedback → pairs logic, position-bias handling |
| (optional, later) an LLM for prompt parsing | Attribute steering, slate policy, exploration |
| | Reference-song search, evaluation harness, simulated users |
| | Backend, worker, database schema, UI |

## Key design decisions

### 1. Generation is decoupled from learning
Most feedback is collected from a **pre-generated pool** (Phase 1). The live loop (Phase 4) proves the system works end to end, but it doesn't have to carry data collection. Phase 0 measured about 9 s per clip, so both are feasible on the Mac.

### 2. Song representation
`x_song = [PCA(mean-centered CLAP audio embedding) | CLAP tag scores | tempo, loudness, brightness, onset density | generation params]`
- Centering is required: Phase 0 found every embedding shares a large common direction (cosine similarities around 0.7 even across unrelated genres).
- The **same tag vocabulary** is used for features, prompt parsing, and generation steering. That's what connects the system.

### 3. The user model: Bayesian Bradley-Terry
- `P(you prefer A over B) = sigmoid(w · (x_A − x_B))`. `w` is your taste vector.
- **Bayesian:** we keep a mean *and* an uncertainty for `w` (Gaussian posterior, updated with a Laplace approximation). The uncertainty gives us:
  - **ranking** (use the mean),
  - **exploration** (Thompson sampling: draw a plausible `w` and act on it),
  - **cold start** (the prior),
  from one model instead of three separate systems.
- **Prior:** "you like songs that match your prompt" (positive weight on prompt-audio similarity), neutral on everything else. There's only one real user, so we can't learn a population prior from real data.
- **Why not average your liked songs?** Averaging ignores dislikes, can't tell which features matter, and breaks when your taste has several distinct clusters (lo-fi *and* metal averages to something in between). It stays in as a **baseline**.
- **Upgrades must earn their place:** a Gaussian-process preference model (for multi-cluster taste) or a neural ranker only get adopted if they beat Bradley-Terry on held-out data.

### 4. Feedback
- 4 clips per slate; you pick a favorite and optionally a least favorite. The favorite beats each other clip, which gives 3 pairs.
- Star ratings only become pairs **within one session** (ratings drift over days).
- Display order is **shuffled and logged**, because people favor the first clip they hear.
- **Blind:** the UI never shows which strategy produced a clip, which clip is the exploration pick, or the model's score.

### 5. Evaluation (one real user + simulated users)
- **Main result:** pairwise accuracy on your *future* comparisons after training on your first N (N = 0, 5, 10, 20, 40, 80), against baselines (random, prompt-match, average-of-liked-songs). Confidence intervals are bootstrapped **over sessions**, because pairs from the same slate aren't independent.
- **Interleaving:** a slate mixes personalized and baseline candidates, and we count whose you pick. This works with a single user.
- **Simulated users** (always reported separately and labeled synthetic): their hidden taste is defined on **MuQ** features while the model sees **CLAP**, so recovering it isn't trivially circular. They're used to test correctness, compare exploration strategies, and test cold start.

### 6. Infrastructure
- PostgreSQL + pgvector in Docker; **the ML code runs natively** (Docker on a Mac can't use the GPU).
- A job queue in Postgres (`SELECT … FOR UPDATE SKIP LOCKED`) instead of Redis or Celery.
- The `personalization/` package is **pure** (no database, no web framework): `fit(pairs, X)` and `score(model, X)`. That makes it easy to test and to replay experiments offline.
- Every stored embedding and model records its **model name/version**, so results stay reproducible when things change.

### 7. Licenses
| Component | License | Note |
|---|---|---|
| ACE-Step 1.5 | MIT | Generated audio is unrestricted |
| CLAP (music+speech) | Apache-2.0 | Default embedding |
| MuQ-MuLan weights | CC BY-NC 4.0 | Research/evaluation only; replace before any commercial use |
| diffusers conversion script (vendored) | Apache-2.0 | `scripts/third_party/` |
