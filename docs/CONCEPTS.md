# Concepts, in plain language

A glossary for the ideas this project uses, written for someone meeting them for the first time.

**Embedding.** A list of numbers (here, 512 of them) that summarizes a song or a sentence. Similar songs get similar lists. "Similar" is measured with **cosine similarity**: 1 means the same direction, 0 means unrelated.

**Shared text-audio space (CLAP, MuQ-MuLan).** These models were trained so that a song and a sentence describing it land near each other. That lets us score a song against *"dark synthwave"* without any labeled data. This is called **zero-shot tagging**.

**Anisotropy.** When all the embeddings point roughly the same way, so even unrelated songs have cosine around 0.7. The fix is to subtract the average embedding first (**mean-centering**) so the differences stand out.

**PCA.** Compresses 512 numbers down to about 32 while keeping as much of the variation as possible. Fewer numbers means the preference model needs less of your feedback to learn.

**Diffusion model (ACE-Step).** Generates audio by starting from random noise and refining it step by step. The **seed** fixes the starting noise, so the same seed and prompt give the same song. "Turbo" means it was trained to need only 8 steps.

**Pairwise preference.** Instead of "rate this 7/10", we ask "A or B?". People answer comparisons much more consistently than they give absolute scores.

**Bradley-Terry model.** A classic model for comparisons: each song has a score, and the chance you prefer A over B depends on the score difference: `P(A > B) = sigmoid(score_A − score_B)`. Here, `score = w · features`, and **`w` is your learned taste**. Each number in `w` says how much you like one feature, for example "+0.8 on warm Rhodes".

**Bayesian.** Instead of one best guess for `w`, we keep a *range of plausible* `w`s (a mean plus an uncertainty). The uncertainty shrinks as you give more feedback.

**Laplace approximation.** A standard shortcut for approximating that range of plausible `w`s with a bell curve (a Gaussian), so we can update it quickly after every rating.

**Exploration vs. exploitation.** *Exploit* means playing what the model is confident you'll like. *Explore* means trying something uncertain so the model can learn. Too much exploitation traps you in a bubble; too much exploration wastes your time.

**Thompson sampling.** A simple, well-studied way to balance the two: draw one plausible `w` from the uncertainty range and act as if it were true. When the model is unsure, the draws vary a lot, so it naturally explores more.

**Slate.** The set of 4 clips shown together in one round.

**Position bias.** People tend to pick the first option they hear. We shuffle the order and record it, so we can measure the effect and correct for it.

**Cold start.** How well the system does before it knows anything about you. It relies on the **prior** (here: "you probably like songs that match your prompt").

**Held-out evaluation.** Train on your first N comparisons and test on later ones the model hasn't seen. Testing on the training data would overstate how good the model is.

**Learning curve.** A plot of accuracy against the number of feedback items. The project's headline result is this curve compared with baselines.

**Bootstrap confidence interval.** Resample your sessions many times and recompute the metric each time. The spread tells you how much of the result could be luck.

**Interleaving.** Mixing clips from two strategies in one slate and counting whose clip you pick. It compares two systems with far less data than a classic A/B test.

**Simulated user.** A fake listener with a known hidden taste. Because we know the right answer, we can check whether the model finds it.

## Engineering terms (Phase 1)

**Database migration (Alembic).** A versioned script that changes the database structure, for example "add a table". Migrations are stored in `alembic/versions/` and applied in order with `alembic upgrade head`, so every copy of the database ends up identical. You never edit tables by hand.

**Transaction.** A group of database changes that either all happen or none do. Saving a song row and marking its job "done" is one transaction, so you can never have one without the other.

**Atomic file write.** Write to a temporary file, then rename it. A rename is instant, so the final filename either doesn't exist yet or holds a complete file, never half of one.

**Idempotent.** Safe to run twice: the second run changes nothing. `generate_pool.py` is idempotent because unique constraints stop duplicate prompts and jobs.

**Unique constraint / `NULLS NOT DISTINCT`.** A database rule such as "no two jobs with the same prompt, seed and settings". By default Postgres treats two empty (NULL) values as *different*, which would quietly allow duplicates. `NULLS NOT DISTINCT` fixes that.

**Job queue.** A table of work items with a status (`pending → running → done/failed`). If the program crashes, jobs stuck in `running` are put back to `pending` on the next run.

**Fake (in tests).** A tiny stand-in for something slow or large, like the 6 GB music model, that behaves the same from the outside. It lets tests check *our* logic in milliseconds, and simulate rare failures on demand.

## Feature terms (Phase 2)

**pgvector.** A Postgres extension that stores embeddings as a column type and finds the nearest ones fast (`ORDER BY embedding <=> target`). It powers "find similar songs".

**Cosine distance.** 1 − cosine similarity: 0 means the same direction, 1 means unrelated, 2 means opposite.

**Zero-shot calibration.** Some tag descriptions are "close to everything", so they win for every song. Standardizing each tag's scores across all songs (subtract the average, divide by the spread) asks "is this song *more* chill than usual?" instead of "is chill the closest word?". It needs no labels.

**Linear probe.** A simple classifier trained on features to test what information they contain. If a straight-line boundary can separate genres, then a linear preference model can learn "I like this genre". It's evaluated on held-out songs.

**Grouped cross-validation.** Split data so related items (the 3 seeds of one prompt) are always on the same side of the train/test split. Otherwise the test is too easy, because the model has already seen a near-copy.

**Octave error.** A tempo estimator reporting half or double the true BPM (e.g. 70 vs 140). It's common and understandable, because both fit the beat.

## Learning terms (Phase 3)

**Prior variance.** How far the model expects your taste weights to stray from zero before seeing data. Small = cautious (needs more evidence before believing a feature matters). We tried 0.01–1.0 in simulation. It barely mattered, and 0.01 was slightly best.

**Newton's method (damped).** The algorithm that finds the most likely taste vector. It jumps toward the best answer using the curve's slope and curvature, and halves the jump if a step would make things worse ("damped"), so it never diverges.

**Oracle.** A cheating model that knows a simulated listener's true taste. Because choices are noisy, even the oracle is only right ~78% of the time. That's the ceiling, so 64% means "about half way from coin-flip to perfect".

**Per-mood model.** Taste = shared taste + a small mood-specific adjustment. It can learn "I like energetic music when hyped, calm music when chilling", but it needs more ratings, because each mood's adjustment only learns from that mood's ratings.

**Slate policy.** The rule that picks which 4 songs to show. Ours: 2 from Thompson sampling (the model's current best guesses, with some healthy randomness), 2 uniformly random (to keep exploring and keep the data fair for evaluation).

**Blind evaluation.** You never see which songs the model picked. If you knew, you might (unconsciously) favor or punish them.

## Generation terms (Phase 4)

**Steering.** Nudging what gets generated toward your taste by adding words to the prompt ("…, plucked guitar, warm"). The words come from the tag weights of a taste vector sampled from the model.

**Candidate generation + re-ranking.** Make more options than you'll show (8), then let the model pick the best (4). It's the standard design of recommender systems: a cheap, broad "candidate" step, then a precise "ranking" step.

**A/B test (built in).** Each request slate has 2 steered and 2 plain songs, shuffled. If steering does nothing, you'd pick a steered song 50% of the time. Consistently more than 50% is evidence that personalization helps.

**Wilson confidence interval.** A range for a win rate that stays honest with small counts. After 10 requests, even 7/10 steered picks gives a range of about 40–89%: not yet convincing. That's why ~100 requests are needed for a confident answer.

**Worker process.** A separate program that does slow work (generating music) in the background. The web API just adds jobs to the database queue and reports progress.
