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
