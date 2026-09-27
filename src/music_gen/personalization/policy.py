"""Choose which songs to show in a slate: the exploration vs exploitation trade-off.

A policy fills the slate slot by slot, in this order:
  exploit:  the best song by the posterior MEAN (what the model currently believes)
  thompson: the best song under a FRESH posterior sample per slot (explores while the
            model is uncertain, converges to exploit as it learns; independent samples
            also spread one slate over several plausible tastes)
  random:   uniformly random (unbiased data for evaluation, and a hedge in case the
            model is confidently wrong)
`diversity` > 0 adds a maximal-marginal-relevance penalty: a candidate's score (in
standard deviations across songs) minus diversity × its highest cosine similarity to a
song already in the slate, so the model's picks aren't 4 variations of one idea.

Phase 3 served "thompson+random". Phase 5 compares the presets below in simulation:
results/phase5_exploration.md.
"""

from dataclasses import dataclass

import numpy as np

from music_gen.personalization.models import PreferenceModel


@dataclass(frozen=True)
class SlatePolicy:
    name: str  # logged with every slate (max 32 characters)
    exploit: int = 0
    thompson: int = 0
    random: int = 0
    diversity: float = 0.0

    @property
    def size(self) -> int:
        return self.exploit + self.thompson + self.random


POLICIES = {
    p.name: p
    for p in [
        SlatePolicy("random", random=4),
        SlatePolicy("greedy", exploit=4),
        SlatePolicy("thompson+random", thompson=2, random=2),  # Phase 3
        SlatePolicy("thompson", thompson=4),
        SlatePolicy("thompson:div", thompson=4, diversity=1.0),
        SlatePolicy("exploit3+thompson1", exploit=3, thompson=1),
        SlatePolicy("exploit3+thompson1:div", exploit=3, thompson=1, diversity=1.0),
        SlatePolicy("thompson3+random1", thompson=3, random=1),
    ]
}


@dataclass(frozen=True)
class SlateItem:
    song: int  # feature-matrix row
    source: str  # "exploit", "model" (Thompson) or "random"; logged, never shown
    score: float  # the score that picked it (for random items: one sampled score)


def select_slate(
    model: PreferenceModel,
    context: str,
    rng: np.random.Generator,
    *,
    policy: SlatePolicy,
    groups: np.ndarray,
    features: np.ndarray | None = None,
    exclude: set[int] = frozenset(),
) -> list[SlateItem]:
    """Pick `policy.size` songs.

    groups: per-song group id (the prompt). At most one song per group per slate, so a
        slate never contains two near-identical seeds of the same prompt.
    features: (n, d) song vectors, required when policy.diversity > 0.
    exclude: songs to avoid (e.g. already shown). Ignored if too few songs remain.
    """
    n = len(groups)
    if len(set(groups.tolist())) < policy.size:
        raise ValueError(f"need at least {policy.size} distinct groups to fill a slate")
    if policy.diversity > 0 and features is None:
        raise ValueError("a diversity penalty needs song features")

    available = np.array([i for i in range(n) if i not in exclude], dtype=int)
    if len(set(groups[available].tolist())) < policy.size:
        available = np.arange(n)  # everything's been shown: start over

    unit = None
    if policy.diversity > 0:
        unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    sampled = model.sampled_scores(context, rng)
    picked: list[SlateItem] = []
    used_groups: set[int] = set()

    def pick_best(scores: np.ndarray, source: str) -> None:
        candidates = available[[groups[i] not in used_groups for i in available]]
        value = scores[candidates]
        if unit is not None and picked:
            z = (value - scores.mean()) / (scores.std() or 1.0)
            similarity = unit[candidates] @ unit[[p.song for p in picked]].T
            value = z - policy.diversity * similarity.max(axis=1)
        i = int(candidates[np.argmax(value)])  # first maximum: stable tie-break
        picked.append(SlateItem(i, source, float(scores[i])))
        used_groups.add(groups[i])

    if policy.exploit:
        mean = model.scores(context)
        for _ in range(policy.exploit):
            pick_best(mean, "exploit")
    for k in range(policy.thompson):
        pick_best(sampled if k == 0 else model.sampled_scores(context, rng), "model")

    for i in rng.permutation(available):
        if len(picked) == policy.size:
            break
        if groups[i] not in used_groups:
            picked.append(SlateItem(int(i), "random", float(sampled[i])))
            used_groups.add(groups[i])
    return picked
