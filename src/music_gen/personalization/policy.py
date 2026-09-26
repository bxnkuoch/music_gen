"""Choose which songs to show in a slate.

Phase 3 policy ("thompson+random"): half the slate from Thompson sampling (the
model's picks under one plausible taste; naturally explores while uncertain), half
uniformly random. The random half keeps collecting unbiased data for evaluation and
stops the system from only ever confirming its own guesses. Phase 5 studies
exploration strategies properly.
"""

from dataclasses import dataclass

import numpy as np

from music_gen.personalization.models import PreferenceModel

POLICY_NAME = "thompson+random"


@dataclass(frozen=True)
class SlateItem:
    song: int  # feature-matrix row
    source: str  # "model" or "random" (logged, never shown to the listener)
    score: float  # the sampled score that picked it (random items too, for analysis)


def select_slate(
    model: PreferenceModel,
    context: str,
    rng: np.random.Generator,
    *,
    groups: np.ndarray,
    exclude: set[int] = frozenset(),
    size: int = 4,
    n_model: int = 2,
) -> list[SlateItem]:
    """Pick `size` songs: `n_model` by Thompson sampling, the rest at random.

    groups: per-song group id (the prompt). At most one song per group per slate, so a
        slate never contains two near-identical seeds of the same prompt.
    exclude: songs to avoid (e.g. recently shown). Ignored if too few songs remain.
    """
    n = len(groups)
    if not 0 <= n_model <= size:
        raise ValueError("n_model must be between 0 and size")
    if len(set(groups.tolist())) < size:
        raise ValueError(f"need at least {size} distinct groups to fill a slate")

    available = np.array([i for i in range(n) if i not in exclude])
    if len(set(groups[available].tolist())) < size:
        available = np.arange(n)  # everything's been shown: start over

    sampled = model.sampled_scores(context, rng)
    picked: list[SlateItem] = []
    used_groups: set[int] = set()

    for i in available[np.argsort(-sampled[available], kind="stable")]:
        if len(picked) == n_model:
            break
        if groups[i] not in used_groups:
            picked.append(SlateItem(int(i), "model", float(sampled[i])))
            used_groups.add(groups[i])

    for i in rng.permutation(available):
        if len(picked) == size:
            break
        if groups[i] not in used_groups:
            picked.append(SlateItem(int(i), "random", float(sampled[i])))
            used_groups.add(groups[i])
    return picked
