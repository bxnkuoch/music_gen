"""Honest evaluation: train on the past, test on the future.

learning_curve(): for N = 0, 5, 10, ... train on the first N choices and measure how
often the model predicts the listener's *later* pairwise preferences. Confidence
intervals resample whole sessions (pairs from one slate/session aren't independent).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from music_gen.personalization.models import PreferenceModel
from music_gen.personalization.pairs import Choice, Pair, to_pairs


def pair_correctness(model: PreferenceModel, pairs: list[Pair]) -> np.ndarray:
    """1 if the model favours the actual winner, 0 if the loser, 0.5 if undecided."""
    probs = np.array([model.prob_prefers(p.winner, p.loser, p.context) for p in pairs])
    return np.where(probs > 0.5, 1.0, np.where(probs < 0.5, 0.0, 0.5))


def bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """CI for the mean of `values`, resampling whole groups (e.g. sessions)."""
    unique = np.unique(groups)
    by_group = [values[groups == g] for g in unique]
    means = []
    for _ in range(n_boot):
        sample = rng.integers(0, len(unique), len(unique))
        means.append(np.concatenate([by_group[i] for i in sample]).mean())
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


@dataclass(frozen=True)
class CurvePoint:
    n_train_choices: int
    n_test_pairs: int
    accuracy: float
    ci_low: float
    ci_high: float


def learning_curve(
    make_model: Callable[[], PreferenceModel],
    choices: list[Choice],
    checkpoints: list[int],
    rng: np.random.Generator,
    *,
    min_test_choices: int = 5,
) -> list[CurvePoint]:
    """`choices` must be in the order they happened."""
    points = []
    for n in checkpoints:
        test = choices[n:]
        if len(test) < min_test_choices:
            break
        model = make_model()
        model.fit(to_pairs(choices[:n]))
        test_pairs = to_pairs(test)
        correct = pair_correctness(model, test_pairs)
        sessions = np.array([c.session for c in test for _ in _pairs_of(c)])
        low, high = bootstrap_ci(correct, sessions, rng)
        points.append(CurvePoint(n, len(test_pairs), float(correct.mean()), low, high))
    return points


def _pairs_of(choice: Choice) -> list[Pair]:
    return to_pairs([choice])


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (e.g. "picked a steered song").

    Wilson's interval stays sensible for small n and for rates near 0 or 1, unlike
    the textbook p ± 1.96·sqrt(p(1-p)/n).
    """
    if n == 0:
        return 0.0, 1.0
    if not 0 <= wins <= n:
        raise ValueError("wins must be between 0 and n")
    p = wins / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
