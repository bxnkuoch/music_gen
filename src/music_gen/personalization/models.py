"""Preference models behind one small interface, so they can be compared fairly.

Songs are rows of a feature matrix X (n_songs, d) from the Phase 2 feature space.
"""

from typing import Protocol

import numpy as np
from scipy.special import expit

from music_gen.personalization import bradley_terry
from music_gen.personalization.pairs import Pair


class PreferenceModel(Protocol):
    name: str

    def fit(self, pairs: list[Pair]) -> None: ...

    def scores(self, context: str) -> np.ndarray:
        """Expected preference score of every song in this context, shape (n_songs,)."""
        ...

    def sampled_scores(self, context: str, rng: np.random.Generator) -> np.ndarray:
        """Scores under one plausible taste (Thompson sampling); = scores() if no uncertainty."""
        ...

    def prob_prefers(self, a: int, b: int, context: str) -> float: ...


class RandomModel:
    """Baseline: knows nothing. Every comparison is a coin flip."""

    name = "random"

    def __init__(self, n_songs: int) -> None:
        self.n_songs = n_songs

    def fit(self, pairs: list[Pair]) -> None:
        pass

    def scores(self, context: str) -> np.ndarray:
        return np.zeros(self.n_songs)

    def sampled_scores(self, context: str, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.n_songs)

    def prob_prefers(self, a: int, b: int, context: str) -> float:
        return 0.5


class MeanLikedModel:
    """Baseline: "average the songs you picked, recommend what's similar".

    The common simple approach. It ignores which features matter, ignores what you
    rejected, and blends distinct tastes into one average.
    """

    name = "mean_liked"

    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.profile = np.zeros(x.shape[1])

    def fit(self, pairs: list[Pair]) -> None:
        winners = sorted({p.winner for p in pairs})
        self.profile = self.x[winners].mean(axis=0) if winners else np.zeros(self.x.shape[1])

    def scores(self, context: str) -> np.ndarray:
        return self.x @ self.profile

    def sampled_scores(self, context: str, rng: np.random.Generator) -> np.ndarray:
        return self.scores(context)

    def prob_prefers(self, a: int, b: int, context: str) -> float:
        return float(expit((self.x[a] - self.x[b]) @ self.profile))


class BradleyTerryModel:
    """Bayesian Bradley-Terry, optionally with a per-context (mood) adjustment.

    per_context=False: one taste w for every mood.
    per_context=True:  taste = w_shared + delta_mood. Implemented by giving each song
        extra feature copies that are "switched on" only in its context; the prior on
        each delta is tighter (context_var_ratio), so a mood only departs from the
        shared taste when the data says so.
    """

    def __init__(
        self,
        x: np.ndarray,
        contexts: list[str],
        *,
        per_context: bool,
        prior_var: float = 0.01,  # best in the Phase 3 simulation sweep (0.01-1.0)
        context_var_ratio: float = 0.5,
    ) -> None:
        self.x = x
        self.contexts = list(contexts)
        self.per_context = per_context
        self.name = "bt_per_mood" if per_context else "bt_global"
        d = x.shape[1]
        blocks = 1 + (len(self.contexts) if per_context else 0)
        self.prior_mean = np.zeros(d * blocks)
        self.prior_var = np.full(d * blocks, prior_var)
        self.prior_var[d:] *= context_var_ratio
        self.posterior = bradley_terry.Posterior(self.prior_mean, np.diag(self.prior_var))

    def design(self, rows: np.ndarray, context: str) -> np.ndarray:
        base = self.x[rows]
        if not self.per_context:
            return base
        if context not in self.contexts:
            raise ValueError(f"unknown context {context!r}; expected one of {self.contexts}")
        k = self.contexts.index(context)
        d = self.x.shape[1]
        out = np.zeros((len(rows), d * (1 + len(self.contexts))))
        out[:, :d] = base
        out[:, d * (1 + k) : d * (2 + k)] = base
        return out

    def fit(self, pairs: list[Pair]) -> None:
        if not pairs:
            self.posterior = bradley_terry.Posterior(self.prior_mean, np.diag(self.prior_var))
            return
        diffs = np.vstack(
            [
                self.design(np.array([p.winner]), p.context)
                - self.design(np.array([p.loser]), p.context)
                for p in pairs
            ]
        )
        self.posterior = bradley_terry.fit(diffs, self.prior_mean, self.prior_var)

    def scores(self, context: str) -> np.ndarray:
        return self.posterior.utility(self.design(np.arange(len(self.x)), context))

    def sampled_scores(self, context: str, rng: np.random.Generator) -> np.ndarray:
        return self.design(np.arange(len(self.x)), context) @ self.posterior.sample(rng)

    def prob_prefers(self, a: int, b: int, context: str) -> float:
        xa, xb = self.design(np.array([a, b]), context)
        return float(self.posterior.prob_prefers(xa, xb)[0])

    def weights(self, context: str | None = None) -> np.ndarray:
        """Learned taste per original feature (shared, plus the mood's adjustment)."""
        d = self.x.shape[1]
        w = self.posterior.mean[:d].copy()
        if self.per_context and context is not None:
            k = self.contexts.index(context)
            w += self.posterior.mean[d * (1 + k) : d * (2 + k)]
        return w
