"""Simulated listeners with a known, hidden taste, used to test the learning code.

Hidden utility of a song in a context (mood):
    u = genre_pref[genre] + energy_pref[mood][energy] + quirk · z_song
The model only sees the CLAP-based feature space, while this taste is defined on
ground-truth labels + MuQ-MuLan features, so recovering it is a real test, not a
circular one. Results from simulated users are always reported separately from
real ones.
"""

from dataclasses import dataclass

import numpy as np

from music_gen.personalization.models import PreferenceModel
from music_gen.personalization.pairs import Choice
from music_gen.personalization.policy import select_slate


@dataclass(frozen=True)
class SongMeta:
    genres: np.ndarray  # (n,) genre label per song
    energies: np.ndarray  # (n,) energy label per song
    hidden: np.ndarray  # (n, k) standardized features the model doesn't see (MuQ PCs)
    groups: np.ndarray  # (n,) prompt id per song


@dataclass(frozen=True)
class SimulatedUser:
    genre_pref: dict[str, float]
    energy_pref: dict[str, dict[str, float]]  # mood -> energy level -> preference
    quirk: np.ndarray  # (k,) idiosyncratic taste on hidden features
    noise: float = 1.0  # Gumbel noise scale: higher = less consistent choices

    def utility(self, meta: SongMeta, context: str) -> np.ndarray:
        genre = np.array([self.genre_pref[g] for g in meta.genres])
        energy = np.array([self.energy_pref[context][e] for e in meta.energies])
        return genre + energy + meta.hidden @ self.quirk

    def choose(
        self, meta: SongMeta, shown: list[int], context: str, rng: np.random.Generator
    ) -> tuple[int, int]:
        """(favourite, least favourite) under Bradley-Terry-consistent random noise."""
        noisy = self.utility(meta, context)[shown] + self.noise * rng.gumbel(size=len(shown))
        return shown[int(np.argmax(noisy))], shown[int(np.argmin(noisy))]


def make_user(
    rng: np.random.Generator,
    meta: SongMeta,
    contexts: list[str],
    *,
    mood_dependent: bool,
    quirk_scale: float = 0.3,
    noise: float = 1.0,
) -> SimulatedUser:
    """A random listener. If mood_dependent, energy preferences differ per mood
    (e.g. calm music when chilling, intense music when hyped)."""
    genres = sorted(set(meta.genres.tolist()))
    energies = sorted(set(meta.energies.tolist()))

    def energy_table() -> dict[str, float]:
        return {e: float(v) for e, v in zip(energies, rng.normal(0, 1, len(energies)), strict=True)}

    shared = energy_table()
    genre_values = rng.normal(0, 1, len(genres))
    return SimulatedUser(
        genre_pref={g: float(v) for g, v in zip(genres, genre_values, strict=True)},
        energy_pref={c: energy_table() if mood_dependent else shared for c in contexts},
        quirk=rng.normal(0, quirk_scale / np.sqrt(meta.hidden.shape[1]), meta.hidden.shape[1]),
        noise=noise,
    )


def simulate_choices(
    user: SimulatedUser,
    meta: SongMeta,
    contexts: list[str],
    n_slates: int,
    rng: np.random.Generator,
    *,
    model: PreferenceModel | None = None,
    slates_per_session: int = 10,
    slate_size: int = 4,
) -> list[Choice]:
    """Let the user rate `n_slates` slates in random moods.

    model=None: fully random slates (clean data for comparing models).
    model given: slates from the live policy, refitting after every slate.
    """
    from music_gen.personalization.models import RandomModel
    from music_gen.personalization.pairs import to_pairs

    policy_model = model or RandomModel(len(meta.groups))
    choices: list[Choice] = []
    for t in range(n_slates):
        context = contexts[rng.integers(len(contexts))]
        items = select_slate(
            policy_model,
            context,
            rng,
            groups=meta.groups,
            size=slate_size,
            n_model=slate_size // 2 if model else 0,
        )
        shown = [item.song for item in items]
        best, worst = user.choose(meta, shown, context, rng)
        choices.append(Choice(tuple(shown), best, context, worst, session=t // slates_per_session))
        if model is not None:
            model.fit(to_pairs(choices))
    return choices


def top_pick_quality(
    model: PreferenceModel, user: SimulatedUser, meta: SongMeta, contexts: list[str]
) -> float:
    """How good is the model's #1 recommendation, per the user's TRUE taste?

    Percentile of the true utility of the model's top song among the other songs
    (ties count half), averaged over moods: 1.0 = it found the user's favourite,
    0.5 = no better than a random pick.
    """
    result = []
    for c in contexts:
        true = user.utility(meta, c)
        best = true[int(np.argmax(model.scores(c)))]
        worse, ties = (true < best).sum(), np.isclose(true, best).sum() - 1
        result.append(float((worse + 0.5 * ties) / (len(true) - 1)))
    return float(np.mean(result))
