"""Phase 3 experiment: do the preference models learn simulated listeners' tastes?

Simulated listeners rate random slates of REAL pool songs. Their hidden taste uses
genre/energy labels + MuQ-MuLan features; the models only see the CLAP feature space.
Two populations: fixed taste, and mood-dependent taste (energy preference changes
with mood, like the real listener said theirs does).

Measured per model, averaged over simulated users (95% CI across users):
  * pairwise accuracy on FUTURE choices after N training slates (learning curve)
  * top-pick quality: true-taste percentile of the model's #1 song (0.5 = random)
  * "oracle" = scores songs by the user's TRUE taste: the ceiling, since choices are
    noisy (even a perfect model can't predict every noisy choice)
  * a sweep of the prior variance, to pick the default from evidence

Usage:  uv run python scripts/phase3_simulation.py [--users 30]
Writes: data/results/phase3_simulation.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from music_gen.config import get_settings
from music_gen.db import make_session_factory
from music_gen.features import store
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.logging_setup import configure_logging
from music_gen.personalization.data import load_contexts, load_song_matrix
from music_gen.personalization.evaluation import learning_curve
from music_gen.personalization.models import BradleyTerryModel, MeanLikedModel, RandomModel
from music_gen.personalization.pairs import to_pairs
from music_gen.personalization.simulation import (
    SongMeta,
    make_user,
    simulate_choices,
    top_pick_quality,
)

logger = logging.getLogger("phase3_simulation")
CHECKPOINTS = [0, 5, 10, 20, 40, 80, 120]
N_SLATES = 150
HIDDEN_DIMS = 16


def hidden_features(lookup: dict[int, np.ndarray], song_ids: np.ndarray) -> np.ndarray:
    """Standardized MuQ-MuLan PCs: taste the model can only partly see."""
    muq = np.stack([lookup[s] for s in song_ids])
    centered = muq - muq.mean(axis=0)
    pcs = centered @ np.linalg.svd(centered, full_matrices=False)[2][:HIDDEN_DIMS].T
    return (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)


class OracleModel:
    """Knows the simulated user's true taste. Not learnable: it's the upper bound."""

    name = "oracle"

    def __init__(self, user, meta: SongMeta) -> None:
        self.user, self.meta = user, meta

    def fit(self, pairs) -> None:
        pass

    def scores(self, context: str) -> np.ndarray:
        return self.user.utility(self.meta, context)

    def sampled_scores(self, context: str, rng) -> np.ndarray:
        return self.scores(context)

    def prob_prefers(self, a: int, b: int, context: str) -> float:
        u = self.scores(context)
        return float(1 / (1 + np.exp(-(u[a] - u[b]))))


PRIOR_VARS = [0.01, 0.05, 0.2, 1.0]


def mean_ci(values: list[float]) -> dict:
    v = np.array(values)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return {
        "mean": float(v.mean()),
        "ci_low": float(v.mean() - half),
        "ci_high": float(v.mean() + half),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    contexts = list(load_contexts(Path("configs/contexts.yaml")))
    space = FeatureSpace.load(settings.data_dir / "models" / "feature_space_v1.npz")
    with make_session_factory(settings.database_url)() as session:
        songs = load_song_matrix(session, space, load_tags(Path("configs/tags.yaml")))
        ids, emb = store.load_embeddings(session, f"muq-mulan:{settings.muq_model_id}")
        muq = dict(zip(ids, emb, strict=True))
        # Only songs with MuQ embeddings (the pool; live-request songs get CLAP only).
        songs = songs.subset(np.array([sid in muq for sid in songs.song_ids]))
        meta = SongMeta(
            songs.genres,
            songs.energies,
            hidden_features(muq, songs.song_ids),
            songs.groups,
        )
    x = songs.x
    results: dict = {"n_songs": len(x), "n_features": x.shape[1], "contexts": contexts}
    for population in ("fixed_taste", "mood_dependent_taste"):
        names = ["oracle", "random", "mean_liked", "bt_global", "bt_per_mood"]
        curves = {m: {n: [] for n in CHECKPOINTS} for m in names}
        top_pick = {m: [] for m in names}
        sweep = {v: {"global": [], "per_mood": []} for v in PRIOR_VARS}
        for u in range(args.users):
            rng = np.random.default_rng([args.seed, u, population == "fixed_taste"])
            user = make_user(rng, meta, contexts, mood_dependent=population != "fixed_taste")
            choices = simulate_choices(user, meta, contexts, N_SLATES, rng)
            models = {
                "oracle": lambda user=user: OracleModel(user, meta),
                "random": lambda: RandomModel(len(x)),
                "mean_liked": lambda: MeanLikedModel(x),
                "bt_global": lambda: BradleyTerryModel(x, contexts, per_context=False),
                "bt_per_mood": lambda: BradleyTerryModel(x, contexts, per_context=True),
            }
            for v in PRIOR_VARS:
                for kind, per_context in (("global", False), ("per_mood", True)):
                    (point,) = learning_curve(
                        lambda v=v, pc=per_context: BradleyTerryModel(
                            x, contexts, per_context=pc, prior_var=v
                        ),
                        choices,
                        [60],
                        rng,
                    )
                    sweep[v][kind].append(point.accuracy)
            for name, make in models.items():
                for point in learning_curve(make, choices, CHECKPOINTS, rng):
                    curves[name][point.n_train_choices].append(point.accuracy)
                model = make()
                model.fit(to_pairs(choices[:120]))
                top_pick[name].append(top_pick_quality(model, user, meta, contexts))
            logger.info("%s: user %d/%d done", population, u + 1, args.users)
        results[population] = {
            "pairwise_accuracy_on_future_choices": {
                m: {str(n): mean_ci(v) for n, v in c.items() if v} for m, c in curves.items()
            },
            "top_pick_quality_after_120_slates": {m: mean_ci(v) for m, v in top_pick.items()},
            "prior_var_sweep_accuracy_after_60_slates": {
                str(v): {k: mean_ci(a) for k, a in r.items()} for v, r in sweep.items()
            },
        }

    out = settings.results_dir / "phase3_simulation.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
