"""Learning curve on YOUR real ratings: does the model predict your future choices?

For N = 0, 5, 10, 20, ... train each model on your first N answered slates and test
on every later one. 95% CIs resample whole rating sessions.

Usage:  uv run python scripts/evaluate_ratings.py
Writes: data/results/real_learning_curve.json
Needs at least ~15 answered slates to say anything; ~100+ to say something solid.
"""

import json
import logging
from pathlib import Path

import numpy as np

from music_gen.api.service import Recommender
from music_gen.config import get_settings
from music_gen.db import make_session_factory
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.logging_setup import configure_logging
from music_gen.personalization.data import load_contexts, load_song_matrix
from music_gen.personalization.evaluation import learning_curve
from music_gen.personalization.models import BradleyTerryModel, MeanLikedModel, RandomModel

logger = logging.getLogger("evaluate_ratings")
CHECKPOINTS = [0, 5, 10, 20, 40, 80, 120, 160, 240, 320]


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    contexts = load_contexts(Path("configs/contexts.yaml"))
    space = FeatureSpace.load(settings.data_dir / "models" / "feature_space_v1.npz")
    with make_session_factory(settings.database_url)() as session:
        songs = load_song_matrix(session, space, load_tags(Path("configs/tags.yaml")))
        choices = Recommender(songs, contexts).load_choices(session)
    logger.info("%d answered slates", len(choices))
    if len(choices) < 15:
        print(f"Only {len(choices)} ratings so far: rate at least 15 slates first.")
        return

    x, names = songs.x, list(contexts)
    models = {
        "random": lambda: RandomModel(len(x)),
        "mean_liked": lambda: MeanLikedModel(x),
        "bt_global": lambda: BradleyTerryModel(x, names, per_context=False),
        "bt_per_mood": lambda: BradleyTerryModel(x, names, per_context=True),
    }
    rng = np.random.default_rng(0)
    results = {"n_ratings": len(choices), "curves": {}}
    print(f"\nPairwise accuracy on your later choices ({len(choices)} ratings)\n")
    print(f"{'trained on':>11} | " + " | ".join(f"{m:>20}" for m in models))
    curves = {m: learning_curve(make, choices, CHECKPOINTS, rng) for m, make in models.items()}
    for i, point in enumerate(curves["random"]):
        row = [
            f"{c[i].accuracy:.2f} [{c[i].ci_low:.2f}-{c[i].ci_high:.2f}]" for c in curves.values()
        ]
        print(f"{point.n_train_choices:>11} | " + " | ".join(f"{r:>20}" for r in row))
    for m, c in curves.items():
        results["curves"][m] = [p.__dict__ for p in c]

    out = settings.results_dir / "real_learning_curve.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
