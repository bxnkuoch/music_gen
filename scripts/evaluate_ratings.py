"""Your real ratings: does the model predict your choices, and does steering help?

1. Learning curve: for N = 0, 5, 10, 20, ... train each model on your first N
   answered slates and test on every later one. 95% CIs resample whole sessions.
2. Steering A/B (Phase 4): on "Create new" slates (2 steered + 2 plain songs, blind),
   how often did you pick a steered one? 50% = steering makes no difference.

Usage:  uv run python scripts/evaluate_ratings.py
Writes: data/results/real_learning_curve.json
Needs at least ~15 answered slates to say anything; ~100+ to say something solid.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import select

from music_gen.api.service import Recommender
from music_gen.config import get_settings
from music_gen.db import GenerationJob, ListenerRequest, Slate, Song, make_session_factory
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.logging_setup import configure_logging
from music_gen.personalization.data import load_contexts, load_song_matrix
from music_gen.personalization.evaluation import learning_curve, wilson_interval
from music_gen.personalization.models import BradleyTerryModel, MeanLikedModel, RandomModel

logger = logging.getLogger("evaluate_ratings")
CHECKPOINTS = [0, 5, 10, 20, 40, 80, 120, 160, 240, 320]


def steering_ab(session) -> dict:
    """Among answered request slates: how often was the favourite a steered song?"""
    rows = session.execute(
        select(GenerationJob.variant)
        .join(Song, Song.job_id == GenerationJob.id)
        .join(Slate, Slate.chosen_song_id == Song.id)
        .join(ListenerRequest, ListenerRequest.session_id == Slate.session_id)
        .where(Slate.answered_at.is_not(None))
    ).all()
    n = len(rows)
    wins = sum(1 for (variant,) in rows if variant == "steered")
    low, high = wilson_interval(wins, n)
    return {
        "n_request_slates": n,
        "steered_picked": wins,
        "rate": wins / n if n else None,
        "ci_low": low,
        "ci_high": high,
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    contexts = load_contexts(Path("configs/contexts.yaml"))
    space = FeatureSpace.load(settings.data_dir / "models" / "feature_space_v1.npz")
    with make_session_factory(settings.database_url)() as session:
        songs = load_song_matrix(session, space, load_tags(Path("configs/tags.yaml")))
        choices = Recommender(songs, contexts).load_choices(session)
        ab = steering_ab(session)
    logger.info("%d answered slates", len(choices))
    if ab["n_request_slates"]:
        print(
            f"Steering A/B: you picked a steered song in {ab['steered_picked']}/"
            f"{ab['n_request_slates']} request slates = {ab['rate']:.0%} "
            f"[95% CI {ab['ci_low']:.0%}-{ab['ci_high']:.0%}]; 50% = no effect.\n"
        )
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
    results = {"n_ratings": len(choices), "steering_ab": ab, "curves": {}}
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
