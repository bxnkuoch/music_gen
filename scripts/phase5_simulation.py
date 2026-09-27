"""Phase 5 experiment: which slate policy balances exploration and exploitation best?

Each simulated listener rates 150 slates of the REAL pool chosen by one policy (the
model refits after every slate; unseen songs first, as in the app). Same listeners
and population setup as Phase 3. Per policy, averaged over listeners (95% CI):

  * experience: true-taste percentile of the BEST song in each slate (100% = the
    listener's favourite song was on offer), over slates 1-30 (cold start) and 31-60.
    Not over all 150: with unseen songs first, 150 slates show nearly the whole
    540-song pool whatever the policy, so a long-run average can't tell them apart.
  * learning: pairwise accuracy on a held-out set of RANDOM slates the listener also
    rated (identical across policies, so data collected by different policies is
    judged on the same test), and top-pick quality at the end

Usage:  uv run python scripts/phase5_simulation.py [--users 30]
Writes: data/results/phase5_simulation.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from phase3_simulation import hidden_features, mean_ci

from music_gen.config import get_settings
from music_gen.db import make_session_factory
from music_gen.features import store
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.logging_setup import configure_logging
from music_gen.personalization.data import load_contexts, load_song_matrix
from music_gen.personalization.evaluation import pair_correctness
from music_gen.personalization.models import BradleyTerryModel
from music_gen.personalization.pairs import to_pairs
from music_gen.personalization.policy import POLICIES
from music_gen.personalization.simulation import (
    SongMeta,
    make_user,
    simulate_choices,
    top_pick_quality,
)

logger = logging.getLogger("phase5_simulation")
N_SLATES = 150
N_TEST_SLATES = 100
WINDOWS = [(0, 30), (30, 60)]
BASELINE = "thompson+random"  # served in Phase 3
CHECKPOINTS = [10, 30, 60, 150]


def percentiles(values: np.ndarray) -> np.ndarray:
    """Rank of each value among the others, 0 = worst, 1 = best."""
    return np.argsort(np.argsort(values)) / (len(values) - 1)


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
        songs = songs.subset(np.array([sid in muq for sid in songs.song_ids]))
        meta = SongMeta(
            songs.genres, songs.energies, hidden_features(muq, songs.song_ids), songs.groups
        )
    x = songs.x

    def new_model() -> BradleyTerryModel:
        return BradleyTerryModel(x, contexts, per_context=False)

    results: dict = {"n_songs": len(x), "n_slates": N_SLATES, "users": args.users}
    for population in ("fixed_taste", "mood_dependent_taste"):
        stats = {
            name: {"experience": {w: [] for w in WINDOWS}, "top_pick": [], "acc": {}}
            for name in POLICIES
        }
        for u in range(args.users):
            seed = [args.seed, u, population == "fixed_taste"]
            rng = np.random.default_rng(seed)
            user = make_user(rng, meta, contexts, mood_dependent=population != "fixed_taste")
            test_pairs = to_pairs(simulate_choices(user, meta, contexts, N_TEST_SLATES, rng))
            pct = {c: percentiles(user.utility(meta, c)) for c in contexts}
            for name, policy in POLICIES.items():
                model = new_model()
                choices = simulate_choices(
                    user,
                    meta,
                    contexts,
                    N_SLATES,
                    np.random.default_rng(seed + [1]),  # same draws for every policy
                    model=model,
                    policy=policy,
                    features=x,
                    exclude_seen=True,
                )
                best = [pct[c.context][list(c.shown)].max() for c in choices]
                s = stats[name]
                for lo, hi in WINDOWS:
                    s["experience"][lo, hi].append(float(np.mean(best[lo:hi])))
                s["top_pick"].append(top_pick_quality(model, user, meta, contexts))
                for n in CHECKPOINTS:
                    m = new_model()
                    m.fit(to_pairs(choices[:n]))
                    s["acc"].setdefault(n, []).append(pair_correctness(m, test_pairs).mean())
            logger.info("%s: user %d/%d done", population, u + 1, args.users)

        results[population] = {
            name: {
                "best_in_slate_percentile": {
                    f"slates_{lo + 1}-{hi}": mean_ci(v) for (lo, hi), v in s["experience"].items()
                },
                f"top_pick_quality_after_{N_SLATES}": mean_ci(s["top_pick"]),
                "heldout_accuracy": {str(n): mean_ci(v) for n, v in s["acc"].items()},
            }
            for name, s in stats.items()
        }
        print(f"\n{population}")
        print(f"{'policy':>24} | best 1-30 | best 31-60 | top pick | acc@30 | acc@150")
        for name, r in results[population].items():
            print(
                f"{name:>24} | "
                f"{r['best_in_slate_percentile']['slates_1-30']['mean']:9.3f} | "
                f"{r['best_in_slate_percentile']['slates_31-60']['mean']:10.3f} | "
                f"{r[f'top_pick_quality_after_{N_SLATES}']['mean']:8.3f} | "
                f"{r['heldout_accuracy']['30']['mean']:6.3f} | "
                f"{r['heldout_accuracy']['150']['mean']:7.3f}"
            )

        # Paired: same listeners under every policy, so compare per listener.
        base = stats[BASELINE]
        paired = {}
        for name, s in stats.items():
            paired[name] = {
                "best_in_slate_1-30": mean_ci(
                    np.subtract(s["experience"][WINDOWS[0]], base["experience"][WINDOWS[0]])
                ),
                "top_pick": mean_ci(np.subtract(s["top_pick"], base["top_pick"])),
                "heldout_accuracy_150": mean_ci(np.subtract(s["acc"][150], base["acc"][150])),
            }
        results[population][f"paired_difference_vs_{BASELINE}"] = paired
        print(f"\nDifference vs {BASELINE} (paired over listeners, 95% CI)")
        for name, d in paired.items():
            print(
                f"{name:>24} | "
                + " | ".join(
                    f"{k} {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
                    for k, v in d.items()
                )
            )

    out = settings.results_dir / "phase5_simulation.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
