"""Phase 2 validation: how much can we trust each feature?

Every pool song was generated from known attributes (genre, mood, energy, BPM), so
we can check features against that ground truth:
  1. BPM: does ACE-Step follow the requested tempo, and does librosa measure it?
  2. Zero-shot tags: does each embedding recognise the genre / mood it was asked for?
  3. Similar-song search: do a song's nearest neighbours share its genre?
     (with and without mean-centering; excluding same-prompt siblings, which are trivially similar)
  4. Linear probes: is genre / mood / energy *linearly* recoverable from each feature
     block? This is what matters for Phase 3, whose preference model is linear.
     Cross-validated with whole prompts held out, so seed siblings can't leak.

Usage:  uv run python scripts/phase2_validation.py
Writes: data/results/phase2_features.json
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from music_gen.config import get_settings
from music_gen.db import GenerationJob, Prompt, Song, make_session_factory
from music_gen.embeddings import l2_normalize
from music_gen.features import signal, store
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import load_tags
from music_gen.logging_setup import configure_logging

logger = logging.getLogger("phase2_validation")
ENERGY_LEVELS = {"low": 0, "medium": 1, "high": 2}


def ground_truth(session) -> dict[int, dict]:
    """Known attributes of every POOL song (live-request songs have no labels)."""
    rows = session.execute(
        select(Song.id, Prompt.id, Prompt.attributes, GenerationJob.bpm)
        .join(GenerationJob, Song.job_id == GenerationJob.id)
        .join(Prompt, GenerationJob.prompt_id == Prompt.id)
        .where(Prompt.source == "pool_grid")
    ).all()
    return {sid: {"prompt_id": pid, "bpm": bpm, **attrs} for sid, pid, attrs, bpm in rows}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        return np.argsort(np.argsort(a))

    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def bpm_adherence(truth: dict[int, dict], ids: list[int], names: list[str], values) -> dict:
    measured = values[:, names.index("tempo_bpm")]
    requested = np.array([truth[s]["bpm"] for s in ids], dtype=float)
    ratio = measured / requested
    exact = np.abs(ratio - 1) <= 0.04
    octave = ~exact & ((np.abs(ratio - 2) <= 0.08) | (np.abs(ratio - 0.5) <= 0.02))
    by_genre: dict[str, list[bool]] = {}
    for sid, ok in zip(ids, exact | octave, strict=True):
        by_genre.setdefault(truth[sid]["genre"], []).append(bool(ok))
    return {
        "n": len(ids),
        "within_4pct": float(exact.mean()),
        "octave_error": float(octave.mean()),
        "miss": float((~exact & ~octave).mean()),
        "spearman_requested_vs_measured": spearman(requested, measured),
        "hit_or_octave_by_genre": {g: float(np.mean(v)) for g, v in sorted(by_genre.items())},
    }


def zero_shot(truth, ids, scores: np.ndarray, vocab, category: str, labels: list[str]) -> dict:
    """Pick the best-matching label per song, raw and calibrated.

    Calibrated: standardize each label's scores across songs first. Some label texts
    sit closer to *all* audio and would otherwise win by default. Uses no labels.
    """
    cols = [vocab.names.index(label) for label in labels]
    raw = scores[:, cols]
    calibrated = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    true = np.array([labels.index(truth[s][category]) for s in ids])
    result = {"chance_top1": 1 / len(labels)}
    for name, sub in (("raw", raw), ("calibrated", calibrated)):
        order = np.argsort(-sub, axis=1)
        top1 = order[:, 0] == true
        confusions = Counter(
            (labels[t], labels[p]) for t, p in zip(true, order[:, 0], strict=True) if t != p
        )
        result[name] = {
            "top1": float(top1.mean()),
            "top3": float((order[:, :3] == true[:, None]).any(axis=1).mean()),
            "top1_by_label": {
                label: float(top1[true == i].mean())
                for i, label in enumerate(labels)
                if (true == i).any()
            },
            "most_confused": [f"{t} -> {p} ({n})" for (t, p), n in confusions.most_common(5)],
        }
    return result


def neighbour_precision(truth, ids, emb: np.ndarray, k: int = 5) -> float:
    """Share of each song's k nearest neighbours (other prompts only) with its genre."""
    sims = emb @ emb.T
    prompts = np.array([truth[s]["prompt_id"] for s in ids])
    genres = np.array([truth[s]["genre"] for s in ids])
    sims[prompts[:, None] == prompts[None, :]] = -np.inf  # drop self + same-prompt siblings
    nearest = np.argsort(-sims, axis=1)[:, :k]
    return float((genres[nearest] == genres[:, None]).mean())


def linear_probe(x: np.ndarray, labels: list[str], groups: list[int]) -> float:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    scores = cross_val_score(model, x, labels, groups=groups, cv=GroupKFold(n_splits=5))
    return float(scores.mean())


def probes(truth, session, settings, vocab) -> dict:
    """Probe each feature block on the songs that have all of them."""
    clap = f"clap:{settings.clap_model_id}"
    ids, clap_emb, tabular, names = store.load_feature_table(
        session, clap, {vocab.feature_set(clap): "tag", signal.FEATURE_SET: "signal"}
    )
    muq_ids, muq_emb = store.load_embeddings(session, f"muq-mulan:{settings.muq_model_id}")
    muq_lookup = dict(zip(muq_ids, muq_emb, strict=True))
    keep = [i for i, sid in enumerate(ids) if sid in muq_lookup and sid in truth]
    ids, clap_emb, tabular = [ids[i] for i in keep], clap_emb[keep], tabular[keep]
    muq_emb = np.stack([muq_lookup[sid] for sid in ids])

    def pcs(emb: np.ndarray, k: int = 32) -> np.ndarray:
        centered = emb - emb.mean(axis=0)
        return centered @ np.linalg.svd(centered, full_matrices=False)[2][:k].T

    tag_cols = [i for i, n in enumerate(names) if n.startswith("tag:")]
    sig_cols = [i for i, n in enumerate(names) if n.startswith("signal:")]
    blocks = {
        "clap_pcs (32)": pcs(clap_emb),
        "muq_pcs (32)": pcs(muq_emb),
        f"clap_tags ({len(tag_cols)})": tabular[:, tag_cols],
        f"signal ({len(sig_cols)})": tabular[:, sig_cols],
        f"full feature space ({32 + len(names)})": np.hstack([pcs(clap_emb), tabular]),
    }
    groups = [truth[s]["prompt_id"] for s in ids]
    targets = {t: [truth[s][t] for s in ids] for t in ("genre", "mood", "energy")}
    full = blocks[f"full feature space ({32 + len(names)})"]
    predicted = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)),
        full,
        targets["genre"],
        groups=groups,
        cv=GroupKFold(n_splits=5),
    )
    true_genres = np.array(targets["genre"])
    genre_recall = {
        g: float((predicted[true_genres == g] == g).mean()) for g in sorted(set(true_genres))
    }
    return {
        "full_space_genre_recall": genre_recall,
        "n_songs": len(ids),
        "chance": {t: max(Counter(v).values()) / len(v) for t, v in targets.items()},
        "accuracy": {
            block: {t: linear_probe(x, y, groups) for t, y in targets.items()}
            for block, x in blocks.items()
        },
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    vocab = load_tags(Path("configs/tags.yaml"))
    grid_genres = [vocab.names[i] for i in vocab.indices("genre")]
    grid_moods = ["chill", "dark", "uplifting", "melancholic"]
    results: dict = {}

    with make_session_factory(settings.database_url)() as session:
        truth = ground_truth(session)
        sig_ids, sig_names, sig_values = store.load_features(session, signal.FEATURE_SET)
        pool = [i for i, sid in enumerate(sig_ids) if sid in truth]
        sig_ids, sig_values = [sig_ids[i] for i in pool], sig_values[pool]
        results["bpm"] = bpm_adherence(truth, sig_ids, sig_names, sig_values)

        results["embeddings"] = {}
        for model in _embedding_models(settings):
            ids, emb = store.load_embeddings(session, model)
            if not ids:
                logger.warning("No %s embeddings; run compute_features.py", model)
                continue
            tag_ids, tag_names, tag_scores = store.load_features(session, vocab.feature_set(model))
            assert tag_ids == ids, "tags and embeddings cover different songs"
            pool = [i for i, sid in enumerate(ids) if sid in truth]
            ids, emb, tag_scores = [ids[i] for i in pool], emb[pool], tag_scores[pool]
            # load_features sorts names; reorder columns to vocabulary order
            tag_scores = tag_scores[:, [tag_names.index(n) for n in vocab.names]]
            energy = np.array([ENERGY_LEVELS[truth[s]["energy"]] for s in ids])
            results["embeddings"][model] = {
                "n_songs": len(ids),
                "genre": zero_shot(truth, ids, tag_scores, vocab, "genre", grid_genres),
                "mood": zero_shot(truth, ids, tag_scores, vocab, "mood", grid_moods),
                "energetic_tag_vs_energy_spearman": spearman(
                    tag_scores[:, vocab.names.index("energetic")], energy
                ),
                "genre_precision_at_5": {
                    "raw": neighbour_precision(truth, ids, emb),
                    "mean_centered": neighbour_precision(
                        truth, ids, l2_normalize(emb - emb.mean(axis=0))
                    ),
                    "chance": _genre_chance(truth, ids),
                },
            }

        results["linear_probes"] = probes(truth, session, settings, vocab)

    space_path = settings.data_dir / "models" / "feature_space_v1.npz"
    if space_path.exists():
        space = FeatureSpace.load(space_path)
        cumulative = np.cumsum(space.explained_variance_ratio)
        results["feature_space"] = {
            "n_features": len(space.names),
            "variance_kept_by_8_16_32_pcs": [float(cumulative[i - 1]) for i in (8, 16, 32)],
        }

    out = settings.results_dir / "phase2_features.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)


def _embedding_models(settings) -> list[str]:
    return [f"clap:{settings.clap_model_id}", f"muq-mulan:{settings.muq_model_id}"]


def _genre_chance(truth, ids) -> float:
    counts = Counter(truth[s]["genre"] for s in ids)
    n = len(ids)
    return float(sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)))


if __name__ == "__main__":
    main()
