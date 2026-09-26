"""Embedding/feature storage and pgvector similarity search (needs the test database)."""

import numpy as np
import pytest

from music_gen.db import GenerationJob, Prompt, Song
from music_gen.embeddings import l2_normalize
from music_gen.features import store

DIM = 512


def add_songs(session_factory, n: int) -> list[int]:
    with session_factory() as s:
        prompt = Prompt(text="p", attributes={}, source="pool_grid")
        s.add(prompt)
        s.flush()
        ids = []
        for seed in range(n):
            job = GenerationJob(
                prompt_id=prompt.id, seed=seed, duration_s=30, model_name="m", status="done"
            )
            s.add(job)
            s.flush()
            song = Song(
                job_id=job.id,
                audio_path=f"a{seed}.wav",
                duration_s=30,
                sample_rate=48_000,
                generation_time_s=1,
                loudness_dbfs=-20,
            )
            s.add(song)
            s.flush()
            ids.append(song.id)
        s.commit()
    return ids


def unit(*values: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return l2_normalize(v)


def test_embedding_roundtrip_and_missing_query(session_factory):
    ids = add_songs(session_factory, 3)
    with session_factory() as s:
        store.save_embedding(s, ids[0], "clap", unit(1, 0))
        store.save_embedding(s, ids[1], "clap", unit(0, 1))
        store.save_embedding(s, ids[1], "muq", unit(1, 1))  # other models coexist
        s.commit()
        assert [x.id for x in store.songs_missing_embedding(s, "clap")] == [ids[2]]
        assert [x.id for x in store.songs_missing_embedding(s, "muq")] == [ids[0], ids[2]]
        loaded_ids, matrix = store.load_embeddings(s, "clap")
    assert loaded_ids == ids[:2]
    np.testing.assert_allclose(matrix, np.stack([unit(1, 0), unit(0, 1)]), atol=1e-6)


def test_saving_twice_keeps_the_first_value(session_factory):
    (song,) = add_songs(session_factory, 1)
    with session_factory() as s:
        store.save_embedding(s, song, "clap", unit(1, 0))
        store.save_embedding(s, song, "clap", unit(0, 1))
        store.save_features(s, song, "signal_v1", {"tempo_bpm": 100.0})
        store.save_features(s, song, "signal_v1", {"tempo_bpm": 999.0})
        s.commit()
        _, matrix = store.load_embeddings(s, "clap")
        _, _, values = store.load_features(s, "signal_v1")
    np.testing.assert_allclose(matrix[0], unit(1, 0), atol=1e-6)
    assert values[0, 0] == 100.0


def test_nonfinite_values_are_rejected(session_factory):
    (song,) = add_songs(session_factory, 1)
    with session_factory() as s:
        with pytest.raises(ValueError, match="NaN"):
            store.save_embedding(s, song, "clap", np.full(DIM, np.nan, dtype=np.float32))
        with pytest.raises(ValueError, match="loudness_dbfs"):
            store.save_features(s, song, "signal_v1", {"loudness_dbfs": float("-inf")})


def test_features_load_in_stable_column_order(session_factory):
    ids = add_songs(session_factory, 2)
    with session_factory() as s:
        store.save_features(s, ids[0], "signal_v1", {"b": 2.0, "a": 1.0})
        store.save_features(s, ids[1], "signal_v1", {"a": 3.0, "b": 4.0})
        s.commit()
        song_ids, names, values = store.load_features(s, "signal_v1")
    assert (song_ids, names) == (ids, ["a", "b"])
    np.testing.assert_array_equal(values, [[1, 2], [3, 4]])


def test_inconsistent_feature_keys_are_an_error(session_factory):
    ids = add_songs(session_factory, 2)
    with session_factory() as s:
        store.save_features(s, ids[0], "signal_v1", {"a": 1.0})
        store.save_features(s, ids[1], "signal_v1", {"a": 1.0, "extra": 2.0})
        s.commit()
        with pytest.raises(ValueError, match="different"):
            store.load_features(s, "signal_v1")


def test_feature_table_keeps_only_songs_with_everything(session_factory):
    ids = add_songs(session_factory, 3)
    with session_factory() as s:
        for sid in ids:
            store.save_embedding(s, sid, "clap", unit(1, sid))
            if sid != ids[1]:  # song 2 is missing its tag scores
                store.save_features(s, sid, "tags", {"melodic": 0.1 * sid})
            store.save_features(s, sid, "signal_v1", {"tempo_bpm": 90.0 + sid})
        s.commit()
        song_ids, emb, tabular, names = store.load_feature_table(
            s, "clap", {"tags": "tag", "signal_v1": "signal"}
        )
    assert song_ids == [ids[0], ids[2]]
    assert names == ["tag:melodic", "signal:tempo_bpm"]
    assert emb.shape == (2, DIM)
    np.testing.assert_allclose(tabular, [[0.1 * ids[0], 90 + ids[0]], [0.1 * ids[2], 90 + ids[2]]])


def test_nearest_songs_are_ordered_by_cosine_distance(session_factory):
    ids = add_songs(session_factory, 4)
    with session_factory() as s:
        store.save_embedding(s, ids[0], "clap", unit(1, 0))
        store.save_embedding(s, ids[1], "clap", unit(0, 1))  # 90 degrees away
        store.save_embedding(s, ids[2], "clap", unit(1, 0.1))  # almost the same
        store.save_embedding(s, ids[3], "clap", unit(-1, 0))  # opposite
        s.commit()
        result = store.nearest_songs(s, ids[0], "clap", k=3)
    assert [sid for sid, _ in result] == [ids[2], ids[1], ids[3]]
    assert ids[0] not in [sid for sid, _ in result]  # never returns itself
    distances = [d for _, d in result]
    assert distances == sorted(distances)
    assert distances[1] == pytest.approx(1.0, abs=1e-5)  # orthogonal
    assert distances[2] == pytest.approx(2.0, abs=1e-5)  # opposite


def test_nearest_songs_without_embedding_raises(session_factory):
    (song,) = add_songs(session_factory, 1)
    with session_factory() as s, pytest.raises(LookupError):
        store.nearest_songs(s, song, "clap")
