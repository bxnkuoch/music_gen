"""The rating API end to end: HTTP -> service -> test database, with tiny fake songs."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select

from music_gen import audio
from music_gen.api.app import create_app
from music_gen.api.service import Recommender
from music_gen.db import GenerationJob, Prompt, Slate, SlateItem, Song
from music_gen.personalization.data import SongMatrix

CONTEXTS = {"chill": "relaxing", "hype": "energetic"}
N_SONGS = 12


@pytest.fixture
def setup(session_factory, tmp_path):
    """12 songs (6 prompts x 2 seeds) with quiet/loud WAVs and random features."""
    rng = np.random.default_rng(0)
    song_ids, groups = [], []
    with session_factory() as s:
        for p in range(6):
            prompt = Prompt(text=f"prompt {p}", attributes={"genre": f"g{p}"}, source="pool_grid")
            s.add(prompt)
            s.flush()
            for seed in range(2):
                job = GenerationJob(
                    prompt_id=prompt.id, seed=seed, duration_s=30, model_name="m", status="done"
                )
                s.add(job)
                s.flush()
                path = f"audio/song_{p}_{seed}.wav"
                amplitude = 0.01 if seed == 0 else 0.5  # very different loudness
                audio.save_wav(
                    (amplitude * rng.standard_normal((2, 4800))).astype(np.float32),
                    48_000,
                    tmp_path / path,
                )
                song = Song(
                    job_id=job.id,
                    audio_path=path,
                    duration_s=0.1,
                    sample_rate=48_000,
                    generation_time_s=1,
                    loudness_dbfs=-20,
                )
                s.add(song)
                s.flush()
                song_ids.append(song.id)
                groups.append(prompt.id)
        s.commit()
    songs = SongMatrix(
        song_ids=np.array(song_ids),
        x=rng.normal(size=(N_SONGS, 3)),
        names=["pc1", "tag:melodic", "signal:tempo_bpm"],
        groups=np.array(groups),
        genres=np.array(["g"] * N_SONGS),
        energies=np.array(["low"] * N_SONGS),
    )
    recommender = Recommender(songs, CONTEXTS, rng=np.random.default_rng(1))
    client = TestClient(create_app(session_factory, recommender, tmp_path))
    return client, session_factory, song_ids


def start(client, context="chill") -> int:
    r = client.post("/api/sessions", json={"context": context})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_contexts_are_listed(setup):
    client, *_ = setup
    assert client.get("/api/contexts").json() == CONTEXTS


def test_unknown_mood_is_rejected(setup):
    client, *_ = setup
    assert client.post("/api/sessions", json={"context": "angry"}).status_code == 422


def test_slate_has_four_distinct_prompt_songs_and_hides_its_strategy(setup):
    client, session_factory, _ = setup
    body = client.post(f"/api/sessions/{start(client)}/slates").json()
    assert len(body["songs"]) == 4
    assert all(set(song) == {"song_id", "audio_url"} for song in body["songs"])  # blind
    with session_factory() as s:
        items = s.scalars(select(SlateItem).where(SlateItem.slate_id == body["slate_id"])).all()
        songs = {sg.id: sg for sg in s.scalars(select(Song)).all()}
        prompts = {songs[i.song_id].job.prompt_id for i in items}
    assert len(prompts) == 4
    assert sorted(i.position for i in items) == [0, 1, 2, 3]
    assert sorted(i.source for i in items) == ["model", "model", "random", "random"]


def test_refreshing_returns_the_same_unanswered_slate(setup):
    client, *_ = setup
    sid = start(client)
    first = client.post(f"/api/sessions/{sid}/slates").json()
    assert client.post(f"/api/sessions/{sid}/slates").json() == first


def test_answering_records_choice_and_next_slate_avoids_repeats(setup):
    client, session_factory, _ = setup
    sid = start(client)
    first = client.post(f"/api/sessions/{sid}/slates").json()
    shown = [s["song_id"] for s in first["songs"]]
    r = client.post(
        f"/api/slates/{first['slate_id']}/answer",
        json={"chosen_song_id": shown[2], "worst_song_id": shown[0]},
    )
    assert r.status_code == 200 and r.json() == {"n_ratings": 1}
    second = client.post(f"/api/sessions/{sid}/slates").json()
    assert not set(shown) & {s["song_id"] for s in second["songs"]}
    with session_factory() as s:
        slate = s.get(Slate, first["slate_id"])
        assert (slate.chosen_song_id, slate.worst_song_id) == (shown[2], shown[0])
        assert slate.answered_at is not None
        assert s.get(Slate, second["slate_id"]).n_training_choices == 1


@pytest.mark.parametrize(
    "answer, status",
    [
        (lambda shown, other: {"chosen_song_id": other}, 422),  # not in slate
        (lambda shown, other: {"chosen_song_id": shown[0], "worst_song_id": shown[0]}, 422),
        (lambda shown, other: {"chosen_song_id": shown[0], "worst_song_id": other}, 422),
    ],
)
def test_invalid_answers_are_rejected(setup, answer, status):
    client, _, song_ids = setup
    slate = client.post(f"/api/sessions/{start(client)}/slates").json()
    shown = [s["song_id"] for s in slate["songs"]]
    other = next(i for i in song_ids if i not in shown)
    r = client.post(f"/api/slates/{slate['slate_id']}/answer", json=answer(shown, other))
    assert r.status_code == status


def test_double_answer_and_missing_things(setup):
    client, *_ = setup
    slate = client.post(f"/api/sessions/{start(client)}/slates").json()
    body = {"chosen_song_id": slate["songs"][0]["song_id"]}
    assert client.post(f"/api/slates/{slate['slate_id']}/answer", json=body).status_code == 200
    assert client.post(f"/api/slates/{slate['slate_id']}/answer", json=body).status_code == 409
    assert client.post("/api/slates/9999/answer", json=body).status_code == 404
    assert client.post("/api/sessions/9999/slates").status_code == 404
    assert client.get("/api/songs/9999/audio").status_code == 404


def test_audio_is_served_at_equal_loudness(setup):
    client, _, song_ids = setup
    loudness = []
    for song_id in song_ids[:2]:  # a quiet and a loud original
        r = client.get(f"/api/songs/{song_id}/audio")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
        clip, _ = sf.read(io.BytesIO(r.content), dtype="float32")
        loudness.append(audio.rms_dbfs(clip))
    assert loudness[0] == pytest.approx(loudness[1], abs=0.5)


def test_profile_and_stats_reflect_answers(setup):
    client, *_ = setup
    assert client.get("/api/profile").json()["n_ratings"] == 0
    for context in ("chill", "hype", "hype"):
        sid = start(client, context)
        slate = client.post(f"/api/sessions/{sid}/slates").json()
        client.post(
            f"/api/slates/{slate['slate_id']}/answer",
            json={"chosen_song_id": slate["songs"][0]["song_id"]},
        )
    assert client.get("/api/stats").json() == {
        "n_ratings": 3,
        "by_context": {"chill": 1, "hype": 2},
    }
    profile = client.get("/api/profile").json()
    assert profile["n_ratings"] == 3
    names = {f["feature"] for f in profile["likes"] + profile["dislikes"]}
    assert names <= {"melodic", "tempo_bpm"}  # principal components are never shown
