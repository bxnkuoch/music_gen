"""Reference songs end to end: upload -> worker embeds -> similar songs + requests.

Fake generator + fake embedder, real database (same setup as test_requests.py).
"""

import io
from collections import Counter

import numpy as np
import pytest
import soundfile as sf
import test_requests
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_pool import FakeGenerator
from test_requests import FakeEmbedder, jobs_of

from music_gen.api.app import create_app
from music_gen.api.service import InvalidAnswer
from music_gen.db import ListenerRequest, ReferenceSong, Song, SongEmbedding
from music_gen.references import InvalidReference, save_reference, similar_songs

world = test_requests.world  # the same featurized 12-song library fixture


def wav_bytes(seconds=6.0, sr=48_000, seed=0, fmt="WAV") -> bytes:
    clip = 0.1 * np.random.default_rng(seed).standard_normal((int(seconds * sr), 2))
    buffer = io.BytesIO()
    sf.write(buffer, clip.astype(np.float32), sr, format=fmt)
    return buffer.getvalue()


def upload(session_factory, tmp_path, data, filename="song.wav") -> int:
    with session_factory() as s:
        return save_reference(s, tmp_path, filename, data).id


def test_upload_is_stored_and_queued(world):
    session_factory, *_, tmp_path = world
    reference_id = upload(session_factory, tmp_path, wav_bytes(), "My Song.wav")
    with session_factory() as s:
        reference = s.get_one(ReferenceSong, reference_id)
        assert (reference.status, reference.filename) == ("pending", "My Song.wav")
        assert reference.duration_s == pytest.approx(6.0)
        assert (tmp_path / reference.audio_path).read_bytes() == wav_bytes()
        assert similar_songs(s, reference) == []  # not embedded yet


def test_mp3_uploads_work(world):
    session_factory, *_, work, tmp_path = world
    reference_id = upload(session_factory, tmp_path, wav_bytes(fmt="MP3"), "song.mp3")
    work()
    with session_factory() as s:
        assert s.get_one(ReferenceSong, reference_id).status == "ready"


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("song.m4a", wav_bytes(), "unsupported"),
        ("song.wav", b"definitely not audio" * 100, "couldn't read"),
        ("song.wav", wav_bytes(seconds=2), "at least"),
    ],
)
def test_unusable_uploads_are_rejected(world, filename, data, message):
    session_factory, *_, tmp_path = world
    with session_factory() as s, pytest.raises(InvalidReference, match=message):
        save_reference(s, tmp_path, filename, data)
    with session_factory() as s:
        assert s.scalars(select(ReferenceSong)).all() == []
    assert not (tmp_path / "audio" / "references").exists()


def test_worker_embeds_reference_and_finds_the_closest_library_songs(world):
    session_factory, *_, work, tmp_path = world
    reference_id = upload(session_factory, tmp_path, wav_bytes())
    assert work() == 1
    with session_factory() as s:
        reference = s.get_one(ReferenceSong, reference_id)
        assert (reference.status, reference.embedding_model) == ("ready", "fake")
        nearest = similar_songs(s, reference, k=5)

        # Same distances as a brute-force cosine search over the library. (Fake songs
        # share embeddings, so the order among equal distances is arbitrary.)
        rows = s.execute(select(SongEmbedding.song_id, SongEmbedding.embedding)).all()
        target = np.asarray(reference.embedding)
        brute = {sid: 1 - float(np.asarray(e) @ target) for sid, e in rows}
        assert len(nearest) == 5
        assert [d for _, d in nearest] == pytest.approx(sorted(brute.values())[:5], abs=1e-5)
        assert all(d == pytest.approx(brute[sid], abs=1e-5) for sid, d in nearest)


def test_embedding_failure_marks_reference_failed_and_worker_carries_on(world):
    session_factory, *_, work, tmp_path = world
    reference_id = upload(session_factory, tmp_path, wav_bytes())
    work(emb=FakeEmbedder(fail=True))
    with session_factory() as s:
        reference = s.get_one(ReferenceSong, reference_id)
        assert reference.status == "failed" and "embedder crashed" in reference.error
    assert work() == 0  # a failed reference isn't retried forever


def test_request_with_reference_makes_four_reference_and_four_plain_jobs(world):
    session_factory, _, requests, work, tmp_path = world
    reference_id = upload(session_factory, tmp_path, wav_bytes())
    with session_factory() as s:
        request_id = requests.create(s, "chill lofi", "chill", reference_id=reference_id).id
        jobs = jobs_of(s, request_id)
        assert sorted(j.variant for j in jobs) == ["plain"] * 4 + ["reference"] * 4
        assert {j.prompt.text for j in jobs} == {"chill lofi"}  # no steering tags
        assert {j.reference_id for j in jobs if j.variant == "reference"} == {reference_id}
        assert {j.reference_id for j in jobs if j.variant == "plain"} == {None}
        reference_file = tmp_path / s.get_one(ReferenceSong, reference_id).audio_path

    generator = FakeGenerator()
    work(generator)
    assert Counter(r.reference_audio for r in generator.requests) == {
        None: 4,
        reference_file: 4,
    }

    with session_factory() as s:
        status = requests.status(s, request_id)
        assert status.status == "ready"
        assert status.slate.policy == "request:top2reference+top2plain"
        variants = sorted(s.get_one(Song, i.song_id).job.variant for i in status.slate.items)
        assert variants == ["plain", "plain", "reference", "reference"]


def test_request_with_unknown_reference_is_rejected(world):
    session_factory, _, requests, *_ = world
    with session_factory() as s:
        with pytest.raises(InvalidAnswer, match="reference"):
            requests.create(s, "lofi", "chill", reference_id=999)
        assert s.scalars(select(ListenerRequest)).all() == []


def test_reference_api(world):
    session_factory, recommender, requests, work, tmp_path = world
    client = TestClient(create_app(session_factory, recommender, tmp_path, requests))
    r = client.post("/api/references", params={"filename": "ref.wav"}, content=wav_bytes())
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["similar"]) == ("pending", [])
    reference_id = body["reference_id"]

    work()
    body = client.get(f"/api/references/{reference_id}").json()
    assert body["status"] == "ready" and len(body["similar"]) == 5
    assert set(body["similar"][0]) == {"song_id", "audio_url", "prompt", "distance"}
    assert body["similar"][0]["prompt"].startswith("pool song")
    audio = client.get(body["audio_url"])
    assert audio.status_code == 200 and audio.content == wav_bytes()

    r = client.post(
        "/api/requests",
        json={"text": "lofi", "context": "chill", "reference_id": reference_id},
    )
    assert r.status_code == 200
    bad = client.post("/api/references", params={"filename": "x.wav"}, content=b"nope" * 100)
    assert bad.status_code == 422
    assert client.get("/api/references/999").status_code == 404
    assert client.get("/api/references/999/audio").status_code == 404
    unknown = {"text": "lofi", "context": "chill", "reference_id": 999}
    assert client.post("/api/requests", json=unknown).status_code == 422
