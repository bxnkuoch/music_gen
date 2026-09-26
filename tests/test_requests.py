"""Live requests end to end: request -> 8 jobs -> worker -> ranked slate.

Fake generator + fake embedder, real database, real feature-space math.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_pool import FakeGenerator

from music_gen.api.app import create_app
from music_gen.api.requests import RequestService
from music_gen.api.service import InvalidAnswer, Recommender, SlateNotFound
from music_gen.db import GenerationJob, ListenerRequest, Prompt, Song, SongEmbedding
from music_gen.embeddings import l2_normalize
from music_gen.features import signal, store
from music_gen.features.pipeline import featurize_song
from music_gen.features.space import FeatureSpace
from music_gen.features.tags import Tag, TagVocabulary
from music_gen.personalization.data import DbSongSource
from music_gen.personalization.steering import RequestWords
from music_gen.pool.grid import GridConfig, PromptSpec
from music_gen.pool.runner import enqueue_grid, run_pending
from music_gen.worker import work_once

VOCAB = TagVocabulary(
    version=1,
    tags=[
        Tag("lofi", "genre", "lofi music"),
        Tag("jazz", "genre", "jazz music"),
        Tag("chill", "mood", "chill music"),
        Tag("plucked guitar", "instrument", "plucked guitar"),
        Tag("warm", "character", "warm music"),
        Tag("glitchy", "character", "glitchy music"),
    ],
)
WORDS = RequestWords(
    aliases={"lo fi": "lofi"}, energy_words={"low": ["chill"]}, energy_bpm={"low": (65, 85)}
)
CONTEXTS = {"chill": "relaxing", "hype": "energetic"}


class FakeEmbedder:
    """Deterministic 512-d "embeddings" derived from the audio (the DB column is 512-d)."""

    name = "fake"
    dim = 512

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def embed_audio(self, audio_array, sample_rate):
        if self.fail:
            raise RuntimeError("embedder crashed")
        seed = int(abs(audio_array[..., :100].sum()) * 1e6) % 2**32
        return l2_normalize(np.random.default_rng(seed).normal(size=self.dim))

    def embed_text(self, texts):
        return l2_normalize(np.random.default_rng(len(texts)).normal(size=(len(texts), self.dim)))


@pytest.fixture
def world(session_factory, tmp_path):
    """A featurized 12-song library, a fitted feature space, and the services."""
    embedder = FakeEmbedder()
    text_vecs = embedder.embed_text(VOCAB.texts)
    grid = GridConfig(
        prompts=[PromptSpec(f"pool song {i}", 90, {"genre": "lofi"}) for i in range(6)],
        seeds_per_prompt=2,
        duration_s=30.0,
    )
    with session_factory() as s:
        enqueue_grid(s, grid, "fake-model")
    run_pending(session_factory, FakeGenerator(), tmp_path)
    with session_factory() as s:
        for song in s.scalars(select(Song)).all():
            featurize_song(s, song, tmp_path, embedder, VOCAB, text_vecs)
        s.commit()
        _, emb, tabular, names = store.load_feature_table(
            s, "fake", {VOCAB.feature_set("fake"): "tag", signal.FEATURE_SET: "signal"}
        )
    space = FeatureSpace.fit(
        emb, tabular, names, n_components=3, version="t", embedding_model="fake"
    )
    source = DbSongSource(space, VOCAB)
    with session_factory() as s:
        recommender = Recommender(
            source.load(s), CONTEXTS, source=source, rng=np.random.default_rng(0)
        )
    requests = RequestService(recommender, VOCAB, WORDS, "fake-model", rng=np.random.default_rng(1))

    def work(generator=None, emb=None) -> int:
        """Run the worker until idle; returns the number of steps that did work."""
        steps = 0
        while work_once(
            session_factory,
            generator or FakeGenerator(),
            tmp_path,
            emb or embedder,
            VOCAB,
            text_vecs,
            requests,
        ):
            steps += 1
        return steps

    return session_factory, recommender, requests, work, tmp_path


def jobs_of(session, request_id):
    return session.scalars(
        select(GenerationJob).where(GenerationJob.request_id == request_id)
    ).all()


def test_request_creates_four_steered_and_four_plain_jobs(world):
    session_factory, _, requests, *_ = world
    with session_factory() as s:
        request = requests.create(s, "chill lo fi please", "chill")
        assert request.parsed == {"tags": ["chill", "lofi"], "energy": "low"}
        jobs = jobs_of(s, request.id)
        assert sorted(j.variant for j in jobs) == ["plain"] * 4 + ["steered"] * 4
        assert all(j.status == "pending" and 65 <= j.bpm <= 85 for j in jobs)
        assert len({j.seed for j in jobs}) == 8
        plain_texts = {j.prompt.text for j in jobs if j.variant == "plain"}
        assert plain_texts == {"chill lo fi please"}
        for job in jobs:
            if job.variant == "steered":
                modifiers = job.prompt.attributes["modifiers"]
                assert len(modifiers) <= 2
                # never re-adds requested tags, never adds a genre
                assert not set(modifiers) & {"chill", "lofi", "jazz"}
                assert job.prompt.source == "user"


def rate_library(session_factory, recommender, n=3):
    """Answer n library slates (always the first song), so the model has a taste."""
    with session_factory() as s:
        sid = recommender.start_session(s, "chill").id
        for _ in range(n):
            slate = recommender.next_slate(s, sid)
            recommender.record_answer(s, slate.id, slate.items[0].song_id, None)


def test_worker_generates_featurizes_and_ranks_into_a_slate(world):
    session_factory, recommender, requests, work, _ = world
    rate_library(session_factory, recommender)  # else every score is 0 and ranking is moot
    with session_factory() as s:
        request_id = requests.create(s, "chill lofi", "chill").id
        assert requests.status(s, request_id).status == "generating"
    assert work() == 8
    with session_factory() as s:
        status = requests.status(s, request_id)
        assert (status.status, status.n_done, status.n_failed, status.n_total) == ("ready", 8, 0, 8)
        items = status.slate.items
        variants = sorted(s.get(Song, i.song_id).job.variant for i in items)
        assert variants == ["plain", "plain", "steered", "steered"]
        assert sorted(i.position for i in items) == [0, 1, 2, 3]
        assert status.slate.n_training_choices == 3
        # every generated song was featurized (8 new + 12 library)
        assert len(s.scalars(select(SongEmbedding)).all()) == 20
        # the slate shows the model's top 2 of each variant
        model, _ = recommender.fitted_model(s)
        scores = model.scores("chill")
        for variant in ("steered", "plain"):
            candidates = [j.song.id for j in jobs_of(s, request_id) if j.variant == variant]
            ranked = sorted(candidates, key=lambda sid: -scores[recommender.songs.row_of(sid)])
            shown = {i.song_id for i in items if s.get(Song, i.song_id).job.variant == variant}
            assert shown == set(ranked[:2])


def test_answering_a_request_slate_feeds_the_model(world):
    session_factory, recommender, requests, work, _ = world
    with session_factory() as s:
        request_id = requests.create(s, "lofi", "chill").id
    work()
    with session_factory() as s:
        slate = requests.status(s, request_id).slate
        recommender.record_answer(s, slate.id, slate.items[0].song_id, None)
        choices = recommender.load_choices(s)
    assert len(choices) == 1 and choices[0].context == "chill"


def test_request_jobs_jump_ahead_of_pool_jobs(world):
    session_factory, _, requests, *_ = world
    with session_factory() as s:
        enqueue_grid(
            s,
            GridConfig([PromptSpec("late pool song", 90, {})], seeds_per_prompt=1, duration_s=30),
            "fake-model",
        )
        request_id = requests.create(s, "lofi", "chill").id
    gen = FakeGenerator()
    run_pending(world[0], gen, world[4], limit=8)
    assert all(r.prompt != "late pool song" for r in gen.requests)
    with session_factory() as s:
        assert all(j.status == "done" for j in jobs_of(s, request_id))


def test_failed_steered_generations_still_give_a_slate(world):
    session_factory, _, requests, work, _ = world
    with session_factory() as s:
        request_id = requests.create(s, "lofi", "chill").id
        plain_prompt = "lofi"
    work(FakeGenerator(lambda r: r.prompt != plain_prompt))  # every steered one fails
    with session_factory() as s:
        status = requests.status(s, request_id)
        assert (status.status, status.n_done, status.n_failed) == ("ready", 4, 4)
        assert len(status.slate.items) == 2


def test_request_fails_cleanly_when_nothing_generates(world):
    session_factory, _, requests, work, _ = world
    with session_factory() as s:
        request_id = requests.create(s, "lofi", "chill").id
    work(FakeGenerator(lambda r: True))
    with session_factory() as s:
        status = requests.status(s, request_id)
        assert status.status == "failed" and "try again" in status.error
        assert status.slate is None


def test_featurizing_failure_leaves_songs_unranked_but_request_finishes(world):
    session_factory, _, requests, work, _ = world
    with session_factory() as s:
        request_id = requests.create(s, "lofi", "chill").id
    work(emb=FakeEmbedder(fail=True))
    with session_factory() as s:
        status = requests.status(s, request_id)
        assert status.status == "failed"  # generated, but nothing could be ranked
        assert status.n_done == 8


def test_invalid_requests(world):
    session_factory, _, requests, *_ = world
    with session_factory() as s:
        with pytest.raises(InvalidAnswer):
            requests.create(s, "   ", "chill")
        with pytest.raises(InvalidAnswer):
            requests.create(s, "lofi", "angry")
        with pytest.raises(SlateNotFound):
            requests.status(s, 999)
        assert s.scalars(select(ListenerRequest)).all() == []  # nothing half-created


def test_same_request_twice_reuses_prompt_rows(world):
    session_factory, _, requests, *_ = world
    with session_factory() as s:
        requests.create(s, "lofi", "chill")
        requests.create(s, "lofi", "hype")
        assert len(s.scalars(select(Prompt).where(Prompt.text == "lofi")).all()) == 1


def test_request_api(world):
    session_factory, recommender, requests, work, tmp_path = world
    client = TestClient(create_app(session_factory, recommender, tmp_path, requests))
    r = client.post("/api/requests", json={"text": "chill lofi", "context": "chill"})
    assert r.status_code == 200
    body = r.json()
    assert body["tags"] == ["chill", "lofi"] and body["energy"] == "low"
    status = client.get(f"/api/requests/{body['request_id']}").json()
    assert (status["status"], status["n_done"], status["slate"]) == ("generating", 0, None)
    work()
    status = client.get(f"/api/requests/{body['request_id']}").json()
    assert status["status"] == "ready" and len(status["slate"]["songs"]) == 4
    assert all(set(song) == {"song_id", "audio_url"} for song in status["slate"]["songs"])
    assert client.post("/api/requests", json={"text": "", "context": "chill"}).status_code == 422
    assert client.get("/api/requests/999").status_code == 404
    no_requests = TestClient(create_app(session_factory, recommender, tmp_path))
    assert (
        no_requests.post("/api/requests", json={"text": "x", "context": "chill"}).status_code == 503
    )
