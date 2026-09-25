"""Pool runner tests: resumability, failure handling, and no duplicates.

Uses a fake generator (instant, configurable failures) and the test database.
"""

from collections.abc import Callable

import numpy as np
import pytest
from sqlalchemy import select, update

from music_gen import audio
from music_gen.db import GenerationJob, Prompt, Song
from music_gen.generation import GeneratedClip, GenerationError, GenerationRequest
from music_gen.pool import runner
from music_gen.pool.grid import GridConfig, PromptSpec
from music_gen.pool.runner import enqueue_grid, job_status_counts, run_pending


class FakeGenerator:
    """Returns 0.5 s of noise instantly. `fail_when(request)` -> raise GenerationError."""

    def __init__(self, fail_when: Callable[[GenerationRequest], bool] = lambda r: False):
        self.fail_when = fail_when
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GeneratedClip:
        self.requests.append(request)
        if self.fail_when(request):
            raise GenerationError("model produced near-silent audio")
        rng = np.random.default_rng(request.seed)
        clip = (0.1 * rng.standard_normal((2, 24_000))).astype(np.float32)
        return GeneratedClip(request, clip, 48_000, elapsed_s=0.01)


def small_grid(genres=("jazz", "house"), seeds=2) -> GridConfig:
    prompts = [PromptSpec(f"chill {g}", 90, {"genre": g, "mood": "chill"}) for g in genres]
    return GridConfig(prompts=prompts, seeds_per_prompt=seeds, duration_s=30.0)


@pytest.fixture
def queued(session_factory):
    """A database with 2 prompts x 2 seeds = 4 pending jobs."""
    with session_factory() as s:
        enqueue_grid(s, small_grid(), "fake-model")
    return session_factory


def statuses(session_factory) -> dict[str, int]:
    with session_factory() as s:
        return job_status_counts(s)


def song_count(session_factory) -> int:
    with session_factory() as s:
        return len(s.scalars(select(Song)).all())


# --- enqueueing --------------------------------------------------------------------


def test_enqueue_creates_prompts_and_jobs(session_factory):
    with session_factory() as s:
        assert enqueue_grid(s, small_grid(), "fake-model") == 4
        assert len(s.scalars(select(Prompt)).all()) == 2
        jobs = s.scalars(select(GenerationJob)).all()
    assert sorted(j.seed for j in jobs) == [0, 0, 1, 1]
    assert {j.bpm for j in jobs} == {90}


def test_enqueue_twice_adds_nothing(queued):
    with queued() as s:
        assert enqueue_grid(s, small_grid(), "fake-model") == 0
    assert statuses(queued) == {"pending": 4}


def test_growing_the_grid_only_adds_new_jobs(queued):
    with queued() as s:
        assert enqueue_grid(s, small_grid(genres=("jazz", "house", "reggae")), "fake-model") == 2


def test_a_new_model_gets_its_own_jobs(queued):
    with queued() as s:
        assert enqueue_grid(s, small_grid(), "other-model") == 4


# --- running ------------------------------------------------------------------------


def test_run_generates_every_job_and_writes_files(queued, tmp_path):
    gen = FakeGenerator()
    summary = run_pending(queued, gen, tmp_path)

    assert (summary.generated, summary.failed, summary.stopped_early) == (4, 0, False)
    assert statuses(queued) == {"done": 4}
    with queued() as s:
        songs = s.scalars(select(Song)).all()
        for song in songs:
            assert not song.audio_path.startswith("/")  # stored relative to data_dir
            clip, sr = audio.load_audio(tmp_path / song.audio_path)
            assert sr == 48_000 and clip.shape == (2, 24_000)
            assert song.job.status == "done"
    assert not list(tmp_path.rglob("*.partial.wav"))


def test_requests_carry_the_job_settings(queued, tmp_path):
    gen = FakeGenerator()
    run_pending(queued, gen, tmp_path)
    assert {(r.prompt, r.seed, r.bpm, r.duration_s) for r in gen.requests} == {
        (p, seed, 90, 30.0) for p in ("chill jazz", "chill house") for seed in (0, 1)
    }


def test_second_run_does_nothing(queued, tmp_path):
    run_pending(queued, FakeGenerator(), tmp_path)
    gen = FakeGenerator()
    summary = run_pending(queued, gen, tmp_path)
    assert summary.generated == 0 and gen.requests == []
    assert song_count(queued) == 4


def test_limit_caps_jobs_per_run(queued, tmp_path):
    assert run_pending(queued, FakeGenerator(), tmp_path, limit=3).generated == 3
    assert statuses(queued) == {"done": 3, "pending": 1}


def test_empty_queue_is_fine(session_factory, tmp_path):
    summary = run_pending(session_factory, FakeGenerator(), tmp_path)
    assert (summary.generated, summary.failed) == (0, 0)


# --- failures -----------------------------------------------------------------------


def fails_for_jazz_seed_1(r: GenerationRequest) -> bool:
    return r.prompt == "chill jazz" and r.seed == 1


def test_one_failure_does_not_stop_the_run(queued, tmp_path):
    summary = run_pending(queued, FakeGenerator(fails_for_jazz_seed_1), tmp_path)
    assert (summary.generated, summary.failed) == (3, 1)
    with queued() as s:
        failed = s.scalars(select(GenerationJob).where(GenerationJob.status == "failed")).one()
        assert failed.attempts == 1
        assert "GenerationError" in failed.error and "silent" in failed.error
        assert failed.song is None


def test_failed_jobs_are_retried_next_run(queued, tmp_path):
    run_pending(queued, FakeGenerator(fails_for_jazz_seed_1), tmp_path)
    summary = run_pending(queued, FakeGenerator(), tmp_path)  # problem went away
    assert summary.generated == 1
    assert statuses(queued) == {"done": 4}
    with queued() as s:
        job = s.scalars(select(GenerationJob).where(GenerationJob.attempts == 2)).one()
        assert job.error is None


def test_retries_stop_after_max_attempts(queued, tmp_path):
    gen = FakeGenerator(fails_for_jazz_seed_1)
    for _ in range(4):
        run_pending(queued, gen, tmp_path, max_attempts=3)
    jazz_seed_1 = [r for r in gen.requests if fails_for_jazz_seed_1(r)]
    assert len(jazz_seed_1) == 3  # the 4th run left it alone
    assert statuses(queued) == {"done": 3, "failed": 1}


def test_repeated_failures_stop_the_run_early(queued, tmp_path):
    summary = run_pending(
        queued, FakeGenerator(lambda r: True), tmp_path, max_consecutive_failures=2
    )
    assert summary.stopped_early and summary.failed == 2
    assert statuses(queued) == {"failed": 2, "pending": 2}


def test_invalid_job_settings_fail_that_job_only(queued, tmp_path):
    with queued() as s:  # e.g. someone edited the database by hand
        s.execute(update(GenerationJob).where(GenerationJob.id == 1).values(bpm=999))
        s.commit()
    summary = run_pending(queued, FakeGenerator(), tmp_path)
    assert (summary.generated, summary.failed) == (3, 1)
    with queued() as s:
        assert "bpm must be in" in s.get(GenerationJob, 1).error


def test_crash_mid_run_resumes_without_duplicates(queued, tmp_path):
    class Crash(BaseException):  # like Ctrl+C: not caught by the runner
        pass

    def crash_on_third(r: GenerationRequest) -> bool:
        if len(gen.requests) == 3:
            raise Crash
        return False

    gen = FakeGenerator(crash_on_third)
    with pytest.raises(Crash):
        run_pending(queued, gen, tmp_path)
    assert statuses(queued) == {"done": 2, "running": 1, "pending": 1}
    assert song_count(queued) == 2

    summary = run_pending(queued, FakeGenerator(), tmp_path)
    assert summary.generated == 2
    assert statuses(queued) == {"done": 4}
    assert song_count(queued) == 4


def test_crash_while_writing_leaves_no_song_and_no_final_file(queued, tmp_path, monkeypatch):
    real_save = audio.save_wav

    def save_then_fail(clip_audio, sr, path):
        real_save(clip_audio, sr, path)  # the .partial.wav gets written...
        raise OSError("disk full")  # ...but we "crash" before the rename

    monkeypatch.setattr(runner.audio, "save_wav", save_then_fail)
    summary = run_pending(queued, FakeGenerator(), tmp_path, limit=1)
    assert summary.failed == 1
    assert song_count(queued) == 0
    assert not (tmp_path / "audio/pool/job_000001.wav").exists()
    with queued() as s:
        assert "disk full" in s.get(GenerationJob, 1).error
