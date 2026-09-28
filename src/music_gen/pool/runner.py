"""Queue generation jobs for the pool and run them, safely resumable.

Guarantees (each one has a test in tests/test_pool.py):
  * Re-running never duplicates prompts, jobs, or songs.
  * A crash mid-run loses at most the clip being generated; the next run resumes.
  * A song row exists only if its WAV file was fully written.
  * One bad job doesn't stop the run, but repeated failures in a row do
    (that usually means something systemic, like running out of memory).
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from music_gen import audio
from music_gen.db import GenerationJob, Prompt, Song
from music_gen.generation import GeneratedClip, GenerationRequest
from music_gen.pool.grid import GridConfig

logger = logging.getLogger(__name__)

POOL_AUDIO_DIR = Path("audio/pool")  # relative to settings.data_dir


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedClip: ...


@dataclass(frozen=True)
class JobOutcome:
    job_id: int
    song_id: int | None  # None if the job failed
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class RunSummary:
    generated: int = 0
    failed: int = 0
    stopped_early: bool = False


def enqueue_grid(session: Session, grid: GridConfig, model_name: str) -> int:
    """Add prompts and jobs from the grid that aren't in the database yet.

    Returns the number of new jobs.
    """
    new_jobs = 0
    for spec in grid.prompts:
        session.execute(
            insert(Prompt)
            .values(text=spec.text, attributes=spec.attributes, source="pool_grid")
            .on_conflict_do_nothing(index_elements=["text"])
        )
        prompt_id = session.scalar(select(Prompt.id).where(Prompt.text == spec.text))
        for seed in range(grid.seeds_per_prompt):
            result = session.execute(
                insert(GenerationJob)
                .values(
                    prompt_id=prompt_id,
                    seed=seed,
                    duration_s=grid.duration_s,
                    bpm=spec.bpm,
                    model_name=model_name,
                    status="pending",
                    attempts=0,
                )
                .on_conflict_do_nothing()
                # rowcount is unreliable (-1) for ON CONFLICT inserts with psycopg;
                # RETURNING gives a row only when something was actually inserted.
                .returning(GenerationJob.id)
            )
            new_jobs += result.first() is not None
    session.commit()
    return new_jobs


def job_status_counts(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(GenerationJob.status, func.count()).group_by(GenerationJob.status)
    )
    return dict(Counter({status: count for status, count in rows}))


def run_pending(
    session_factory: sessionmaker,
    generator: Generator,
    data_dir: Path,
    *,
    limit: int | None = None,
    max_attempts: int = 3,
    max_consecutive_failures: int = 5,
) -> RunSummary:
    """Generate every pending job (or at most `limit` of them)."""
    with session_factory() as session:
        requeue(session, max_attempts)
        total = session.scalar(select(func.count()).where(GenerationJob.status == "pending"))
    if limit is not None:
        total = min(total, limit)
    logger.info("%d jobs to run", total)

    summary = RunSummary()
    consecutive_failures = 0
    while limit is None or summary.generated + summary.failed < limit:
        outcome = process_next_job(session_factory, generator, data_dir)
        if outcome is None:
            break
        done = summary.generated + summary.failed + 1
        if outcome.song_id is None:
            logger.error("[%d/%d] job %d failed: %s", done, total, outcome.job_id, outcome.error)
            summary.failed += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Stopping: %d failures in a row suggests a systemic problem",
                    consecutive_failures,
                )
                summary.stopped_early = True
                break
            continue
        summary.generated += 1
        consecutive_failures = 0
        logger.info("[%d/%d] job %d done in %.1fs", done, total, outcome.job_id, outcome.elapsed_s)
    return summary


def process_next_job(
    session_factory: sessionmaker, generator: Generator, data_dir: Path
) -> JobOutcome | None:
    """Claim the next pending job, generate it, and save the song. None if no work.

    Shared by the pool runner and the live-request worker. Never raises for a bad
    job: failures are recorded on the job and returned.
    """
    claimed = _claim_next_job(session_factory, data_dir)
    if claimed is None:
        return None
    job_id, request_kwargs = claimed
    try:
        clip = generator.generate(GenerationRequest(**request_kwargs))
        rel_path = POOL_AUDIO_DIR / f"job_{job_id:06d}.wav"
        _write_atomically(clip, data_dir / rel_path)
    except Exception as e:  # noqa: BLE001 - one bad job must not kill the run
        logger.debug("job %d failed", job_id, exc_info=True)
        error = f"{type(e).__name__}: {e}"
        _mark_failed(session_factory, job_id, error)
        return JobOutcome(job_id, None, error)
    song_id = _record_song(session_factory, job_id, clip, rel_path)
    return JobOutcome(job_id, song_id, elapsed_s=clip.elapsed_s)


def requeue(session: Session, max_attempts: int) -> None:
    """Put crashed ('running') and retryable failed jobs back in the queue."""
    stale = session.execute(
        update(GenerationJob).where(GenerationJob.status == "running").values(status="pending")
    ).rowcount
    retry = session.execute(
        update(GenerationJob)
        .where(GenerationJob.status == "failed", GenerationJob.attempts < max_attempts)
        .values(status="pending")
    ).rowcount
    session.commit()
    if stale or retry:
        logger.info("Re-queued %d interrupted and %d failed jobs", stale, retry)


def _claim_next_job(session_factory: sessionmaker, data_dir: Path) -> tuple[int, dict] | None:
    with session_factory() as session:
        job = session.scalars(
            select(GenerationJob)
            .where(GenerationJob.status == "pending")
            # Live requests first (someone is waiting for them), then pool jobs.
            .order_by(GenerationJob.request_id.is_(None), GenerationJob.id)
            .limit(1)
            .with_for_update(skip_locked=True)  # safe if two runners ever overlap
        ).first()
        if job is None:
            return None
        job.status = "running"
        job.attempts += 1
        request_kwargs = {
            "prompt": job.prompt.text,
            "duration_s": job.duration_s,
            "seed": job.seed,
            "bpm": job.bpm,
            "keyscale": job.keyscale,
            "reference_audio": data_dir / job.reference.audio_path if job.reference else None,
        }
        session.commit()
        return job.id, request_kwargs


def _write_atomically(clip: GeneratedClip, path: Path) -> None:
    # Write to a temporary name, then rename: a crash can never leave a
    # half-written file under the final name.
    partial = path.with_suffix(".partial.wav")
    audio.save_wav(clip.audio, clip.sample_rate, partial)
    os.replace(partial, path)


def _record_song(
    session_factory: sessionmaker, job_id: int, clip: GeneratedClip, rel_path: Path
) -> int:
    with session_factory() as session:  # song row + status change commit together
        job = session.get(GenerationJob, job_id)
        job.status = "done"
        job.error = None
        song = Song(
            job_id=job_id,
            audio_path=str(rel_path),
            duration_s=clip.duration_s,
            sample_rate=clip.sample_rate,
            generation_time_s=clip.elapsed_s,
            loudness_dbfs=audio.rms_dbfs(clip.audio),
        )
        session.add(song)
        session.commit()
        return song.id


def _mark_failed(session_factory: sessionmaker, job_id: int, error: str) -> None:
    with session_factory() as session:
        job = session.get(GenerationJob, job_id)
        job.status = "failed"
        job.error = error[:2000]
        session.commit()
