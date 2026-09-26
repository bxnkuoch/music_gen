"""Live requests (Phase 4): "make me something" -> 8 candidates -> ranked slate of 4.

Per request:
  * 4 STEERED candidates: request text + 2 tags from one Thompson sample of the
    learned taste each (different samples -> different directions).
  * 4 PLAIN candidates: request text only, different seeds. The non-personalized
    baseline.
  * When all are generated, the model ranks them and the slate shows the best 2 of
    each kind, shuffled and blind. How often the listener picks a steered song
    (vs 50% by chance) measures whether personalized generation helps.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from music_gen.api.service import SERVING_MODEL, InvalidAnswer, Recommender, SlateNotFound
from music_gen.db import GenerationJob, ListenerRequest, Prompt, Slate, SlateItem
from music_gen.features.tags import TagVocabulary
from music_gen.personalization.steering import (
    RequestWords,
    build_prompt,
    choose_bpm,
    parse_request,
    steer_modifiers,
)

N_STEERED = 4
N_PLAIN = 4
SHOWN_PER_VARIANT = 2
DURATION_S = 30.0
POLICY = "request:top2steered+top2plain"
MIN_SLATE_SIZE = 2


@dataclass(frozen=True)
class RequestStatus:
    status: str
    n_done: int
    n_failed: int
    n_total: int
    slate: Slate | None
    error: str | None


class RequestService:
    def __init__(
        self,
        recommender: Recommender,
        vocab: TagVocabulary,
        words: RequestWords,
        generator_model: str,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.recommender = recommender
        self.vocab = vocab
        self.words = words
        self.generator_model = generator_model
        self.rng = rng or np.random.default_rng()

    def create(self, session: Session, text: str, context: str) -> ListenerRequest:
        try:
            parsed = parse_request(text, self.vocab, self.words)
        except ValueError as e:
            raise InvalidAnswer(str(e)) from e
        rating_session = self.recommender.start_session(session, context)
        model, _ = self.recommender.fitted_model(session)

        names = self.recommender.songs.names
        tag_columns = {n.split(":", 1)[1]: i for i, n in enumerate(names) if n.startswith("tag:")}
        d = len(names)
        plans: list[tuple[str, list[str]]] = []
        for _ in range(N_STEERED):
            w = model.posterior.sample(self.rng)[:d]  # one plausible taste
            weights = {tag: float(w[i]) for tag, i in tag_columns.items()}
            plans.append(("steered", steer_modifiers(weights, self.vocab, parsed)))
        plans += [("plain", [])] * N_PLAIN

        request = ListenerRequest(
            text=parsed.text,
            parsed={"tags": parsed.tags, "energy": parsed.energy},
            session_id=rating_session.id,
            status="generating",
        )
        session.add(request)
        session.flush()
        bpm = choose_bpm(parsed, self.words, self.rng)  # same for all: a fair comparison
        for variant, modifiers in plans:
            prompt_id = _get_or_create_prompt(
                session,
                build_prompt(parsed, modifiers),
                {"request_tags": parsed.tags, "modifiers": modifiers},
            )
            session.add(
                GenerationJob(
                    prompt_id=prompt_id,
                    seed=int(self.rng.integers(0, 2**31 - 1)),
                    duration_s=DURATION_S,
                    bpm=bpm,
                    model_name=self.generator_model,
                    status="pending",
                    attempts=0,
                    request_id=request.id,
                    variant=variant,
                )
            )
        session.commit()
        return request

    def status(self, session: Session, request_id: int) -> RequestStatus:
        request = session.get(ListenerRequest, request_id)
        if request is None:
            raise SlateNotFound(f"request {request_id} doesn't exist")
        statuses = [job.status for job in request.jobs]
        slate = session.scalars(select(Slate).where(Slate.session_id == request.session_id)).first()
        return RequestStatus(
            status=request.status,
            n_done=statuses.count("done"),
            n_failed=statuses.count("failed"),
            n_total=len(statuses),
            slate=slate if request.status == "ready" else None,
            error=request.error,
        )

    def finalize_ready(self, session: Session) -> list[int]:
        """Rank every request whose jobs have all finished. Returns their ids."""
        finished = []
        for request in session.scalars(
            select(ListenerRequest).where(ListenerRequest.status == "generating")
        ).all():
            if all(job.status in ("done", "failed") for job in request.jobs):
                self._finalize(session, request)
                finished.append(request.id)
        return finished

    def _finalize(self, session: Session, request: ListenerRequest) -> None:
        model, n_choices = self.recommender.fitted_model(session)  # also loads new songs
        songs = self.recommender.songs
        scores = model.scores(request.rating_session.context)

        ranked: dict[str, list[tuple[float, int]]] = {"steered": [], "plain": []}
        for job in request.jobs:
            if job.song is None:
                continue
            try:
                row = songs.row_of(job.song.id)
            except KeyError:
                continue  # generated but not featurized: can't rank it
            ranked[job.variant].append((float(scores[row]), job.song.id))
        picks = []
        for variant in ("steered", "plain"):
            # Random tie-break: with no ratings yet every score is 0, and the pick
            # should then be random rather than, say, always the newest song.
            candidates = ranked[variant]
            order = sorted(
                range(len(candidates)), key=lambda i: (-candidates[i][0], self.rng.random())
            )
            picks += [candidates[i] for i in order[:SHOWN_PER_VARIANT]]
        if len(picks) < MIN_SLATE_SIZE:
            request.status = "failed"
            request.error = "too few songs were generated successfully; please try again"
            session.commit()
            return

        slate = Slate(
            session_id=request.session_id,
            policy=POLICY,
            model_name=SERVING_MODEL,
            n_training_choices=n_choices,
        )
        session.add(slate)
        session.flush()
        for position, (score, song_id) in zip(self.rng.permutation(len(picks)), picks, strict=True):
            session.add(
                SlateItem(
                    slate_id=slate.id,
                    song_id=song_id,
                    position=int(position),
                    source="model",
                    score=score,
                )
            )
        request.status = "ready"
        session.commit()


def _get_or_create_prompt(session: Session, text: str, attributes: dict) -> int:
    session.execute(
        insert(Prompt)
        .values(text=text, attributes=attributes, source="user")
        .on_conflict_do_nothing(index_elements=["text"])
    )
    return session.scalar(select(Prompt.id).where(Prompt.text == text))
