"""Rating-loop logic: create slates, record answers, explain the learned taste.

Kept separate from the web layer (app.py) so it can be tested directly.
"""

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from music_gen.db import RatingSession, Slate, SlateItem
from music_gen.personalization.data import SongMatrix
from music_gen.personalization.models import BradleyTerryModel
from music_gen.personalization.pairs import Choice, to_pairs
from music_gen.personalization.policy import POLICY_NAME, select_slate

SERVING_MODEL = "bt_global"  # Phase 3 simulation: per-mood didn't beat global at <150 ratings


class SlateNotFound(LookupError):
    pass


class SlateAlreadyAnswered(RuntimeError):
    pass


class InvalidAnswer(ValueError):
    pass


class Recommender:
    def __init__(
        self,
        songs: SongMatrix,
        contexts: dict[str, str],
        *,
        rng: np.random.Generator | None = None,
        slate_size: int = 4,
        n_model: int = 2,
    ) -> None:
        self.songs = songs
        self.contexts = contexts
        self.rng = rng or np.random.default_rng()
        self.slate_size = slate_size
        self.n_model = n_model

    # --- reading history ----------------------------------------------------------------

    def load_choices(self, session: Session) -> list[Choice]:
        """All answered slates, oldest first, as Choices over feature-matrix rows."""
        slates = session.scalars(
            select(Slate)
            .where(Slate.answered_at.is_not(None))
            .order_by(Slate.answered_at, Slate.id)
        ).all()
        choices = []
        for slate in slates:
            try:
                shown = tuple(self.songs.row_of(item.song_id) for item in slate.items)
                chosen = self.songs.row_of(slate.chosen_song_id)
                worst = self.songs.row_of(slate.worst_song_id) if slate.worst_song_id else None
            except KeyError:
                continue  # a song lost its features (e.g. feature space changed): skip slate
            choices.append(Choice(shown, chosen, slate.session.context, worst, slate.session_id))
        return choices

    def fitted_model(self, session: Session) -> tuple[BradleyTerryModel, int]:
        choices = self.load_choices(session)
        model = BradleyTerryModel(self.songs.x, list(self.contexts), per_context=False)
        model.fit(to_pairs(choices))
        return model, len(choices)

    # --- the rating loop ------------------------------------------------------------------

    def start_session(self, session: Session, context: str) -> RatingSession:
        if context not in self.contexts:
            raise InvalidAnswer(f"unknown mood {context!r}; choose from {list(self.contexts)}")
        rating_session = RatingSession(context=context)
        session.add(rating_session)
        session.commit()
        return rating_session

    def next_slate(self, session: Session, session_id: int) -> Slate:
        rating_session = session.get(RatingSession, session_id)
        if rating_session is None:
            raise SlateNotFound(f"rating session {session_id} doesn't exist")
        pending = session.scalars(
            select(Slate).where(Slate.session_id == session_id, Slate.answered_at.is_(None))
        ).first()
        if pending is not None:
            return pending  # e.g. the page was refreshed: don't create a second one

        model, n_choices = self.fitted_model(session)
        already_shown = set(session.scalars(select(SlateItem.song_id)).all())
        exclude = {i for i, sid in enumerate(self.songs.song_ids) if sid in already_shown}
        items = select_slate(
            model,
            rating_session.context,
            self.rng,
            groups=self.songs.groups,
            exclude=exclude,
            size=self.slate_size,
            n_model=self.n_model,
        )
        slate = Slate(
            session_id=session_id,
            policy=POLICY_NAME,
            model_name=SERVING_MODEL,
            n_training_choices=n_choices,
        )
        session.add(slate)
        session.flush()
        # Shuffle the display order so "model" picks aren't always first.
        for position, item in zip(self.rng.permutation(len(items)), items, strict=True):
            session.add(
                SlateItem(
                    slate_id=slate.id,
                    song_id=int(self.songs.song_ids[item.song]),
                    position=int(position),
                    source=item.source,
                    score=item.score,
                )
            )
        session.commit()
        session.refresh(slate)
        return slate

    def record_answer(
        self, session: Session, slate_id: int, chosen_song_id: int, worst_song_id: int | None
    ) -> Slate:
        slate = session.get(Slate, slate_id, with_for_update=True)
        if slate is None:
            raise SlateNotFound(f"slate {slate_id} doesn't exist")
        if slate.answered_at is not None:
            raise SlateAlreadyAnswered(f"slate {slate_id} was already answered")
        shown = {item.song_id for item in slate.items}
        if chosen_song_id not in shown:
            raise InvalidAnswer("the chosen song wasn't in this slate")
        if worst_song_id is not None and (
            worst_song_id not in shown or worst_song_id == chosen_song_id
        ):
            raise InvalidAnswer("the least-favourite song must be a different song in this slate")
        slate.chosen_song_id = chosen_song_id
        slate.worst_song_id = worst_song_id
        slate.answered_at = datetime.now(UTC)
        session.commit()
        return slate

    # --- explanation ------------------------------------------------------------------------

    def profile(self, session: Session, top_k: int = 6) -> dict:
        """What the model has learned, in readable terms (tags + signal features only;
        the embedding's principal components have no human-readable meaning)."""
        model, n_choices = self.fitted_model(session)
        weights = model.weights()
        named = [
            (name.split(":", 1)[1], float(w))
            for name, w in zip(self.songs.names, weights, strict=True)
            if name.startswith(("tag:", "signal:"))
        ]
        named.sort(key=lambda nw: nw[1], reverse=True)
        return {
            "n_ratings": n_choices,
            "likes": [{"feature": n, "weight": w} for n, w in named[:top_k] if w > 0],
            "dislikes": [{"feature": n, "weight": w} for n, w in named[::-1][:top_k] if w < 0],
        }
