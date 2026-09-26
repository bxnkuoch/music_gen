"""Turn slate feedback ("I liked this one best / that one least") into pairs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    """The listener's answer to one slate. Song ids are feature-matrix row indices
    in the personalization code, database ids elsewhere; only equality matters here."""

    shown: tuple[int, ...]
    chosen: int
    context: str
    worst: int | None = None
    session: int = 0  # groups choices for bootstrap confidence intervals

    def __post_init__(self) -> None:
        if len(set(self.shown)) != len(self.shown):
            raise ValueError("a slate can't show the same song twice")
        if len(self.shown) < 2:
            raise ValueError("a slate needs at least 2 songs")
        if self.chosen not in self.shown:
            raise ValueError("chosen song wasn't in the slate")
        if self.worst is not None and (self.worst not in self.shown or self.worst == self.chosen):
            raise ValueError("worst song must be in the slate and differ from the chosen one")


@dataclass(frozen=True)
class Pair:
    winner: int
    loser: int
    context: str


def choice_to_pairs(choice: Choice) -> list[Pair]:
    """Favourite beats every other song; least-favourite loses to every other song.

    A 4-song slate gives 3 pairs, or 5 with a least-favourite (never the same pair twice).
    """
    pairs = [Pair(choice.chosen, s, choice.context) for s in choice.shown if s != choice.chosen]
    if choice.worst is not None:
        pairs += [
            Pair(s, choice.worst, choice.context)
            for s in choice.shown
            if s not in (choice.chosen, choice.worst)
        ]
    return pairs


def to_pairs(choices: list[Choice]) -> list[Pair]:
    return [p for c in choices for p in choice_to_pairs(c)]
