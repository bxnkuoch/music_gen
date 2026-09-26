"""Turn a free-text request + the learned taste into generation prompts.

1. parse_request(): find known tags in the text ("chill lofi for studying" ->
   genre lo-fi hip hop, energy low). Plain keyword matching, no LLM.
2. steer_modifiers(): from ONE sampled taste vector, pick the descriptive tags the
   listener seems to like most and that the request doesn't already say. Each
   candidate uses a different sample (Thompson sampling), so the steered candidates
   explore different plausible directions instead of all saying the same thing.
3. build_prompt(): request text + modifiers, e.g.
   "chill lofi for studying, plucked guitar, warm".
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from music_gen.features.tags import TagVocabulary

# Genre stays the listener's call: steering only adds these kinds of tags.
STEERABLE_CATEGORIES = ("instrument", "character", "mood")


@dataclass(frozen=True)
class RequestWords:
    aliases: dict[str, str]  # phrase -> tag name
    energy_words: dict[str, list[str]]  # "low"/"high" -> words
    energy_bpm: dict[str, tuple[int, int]]


def load_request_words(path: Path, vocab: TagVocabulary) -> RequestWords:
    raw = yaml.safe_load(path.read_text())
    aliases: dict[str, str] = {}
    for tag, phrases in raw["aliases"].items():
        if tag not in vocab.names:
            raise ValueError(f"alias target {tag!r} isn't a tag in the vocabulary")
        for phrase in phrases:
            phrase = str(phrase).lower()
            if phrase in aliases and aliases[phrase] != tag:
                raise ValueError(f"phrase {phrase!r} maps to two tags")
            aliases[phrase] = tag
    return RequestWords(
        aliases=aliases,
        energy_words={k: [w.lower() for w in v["words"]] for k, v in raw["energy"].items()},
        energy_bpm={k: tuple(v["bpm"]) for k, v in raw["energy"].items()},
    )


@dataclass(frozen=True)
class ParsedRequest:
    text: str
    tags: list[str] = field(default_factory=list)  # tag names found, in text order
    energy: str | None = None  # "low" | "high" | None

    def genres(self, vocab: TagVocabulary) -> list[str]:
        genre_names = {vocab.names[i] for i in vocab.indices("genre")}
        return [t for t in self.tags if t in genre_names]


def parse_request(text: str, vocab: TagVocabulary, words: RequestWords) -> ParsedRequest:
    text = " ".join(text.split())
    if not text:
        raise ValueError("request text must not be empty")
    lowered = text.lower()

    phrases = {name.lower(): name for name in vocab.names} | words.aliases
    found: list[tuple[int, str]] = []
    taken = [False] * len(lowered)
    for phrase in sorted(phrases, key=len, reverse=True):  # longest phrase wins
        for m in re.finditer(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", lowered):
            if not any(taken[m.start() : m.end()]):
                taken[m.start() : m.end()] = [True] * (m.end() - m.start())
                found.append((m.start(), phrases[phrase]))
    tags = list(dict.fromkeys(tag for _, tag in sorted(found)))  # text order, no repeats

    tokens = set(re.findall(r"[\w-]+", lowered))
    hits = {level: len(tokens & set(w)) for level, w in words.energy_words.items()}
    energy = None
    if any(hits.values()):
        best = max(hits.values())
        winners = [level for level, n in hits.items() if n == best]
        energy = winners[0] if len(winners) == 1 else None  # contradictory: don't guess
    return ParsedRequest(text=text, tags=tags, energy=energy)


def steer_modifiers(
    tag_weights: dict[str, float],
    vocab: TagVocabulary,
    parsed: ParsedRequest,
    k: int = 2,
) -> list[str]:
    """The k steerable tags with the highest (sampled) taste weight.

    tag_weights: tag name -> weight from one sampled taste vector.
    Tags already in the request are skipped, and only positive weights are used:
    if the model thinks you dislike everything left, it adds nothing.
    """
    allowed = {vocab.names[i] for c in STEERABLE_CATEGORIES for i in vocab.indices(c)}
    options = [
        (w, tag)
        for tag, w in tag_weights.items()
        if tag in allowed and tag not in parsed.tags and w > 0
    ]
    return [tag for _, tag in sorted(options, reverse=True)[:k]]


def build_prompt(parsed: ParsedRequest, modifiers: list[str]) -> str:
    return ", ".join([parsed.text, *modifiers])


def choose_bpm(parsed: ParsedRequest, words: RequestWords, rng: np.random.Generator) -> int | None:
    """A BPM in the request's energy range, or None to let the generator decide."""
    if parsed.energy is None:
        return None
    low, high = words.energy_bpm[parsed.energy]
    return int(rng.integers(low, high + 1))
