from pathlib import Path

import numpy as np
import pytest
import yaml

from music_gen.features.tags import load_tags
from music_gen.personalization.steering import (
    ParsedRequest,
    build_prompt,
    choose_bpm,
    load_request_words,
    parse_request,
    steer_modifiers,
)

ROOT = Path(__file__).parents[1]
VOCAB = load_tags(ROOT / "configs/tags.yaml")
WORDS = load_request_words(ROOT / "configs/request_words.yaml", VOCAB)


def parse(text: str) -> ParsedRequest:
    return parse_request(text, VOCAB, WORDS)


# --- parsing -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_tags",
    [
        ("chill lofi for studying", ["chill", "lo-fi hip hop"]),
        ("Chill LoFi guitar with guitar plucking", ["chill", "lo-fi guitar", "plucked guitar"]),
        ("hyperpop banger", ["hyperpop"]),  # "pop" must not match inside "hyperpop"
        ("upbeat pop", ["energetic", "pop"]),
        ("zelda style adventure music with ocarina", ["fantasy adventure video game"]),
        ("dark melodic anime", ["dark", "melodic", "anime soundtrack"]),
        ("something nice", []),
    ],
)
def test_parse_finds_tags_in_text_order(text, expected_tags):
    assert parse(text).tags == expected_tags


def test_all_config_aliases_point_to_real_tags():
    raw = yaml.safe_load((ROOT / "configs/request_words.yaml").read_text())
    assert set(raw["aliases"]) <= set(VOCAB.names)


def test_parse_normalizes_whitespace_and_rejects_empty():
    assert parse("  chill    lofi \n").text == "chill lofi"
    with pytest.raises(ValueError):
        parse("   ")


@pytest.mark.parametrize(
    "text, energy",
    [
        ("slow sleepy piano", "low"),
        ("fast workout music", "high"),
        ("jazz", None),
        ("slow but energetic", None),  # contradictory: don't guess
    ],
)
def test_energy_detection(text, energy):
    assert parse(text).energy == energy


def test_genres_helper():
    assert parse("chill lofi and some jazz").genres(VOCAB) == ["lo-fi hip hop", "jazz"]


def test_alias_to_unknown_tag_is_rejected(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"aliases": {"not a tag": ["x"]}, "energy": {}}))
    with pytest.raises(ValueError, match="isn't a tag"):
        load_request_words(path, VOCAB)


# --- steering ----------------------------------------------------------------------------


def test_steering_picks_highest_positive_steerable_tags():
    weights = {name: 0.0 for name in VOCAB.names}
    weights |= {"plucked guitar": 0.9, "warm": 0.5, "melodic": 0.3, "jazz": 5.0, "dark": -1.0}
    parsed = parse("chill lofi")
    # "jazz" is a genre: never added by steering, even with the highest weight
    assert steer_modifiers(weights, VOCAB, parsed, k=2) == ["plucked guitar", "warm"]


def test_steering_skips_tags_already_requested():
    weights = {"plucked guitar": 0.9, "warm": 0.5}
    assert steer_modifiers(weights, VOCAB, parse("lofi with plucked guitar")) == ["warm"]


def test_steering_adds_nothing_when_everything_is_disliked():
    assert steer_modifiers({"warm": -0.1, "melodic": -2.0}, VOCAB, parse("pop")) == []


def test_build_prompt_appends_modifiers():
    assert build_prompt(parse("chill lofi"), ["plucked guitar", "warm"]) == (
        "chill lofi, plucked guitar, warm"
    )
    assert build_prompt(parse("chill lofi"), []) == "chill lofi"


def test_bpm_follows_requested_energy():
    rng = np.random.default_rng(0)
    assert all(65 <= choose_bpm(parse("slow piano"), WORDS, rng) <= 85 for _ in range(20))
    assert all(120 <= choose_bpm(parse("hype edm"), WORDS, rng) <= 150 for _ in range(20))
    assert choose_bpm(parse("jazz"), WORDS, rng) is None
