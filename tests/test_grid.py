import copy
from pathlib import Path

import pytest
import yaml

from music_gen.generation import GenerationRequest
from music_gen.pool.grid import build_prompts, load_grid

CONFIG = Path(__file__).parents[1] / "configs" / "pool_grid.yaml"


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_real_config_expands_to_full_grid(raw):
    grid = load_grid(CONFIG)
    expected = len(raw["genres"]) * len(raw["moods"]) * len(raw["energy"])
    assert len(grid.prompts) == expected == 180
    assert grid.seeds_per_prompt == 3
    assert grid.duration_s == 30.0


def test_prompt_texts_are_unique(raw):
    texts = [p.text for p in build_prompts(raw)]
    assert len(set(texts)) == len(texts)


def test_grid_is_deterministic(raw):
    assert build_prompts(raw) == build_prompts(copy.deepcopy(raw))


def test_appending_a_genre_leaves_existing_prompts_unchanged(raw):
    # The grid is append-only: otherwise already-generated songs would be regenerated.
    shorter = copy.deepcopy(raw)
    last_genre = list(shorter["genres"])[-1]
    del shorter["genres"][last_genre]
    before, after = build_prompts(shorter), build_prompts(raw)
    assert after[: len(before)] == before
    assert all(p.attributes["genre"] == last_genre for p in after[len(before) :])


def test_different_random_seed_changes_choices(raw):
    other = copy.deepcopy(raw) | {"random_seed": raw["random_seed"] + 1}
    assert [p.bpm for p in build_prompts(raw)] != [p.bpm for p in build_prompts(other)]


def test_choices_respect_the_config(raw):
    for p in build_prompts(raw):
        a = p.attributes
        assert a["instrument"] in raw["genres"][a["genre"]]
        low, high = raw["energy"][a["energy"]]["bpm"]
        assert low <= p.bpm <= high
        assert a["genre"] in p.text and a["mood"] in p.text and a["instrument"] in p.text


def test_every_prompt_is_a_valid_generation_request(raw):
    for p in build_prompts(raw):
        GenerationRequest(prompt=p.text, bpm=p.bpm, duration_s=raw["duration_s"])


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.update(moods=[]), "at least one"),
        (lambda r: r.update(genres={}), "at least one"),
        (lambda r: r.update(seeds_per_prompt=0), "seeds_per_prompt"),
        (lambda r: r["genres"].update({"jazz": []}), "no instruments"),
        (lambda r: r["energy"]["low"].update(bpm=[90, 60]), "reversed"),
    ],
)
def test_invalid_configs_are_rejected(raw, mutate, match):
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        build_prompts(raw)
