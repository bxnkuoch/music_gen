"""Tests for the preference-learning core (pure math; no database or models)."""

import numpy as np
import pytest

from music_gen.personalization import bradley_terry
from music_gen.personalization.evaluation import bootstrap_ci, learning_curve, pair_correctness
from music_gen.personalization.models import BradleyTerryModel, MeanLikedModel, RandomModel
from music_gen.personalization.pairs import Choice, Pair, choice_to_pairs, to_pairs
from music_gen.personalization.policy import POLICIES, SlatePolicy, select_slate
from music_gen.personalization.simulation import (
    SongMeta,
    make_user,
    simulate_choices,
    top_pick_quality,
)

# --- pairs --------------------------------------------------------------------------------


def test_favourite_beats_everyone_else():
    pairs = choice_to_pairs(Choice((1, 2, 3, 4), chosen=3, context="chill"))
    assert {(p.winner, p.loser) for p in pairs} == {(3, 1), (3, 2), (3, 4)}
    assert all(p.context == "chill" for p in pairs)


def test_least_favourite_adds_pairs_without_duplicates():
    pairs = choice_to_pairs(Choice((1, 2, 3, 4), chosen=3, context="c", worst=1))
    assert {(p.winner, p.loser) for p in pairs} == {(3, 1), (3, 2), (3, 4), (2, 1), (4, 1)}
    assert len(pairs) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shown": (1, 2), "chosen": 9},
        {"shown": (1, 2, 3), "chosen": 1, "worst": 1},
        {"shown": (1, 2, 3), "chosen": 1, "worst": 7},
        {"shown": (1, 1, 2), "chosen": 1},
        {"shown": (1,), "chosen": 1},
    ],
)
def test_invalid_choices_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Choice(context="c", **kwargs)


# --- Bradley-Terry math -----------------------------------------------------------------------


def synthetic_pairs(w_true, n, rng, d=None):
    x = rng.normal(size=(n, 2, len(w_true)))
    diffs = x[:, 0] - x[:, 1]
    first_wins = rng.random(n) < 1 / (1 + np.exp(-diffs @ w_true))
    return np.where(first_wins[:, None], diffs, -diffs)


def test_no_data_returns_the_prior():
    prior_mean, prior_var = np.array([0.5, -1.0]), np.array([2.0, 3.0])
    post = bradley_terry.fit(np.zeros((0, 2)), prior_mean, prior_var)
    np.testing.assert_allclose(post.mean, prior_mean)
    np.testing.assert_allclose(post.cov, np.diag(prior_var))


def test_recovers_true_weights_from_many_pairs():
    rng = np.random.default_rng(0)
    w_true = np.array([1.5, -1.0, 0.0])
    post = bradley_terry.fit(synthetic_pairs(w_true, 4000, rng), np.zeros(3), np.full(3, 10.0))
    np.testing.assert_allclose(post.mean, w_true, atol=0.15)


def test_uncertainty_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    w = np.array([1.0, 1.0])
    few = bradley_terry.fit(synthetic_pairs(w, 20, rng), np.zeros(2), np.ones(2))
    many = bradley_terry.fit(synthetic_pairs(w, 2000, rng), np.zeros(2), np.ones(2))
    assert np.trace(many.cov) < np.trace(few.cov) / 10


def test_perfectly_separable_data_stays_finite():
    # One feature always decides: an unregularized fit would diverge to infinity.
    diffs = np.array([[1.0, 0.0]] * 50)
    post = bradley_terry.fit(diffs, np.zeros(2), np.ones(2))
    assert np.isfinite(post.mean).all() and post.mean[0] > 1
    assert post.mean[1] == pytest.approx(0, abs=1e-9)


def test_prob_prefers_is_symmetric_and_shrinks_under_uncertainty():
    confident = bradley_terry.Posterior(np.array([2.0]), np.array([[1e-6]]))
    unsure = bradley_terry.Posterior(np.array([2.0]), np.array([[10.0]]))
    a, b = np.array([1.0]), np.array([0.0])
    p = confident.prob_prefers(a, b)[0]
    assert p + confident.prob_prefers(b, a)[0] == pytest.approx(1.0)
    assert 0.5 < unsure.prob_prefers(a, b)[0] < p


@pytest.mark.parametrize(
    "diffs, prior_var",
    [
        (np.zeros((3, 4)), np.ones(2)),  # wrong width
        (np.zeros((3, 2)), np.array([1.0, 0.0])),  # non-positive variance
        (np.full((3, 2), np.nan), np.ones(2)),
    ],
)
def test_fit_rejects_bad_inputs(diffs, prior_var):
    with pytest.raises(ValueError):
        bradley_terry.fit(diffs, np.zeros(2), prior_var)


# --- models ------------------------------------------------------------------------------


@pytest.fixture
def songs():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 5))


def pairs_for(w, x, contexts, n, rng):
    """Noise-free pairs: the song with higher x @ w[context] wins."""
    out = []
    for _ in range(n):
        a, b = rng.choice(len(x), 2, replace=False)
        c = contexts[rng.integers(len(contexts))]
        wc = w[c]
        out.append(Pair(a, b, c) if x[a] @ wc > x[b] @ wc else Pair(b, a, c))
    return out


def test_bt_global_learns_a_consistent_taste(songs):
    rng = np.random.default_rng(1)
    w = np.array([1.0, -1.0, 0.5, 0, 0])
    model = BradleyTerryModel(songs, ["a", "b"], per_context=False, prior_var=1.0)
    model.fit(pairs_for({"a": w, "b": w}, songs, ["a", "b"], 300, rng))
    test = pairs_for({"a": w, "b": w}, songs, ["a", "b"], 300, rng)
    assert pair_correctness(model, test).mean() > 0.9
    assert np.corrcoef(model.weights(), w)[0, 1] > 0.95


def test_bt_per_mood_learns_opposite_tastes_that_global_cannot(songs):
    rng = np.random.default_rng(2)
    w = np.array([1.0, 0, 0, 0, 0])
    tastes = {"chill": w, "hype": -w}  # one feature: liked in one mood, disliked in the other
    train = pairs_for(tastes, songs, ["chill", "hype"], 400, rng)
    test = pairs_for(tastes, songs, ["chill", "hype"], 400, rng)
    per_mood = BradleyTerryModel(songs, ["chill", "hype"], per_context=True, prior_var=1.0)
    global_ = BradleyTerryModel(songs, ["chill", "hype"], per_context=False, prior_var=1.0)
    per_mood.fit(train)
    global_.fit(train)
    assert pair_correctness(per_mood, test).mean() > 0.85
    assert pair_correctness(global_, test).mean() < 0.65
    assert per_mood.weights("chill")[0] > 0 > per_mood.weights("hype")[0]


def test_unknown_context_is_an_error(songs):
    model = BradleyTerryModel(songs, ["a"], per_context=True)
    with pytest.raises(ValueError, match="unknown context"):
        model.scores("nope")


def test_unfitted_bt_is_neutral(songs):
    model = BradleyTerryModel(songs, ["a"], per_context=False)
    assert model.prob_prefers(0, 1, "a") == pytest.approx(0.5)
    np.testing.assert_array_equal(model.scores("a"), 0)


def test_baselines(songs):
    assert RandomModel(40).prob_prefers(0, 1, "a") == 0.5
    liked = MeanLikedModel(songs)
    liked.fit([Pair(3, 5, "a"), Pair(3, 6, "a")])
    np.testing.assert_allclose(liked.profile, songs[3])
    assert np.argmax(liked.scores("a")) == 3 or liked.scores("a")[3] > 0


# --- policy ------------------------------------------------------------------------------

PHASE3 = POLICIES["thompson+random"]


def test_slate_has_distinct_prompts_and_both_sources(songs):
    groups = np.repeat(np.arange(20), 2)  # pairs of songs share a prompt
    items = select_slate(
        RandomModel(40), "a", np.random.default_rng(0), policy=PHASE3, groups=groups
    )
    assert len(items) == 4
    assert len({groups[i.song] for i in items}) == 4
    assert [i.source for i in items] == ["model", "model", "random", "random"]


def test_model_slots_take_the_highest_scored_songs(songs):
    model = BradleyTerryModel(songs, ["a"], per_context=False)
    model.posterior = bradley_terry.Posterior(np.array([1.0, 0, 0, 0, 0]), np.eye(5) * 1e-12)
    top = np.argsort(-songs[:, 0])
    for policy, n_top in [(PHASE3, 2), (POLICIES["greedy"], 4), (POLICIES["thompson"], 4)]:
        items = select_slate(
            model, "a", np.random.default_rng(0), policy=policy, groups=np.arange(40)
        )
        assert {i.song for i in items if i.source != "random"} == set(top[:n_top])


class SequenceModel:
    """Every Thompson draw returns the next score vector in a fixed list."""

    def __init__(self, mean, draws):
        self.mean, self.draws = mean, iter(draws)

    def scores(self, context):
        return self.mean

    def sampled_scores(self, context, rng):
        return next(self.draws, self.mean)


def test_each_thompson_slot_uses_a_fresh_sample_after_the_exploit_slots():
    mean = np.array([9.0, 8, 7, 0, 0, 0, 0, 0])
    draws = [np.eye(8)[k] for k in (0, 3, 5)]  # 1st draw favours song 0 (already taken)
    policy = SlatePolicy("t", exploit=2, thompson=2)
    items = select_slate(
        SequenceModel(mean, draws),
        "a",
        np.random.default_rng(0),
        policy=policy,
        groups=np.arange(8),
    )
    assert [(i.song, i.source) for i in items] == [
        (0, "exploit"),
        (1, "exploit"),
        (2, "model"),
        (3, "model"),
    ]  # 1st draw: 0 and 1 taken, all others tie -> first free song (2); 2nd draw -> song 3


def test_diversity_penalty_avoids_near_duplicates():
    # Songs 0-2 point the same way and score highest; song 3 is different, slightly lower.
    features = np.array([[1.0, 0], [1.0, 0.01], [1.0, 0.02], [0, 1.0], [-1, 0], [0, -1]])
    mean = np.array([1.0, 0.99, 0.98, 0.9, -1, -1])
    model = SequenceModel(mean, [])
    rng = np.random.default_rng(0)
    groups = np.arange(6)
    plain = SlatePolicy("g", exploit=2)
    diverse = SlatePolicy("g:div", exploit=2, diversity=1.0)
    assert [i.song for i in select_slate(model, "a", rng, policy=plain, groups=groups)] == [0, 1]
    items = select_slate(model, "a", rng, policy=diverse, groups=groups, features=features)
    assert [i.song for i in items] == [0, 3]
    with pytest.raises(ValueError, match="features"):
        select_slate(model, "a", rng, policy=diverse, groups=groups)


def test_every_preset_fills_a_valid_slate(songs):
    model = BradleyTerryModel(songs, ["a"], per_context=False)
    groups = np.repeat(np.arange(20), 2)
    for policy in POLICIES.values():
        assert len(policy.name) <= 32  # database column width
        items = select_slate(
            model, "a", np.random.default_rng(0), policy=policy, groups=groups, features=songs
        )
        assert len(items) == policy.size == 4
        assert len({groups[i.song] for i in items}) == 4


def test_excluded_songs_are_avoided_until_exhausted(songs):
    groups = np.arange(40)
    rng = np.random.default_rng(0)
    items = select_slate(
        RandomModel(40), "a", rng, policy=PHASE3, groups=groups, exclude=set(range(36))
    )
    assert {i.song for i in items} == {36, 37, 38, 39}
    # Only 2 songs left unseen: start over rather than fail.
    items = select_slate(
        RandomModel(40), "a", rng, policy=PHASE3, groups=groups, exclude=set(range(38))
    )
    assert len(items) == 4
    # Every song already shown (happens after ~137 slates in the app): start over too.
    items = select_slate(
        RandomModel(40), "a", rng, policy=PHASE3, groups=groups, exclude=set(range(40))
    )
    assert len(items) == 4


def test_slate_needs_enough_distinct_prompts():
    with pytest.raises(ValueError, match="distinct groups"):
        select_slate(
            RandomModel(6),
            "a",
            np.random.default_rng(0),
            policy=PHASE3,
            groups=np.array([0, 0, 1, 1, 2, 2]),
        )


# --- evaluation ---------------------------------------------------------------------------


def test_pair_correctness_scores_ties_as_half(songs):
    assert pair_correctness(RandomModel(40), [Pair(0, 1, "a")]).tolist() == [0.5]


def test_bootstrap_ci_contains_mean_and_resamples_groups():
    rng = np.random.default_rng(0)
    values = np.array([1.0] * 10 + [0.0] * 10)
    low, high = bootstrap_ci(values, np.repeat([0, 1], 10), rng)  # only 2 sessions
    assert low == 0.0 and high == 1.0  # whole sessions resampled: very wide interval
    low, high = bootstrap_ci(values, np.arange(20), rng)  # 20 independent items
    assert 0.2 < low < 0.5 < high < 0.8  # SE of the mean is about 0.11


def test_learning_curve_trains_on_past_and_stops_when_test_set_is_small(songs):
    choices = [Choice((0, 1, 2, 3), 0, "a", session=i // 5) for i in range(30)]
    curve = learning_curve(
        lambda: BradleyTerryModel(songs, ["a"], per_context=False, prior_var=1.0),
        choices,
        checkpoints=[0, 10, 20, 26],
        rng=np.random.default_rng(0),
    )
    assert [p.n_train_choices for p in curve] == [0, 10, 20]  # 26 leaves < 5 test choices
    assert curve[0].accuracy == 0.5  # untrained model: all ties
    assert curve[1].accuracy == 1.0  # song 0 always wins
    assert curve[1].n_test_pairs == 20 * 3


# --- simulation --------------------------------------------------------------------------


@pytest.fixture
def meta():
    rng = np.random.default_rng(0)
    n = 120
    return SongMeta(
        genres=np.array(["pop", "metal", "jazz", "lofi"] * 30),
        energies=np.array(["low", "high", "medium"] * 40),
        hidden=rng.normal(size=(n, 4)),
        groups=np.arange(n) // 2,
    )


def test_simulated_user_is_mood_dependent_only_when_asked(meta):
    rng = np.random.default_rng(0)
    fixed = make_user(rng, meta, ["chill", "hype"], mood_dependent=False)
    moody = make_user(rng, meta, ["chill", "hype"], mood_dependent=True)
    np.testing.assert_array_equal(fixed.utility(meta, "chill"), fixed.utility(meta, "hype"))
    assert not np.allclose(moody.utility(meta, "chill"), moody.utility(meta, "hype"))


def test_simulated_choices_are_valid_and_follow_taste(meta):
    rng = np.random.default_rng(0)
    user = make_user(rng, meta, ["chill"], mood_dependent=False, noise=0.01)
    choices = simulate_choices(user, meta, ["chill"], 30, rng)
    assert len(choices) == 30
    u = user.utility(meta, "chill")
    for c in choices:  # almost no noise: favourite really is the best shown
        assert c.chosen == max(c.shown, key=lambda s: u[s])
        assert c.worst == min(c.shown, key=lambda s: u[s])


def test_simulated_policy_slates_show_unseen_songs_first(meta):
    rng = np.random.default_rng(0)
    user = make_user(rng, meta, ["chill"], mood_dependent=False)
    x = np.hstack([meta.hidden, rng.normal(size=(len(meta.groups), 2))])
    model = BradleyTerryModel(x, ["chill"], per_context=False)
    choices = simulate_choices(
        user,
        meta,
        ["chill"],
        15,
        rng,
        model=model,
        policy=POLICIES["exploit3+thompson1:div"],
        features=x,
        exclude_seen=True,
    )
    shown = [s for c in choices for s in c.shown]
    assert len(shown) == len(set(shown)) == 60  # 15 slates x 4, no repeats
    with pytest.raises(ValueError, match="needs a model"):
        simulate_choices(user, meta, ["chill"], 1, rng, policy=POLICIES["greedy"])


def test_model_learns_simulated_taste_end_to_end(meta):
    """When the taste is fully visible in the features, BT finds (near-)favourite songs.

    Averaged over 10 users: a single user may like two genres almost equally, so
    second-best is a fair pick for them.
    """
    genres, energies = np.unique(meta.genres), np.unique(meta.energies)
    x = np.hstack(
        [(meta.genres[:, None] == genres).astype(float), (meta.energies[:, None] == energies)]
    )
    quality = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        user = make_user(rng, meta, ["chill"], mood_dependent=False, quirk_scale=0.0)
        model = BradleyTerryModel(x, ["chill"], per_context=False, prior_var=1.0)
        model.fit(to_pairs(simulate_choices(user, meta, ["chill"], 200, rng)))
        quality.append(top_pick_quality(model, user, meta, ["chill"]))
    assert np.mean(quality) > 0.93


def test_top_pick_quality_counts_ties_as_half(meta):
    user = make_user(np.random.default_rng(0), meta, ["chill"], mood_dependent=False, quirk_scale=0)
    flat = RandomModel(len(meta.groups))  # scores all 0 -> picks song 0
    u = user.utility(meta, "chill")
    expected = ((u < u[0]).sum() + 0.5 * (np.isclose(u, u[0]).sum() - 1)) / (len(u) - 1)
    assert top_pick_quality(flat, user, meta, ["chill"]) == pytest.approx(expected)


def test_wilson_interval():
    from music_gen.personalization.evaluation import wilson_interval

    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.002) and high == pytest.approx(0.596, abs=0.002)
    assert wilson_interval(0, 0) == (0.0, 1.0)  # no data: anything is possible
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.35  # never negative, unlike the textbook formula
    assert wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
