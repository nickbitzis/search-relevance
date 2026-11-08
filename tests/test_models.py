import math

import numpy as np
import pytest

from searchrel.clickmodel import ATTRACT_BY_GRADE, ctr_labels, examination, fit_pbm, simulate_clicks
from searchrel.interleave import ab_sample_size, credit, interleaving_sample_size, sign_test_p, team_draft
from searchrel.ir import dcg, ndcg, recall_at, reciprocal_rank
from searchrel.ltr import Ranker, rank_order


def test_dcg_hand_computed():
    # (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4) = 7 + 0 + 1.5
    assert dcg([3, 0, 2]) == pytest.approx(8.5)


def test_ndcg_perfect_and_hand_value():
    assert ndcg([3, 2, 1], [1, 2, 3]) == pytest.approx(1.0)
    ideal = 7 + 3 / math.log2(3)  # [3, 2]
    assert ndcg([2, 3], [3, 2]) == pytest.approx((3 + 7 / math.log2(3)) / ideal)


def test_ndcg_ideal_uses_all_known_grades():
    # Retrieving only one of two relevant products cannot score 1.0.
    assert ndcg([3], [3, 3]) < 1.0
    assert ndcg([], [3]) == 0.0 and ndcg([0, 0], []) == 0.0


def test_mrr_and_recall():
    assert reciprocal_rank([0, 1, 2, 3]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 1]) == 0.0
    assert recall_at([3, 0, 2, 0], n_relevant=4, k=50) == 0.5
    assert recall_at([3, 2], n_relevant=100, k=2) == 1.0  # denominator capped at k
    assert recall_at([3], n_relevant=0) == 0.0


def _tiny_ltr_data(rng, n_q=60, per=20):
    X, y, groups = [], [], []
    for _ in range(n_q):
        rel = rng.random(per)
        noise = rng.random((per, 2))
        X.append(np.column_stack([rel + rng.normal(0, 0.05, per), noise]))
        y.append(np.clip(np.floor(rel * 4), 0, 3))
        groups.append(per)
    return np.vstack(X), np.concatenate(y), groups


@pytest.mark.parametrize("backend", ["sklearn"])
def test_ltr_learns_the_signal_on_a_tiny_set(backend):
    rng = np.random.default_rng(0)
    X, y, groups = _tiny_ltr_data(rng)
    m = Ranker(backend=backend, n_estimators=60).fit(X, y, groups)
    Xt, yt, _ = _tiny_ltr_data(np.random.default_rng(1), n_q=1)
    order = rank_order(m.predict(Xt))
    assert ndcg(yt[order], yt) > 0.9
    c = m.contributions(Xt)
    assert c.shape == Xt.shape
    assert np.abs(c[:, 0]).mean() > 3 * np.abs(c[:, 1:]).mean()  # the real feature explains the score
    assert np.allclose(c.sum(axis=1) + m.bias(), m.predict(Xt))  # contributions add up to the score


def test_rank_order_is_stable():
    assert rank_order(np.array([1.0, 2.0, 2.0, 0.5])).tolist() == [1, 2, 0, 3]
    assert rank_order(np.array([2.001, 2.004, 1.0])).tolist() == [0, 1, 2]  # near ties keep BM25 order


def test_pbm_recovers_known_position_bias():
    rng = np.random.default_rng(42)
    theta = examination(10, power=1.0)
    n_q, docs_per = 40, 12
    grades = rng.integers(0, 4, size=(n_q, docs_per))
    sessions = []
    for _ in range(12000):
        q = int(rng.integers(n_q))
        order = rng.permutation(docs_per)[:10]  # random placement gives every doc every position
        sessions.append((q, order, grades[q, order]))
    log = simulate_clicks(rng, sessions, theta)
    fit = fit_pbm(log, iters=80, prior=0.0)
    assert np.allclose(fit.theta, theta, atol=0.07)
    est = np.array([[fit.alpha.get((q, d), np.nan) for d in range(docs_per)] for q in range(n_q)])
    for g in range(4):
        assert abs(np.nanmean(est[grades == g]) - ATTRACT_BY_GRADE[g]) < 0.08


def test_raw_ctr_is_position_biased_and_pbm_is_not():
    rng = np.random.default_rng(7)
    theta = examination(10)
    sessions = []
    for _ in range(6000):
        top = rng.random() < 0.8  # doc 0 shown first most of the time, doc 1 last
        order = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 1]) if top else np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        sessions.append((0, order, np.full(10, 2)))
    log = simulate_clicks(rng, sessions, theta)
    ctr = ctr_labels(log)
    fit = fit_pbm(log, prior=0.0)
    assert ctr[(0, 0)] > 2 * ctr[(0, 1)]  # same true grade, very different CTR
    assert abs(fit.alpha[(0, 0)] - fit.alpha[(0, 1)]) < 0.1


def test_team_draft_has_no_duplicates_and_balanced_teams():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = list(rng.permutation(15)[:10])
        b = list(rng.permutation(15)[:10])
        shown, team = team_draft(a, b, rng)
        assert len(shown) == len(set(shown)) == 10
        assert abs(team.count("A") - team.count("B")) <= 1


def test_team_draft_identical_rankings_keep_the_order():
    rng = np.random.default_rng(2)
    shown, team = team_draft([1, 2, 3, 4], [1, 2, 3, 4], rng, k=4)
    assert shown == [1, 2, 3, 4]


def test_team_draft_identical_rankings_tie_on_average():
    rng = np.random.default_rng(3)
    a = list(range(10))
    outcomes = []
    for _ in range(4000):
        shown, team = team_draft(a, a, rng)
        clicks = rng.random(10) < 0.5 / (np.arange(10) + 1)
        outcomes.append(credit(team, clicks))
    assert abs(np.mean(outcomes)) < 0.03


@pytest.mark.parametrize("team,clicks,expected", [
    (["A", "B", "A", "B"], [1, 0, 0, 0], -1), (["A", "B", "A", "B"], [0, 1, 0, 1], 1),
    (["A", "B", "A", "B"], [1, 1, 0, 0], 0), (["A", "B"], [0, 0], 0), (["B", "A", "A"], [1, 1, 1], -1),
])
def test_credit_assignment(team, clicks, expected):
    assert credit(team, clicks) == expected


def test_sample_size_formulas():
    il = np.array([1] * 30 + [-1] * 20 + [0] * 50)
    n = interleaving_sample_size(il)
    mu, sd = il.mean(), il.std(ddof=1)
    assert n == pytest.approx((2.8016 * sd / mu) ** 2, rel=1e-3)
    assert interleaving_sample_size(np.array([1, -1, 0, 0])) == math.inf
    a, b = np.array([0, 1] * 50, dtype=float), np.array([0, 1, 1, 1] * 25, dtype=float)
    assert ab_sample_size(a, b) > 0
    assert sign_test_p(np.array([1] * 60 + [-1] * 40)) == pytest.approx(0.0455, abs=0.002)
