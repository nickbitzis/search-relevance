"""Position-based click model (PBM): P(click at rank k) = examination(k) * attractiveness(query, doc).

simulate_clicks plays shoppers who mostly look at the top of the page. fit_pbm recovers both factors from the log
with expectation maximization, which turns biased clicks into debiased relevance estimates: a result clicked 5% of
the time at rank 9 can be more attractive than one clicked 20% of the time at rank 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATTRACT_BY_GRADE = np.array([0.03, 0.15, 0.45, 0.8])


def examination(n_ranks: int = 10, power: float = 0.9) -> np.ndarray:
    return 1.0 / np.arange(1, n_ranks + 1) ** power


@dataclass
class ClickLog:
    query: np.ndarray  # query id per impression row
    doc: np.ndarray  # doc row per impression
    rank: np.ndarray  # 0-based position
    click: np.ndarray  # 0 or 1
    session: np.ndarray


def simulate_clicks(rng: np.random.Generator, sessions: list[tuple[int, np.ndarray, np.ndarray]],
                    theta: np.ndarray, attract_by_grade: np.ndarray = ATTRACT_BY_GRADE) -> ClickLog:
    """sessions: (query id, doc rows as shown, their true grades)."""
    q, d, r, c, s = [], [], [], [], []
    for sid, (qid, docs, grades) in enumerate(sessions):
        k = min(len(docs), len(theta))
        if k == 0:
            continue
        p = theta[:k] * attract_by_grade[grades[:k]]
        clicks = rng.random(k) < p
        q.append(np.full(k, qid))
        d.append(docs[:k])
        r.append(np.arange(k))
        c.append(clicks.astype(np.int8))
        s.append(np.full(k, sid))
    def cat(xs):
        return np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)

    return ClickLog(cat(q), cat(d), cat(r), cat(c), cat(s))


@dataclass
class PBMFit:
    theta: np.ndarray  # examination per rank, scaled so rank 1 is 1.0
    alpha: dict[tuple[int, int], float]  # attractiveness per (query, doc)
    impressions: dict[tuple[int, int], int]
    iterations: int


def fit_pbm(log: ClickLog, n_ranks: int = 10, iters: int = 150, prior: float = 0.2) -> PBMFit:
    """EM for the position-based model. prior adds a Beta-like pseudo count so rare pairs shrink toward 0.2."""
    keys = log.query.astype(np.int64) * 10_000_000 + log.doc.astype(np.int64)
    uniq, pair = np.unique(keys, return_inverse=True)
    n_pairs = len(uniq)
    theta = np.full(n_ranks, 0.5)
    alpha = np.full(n_pairs, 0.2)
    c = log.click.astype(np.float64)
    rk = log.rank
    counts = np.bincount(pair, minlength=n_pairs).astype(np.float64)
    rank_counts = np.bincount(rk, minlength=n_ranks).astype(np.float64)
    for _ in range(iters):
        t, a = theta[rk], alpha[pair]
        denom = np.maximum(1 - t * a, 1e-9)
        # posterior that the doc was attractive (and that the rank was examined) given the observation
        post_a = c + (1 - c) * (1 - t) * a / denom
        post_e = c + (1 - c) * t * (1 - a) / denom
        alpha = (np.bincount(pair, weights=post_a, minlength=n_pairs) + prior * 0.2) / (counts + prior)
        theta = np.bincount(rk, weights=post_e, minlength=n_ranks) / np.maximum(rank_counts, 1)
    scale = theta[0] if theta[0] > 0 else 1.0
    theta_n = theta / scale
    alpha = np.clip(alpha * scale, 0, 1)
    qs, ds = uniq // 10_000_000, uniq % 10_000_000
    return PBMFit(theta=theta_n, alpha={(int(q), int(d)): float(a) for q, d, a in zip(qs, ds, alpha, strict=True)},
                  impressions={(int(q), int(d)): int(n) for q, d, n in zip(qs, ds, counts, strict=True)},
                  iterations=iters)


def ctr_labels(log: ClickLog) -> dict[tuple[int, int], float]:
    """Raw click-through rate per (query, doc): the biased label PBM is compared against."""
    out: dict[tuple[int, int], list[float]] = {}
    for q, d, c in zip(log.query, log.doc, log.click, strict=True):
        v = out.setdefault((int(q), int(d)), [0.0, 0.0])
        v[0] += c
        v[1] += 1
    return {k: v[0] / v[1] for k, v in out.items()}
