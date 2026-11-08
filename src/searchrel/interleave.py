"""Team-draft interleaving and sample size math for interleaving vs an A/B test.

Each impression merges two rankings the way captains pick teams: a coin flip decides who picks first in each
round, and each team adds its highest ranked result not already shown. Clicks are credited to the team that
contributed the clicked result; the impression goes to the team with more clicks, or is a tie.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def team_draft(a: list[int], b: list[int], rng: np.random.Generator, k: int = 10) -> tuple[list[int], list[str]]:
    shown: list[int] = []
    team: list[str] = []
    used: set[int] = set()
    ia = ib = 0
    count = {"A": 0, "B": 0}
    while len(shown) < k:
        while ia < len(a) and a[ia] in used:
            ia += 1
        while ib < len(b) and b[ib] in used:
            ib += 1
        if ia >= len(a) and ib >= len(b):
            break
        a_turn = count["A"] < count["B"] or (count["A"] == count["B"] and rng.random() < 0.5)
        if (a_turn and ia < len(a)) or ib >= len(b):
            doc, t = a[ia], "A"
        else:
            doc, t = b[ib], "B"
        used.add(doc)
        shown.append(doc)
        team.append(t)
        count[t] += 1
    return shown, team


def credit(team: list[str], clicks) -> int:
    """+1 if B's results got more clicks, -1 if A's did, 0 for a tie (including no clicks)."""
    a = sum(1 for t, c in zip(team, clicks, strict=False) if c and t == "A")
    b = sum(1 for t, c in zip(team, clicks, strict=False) if c and t == "B")
    return (b > a) - (a > b)


def _z(alpha: float, power: float) -> float:
    nd = NormalDist()
    return nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)


def interleaving_sample_size(outcomes: np.ndarray, alpha: float = 0.05, power: float = 0.8) -> float:
    """Impressions needed to detect the observed mean preference with a one-sample z test."""
    mu, sd = float(np.mean(outcomes)), float(np.std(outcomes, ddof=1))
    return math.inf if mu == 0 else (_z(alpha, power) * sd / mu) ** 2


def ab_sample_size(metric_a: np.ndarray, metric_b: np.ndarray, alpha: float = 0.05, power: float = 0.8,
                   delta: float | None = None) -> float:
    """Total sessions (both arms) to detect the difference in means with a two-sample z test. Pass delta when a
    less noisy estimate of the true effect exists (the simulation uses common random numbers for that)."""
    if delta is None:
        delta = float(np.mean(metric_b) - np.mean(metric_a))
    var = (float(np.var(metric_a, ddof=1)) + float(np.var(metric_b, ddof=1))) / 2
    return math.inf if delta == 0 else 2 * 2 * _z(alpha, power) ** 2 * var / delta ** 2


def sign_test_p(outcomes: np.ndarray) -> float:
    """Two-sided normal approximation to the sign test over non-tied impressions."""
    wins, losses = int(np.sum(outcomes > 0)), int(np.sum(outcomes < 0))
    n = wins + losses
    if n == 0:
        return 1.0
    z = (wins - n / 2) / math.sqrt(n / 4)
    return 2 * (1 - NormalDist().cdf(abs(z)))
