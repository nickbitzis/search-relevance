"""Spelling correction: Damerau-Levenshtein distance against a vocabulary with frequency priors.

Candidate lookup uses a symmetric delete index (the SymSpell idea): every vocabulary word is stored under all of its
1 and 2 character deletions, so a misspelling finds its neighbors by generating its own deletions instead of
scanning the vocabulary. Candidates are then verified with the true edit distance and scored as
log(frequency) - 4.5 * distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insertions, deletions, substitutions, and adjacent transpositions."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def _deletes(word: str, depth: int) -> set[str]:
    out = {word}
    frontier = {word}
    for _ in range(depth):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        out |= nxt
        frontier = nxt
    return out


def max_distance(token: str) -> int:
    return 1 if len(token) <= 4 else 2


@dataclass
class Correction:
    original: str
    corrected: str
    distance: int
    confident: bool  # applied automatically, or only offered as "did you mean"


@dataclass
class SpellCorrector:
    freq: dict[str, int]
    protected: frozenset[str] = frozenset()
    distance_penalty: float = 4.5
    _index: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for w in self.freq:
            for d in _deletes(w, max_distance(w)):
                self._index.setdefault(d, []).append(w)

    def known(self, token: str) -> bool:
        return token in self.freq or token in self.protected

    def candidates(self, token: str) -> list[tuple[str, int, float]]:
        """(word, distance, score) for vocabulary words within the allowed distance, best first."""
        limit = max_distance(token)
        seen: dict[str, int] = {}
        for d in _deletes(token, limit):
            for w in self._index.get(d, ()):
                if w not in seen:
                    seen[w] = edit_distance(token, w)
        out = [(w, dist, math.log(self.freq[w] + 1) - self.distance_penalty * dist)
               for w, dist in seen.items() if 0 < dist <= limit]
        return sorted(out, key=lambda t: (-t[2], t[0]))

    def correct_token(self, token: str) -> Correction | None:
        """Unknown tokens get the best candidate applied. Known but rare tokens next to a much more common word get a
        suggestion only (the fallback can apply it later)."""
        if len(token) <= 2 or token.isdigit() or token in self.protected or any(c.isdigit() for c in token):
            return None
        if token not in self.freq:
            cands = self.candidates(token)
            if not cands:
                return None
            best, dist, _ = cands[0]
            return Correction(token, best, dist, confident=True)
        own = self.freq[token]
        if own >= 5:
            return None
        cands = [c for c in self.candidates(token) if c[1] == 1 and self.freq[c[0]] >= 50 * max(own, 1)]
        if cands:
            return Correction(token, cands[0][0], 1, confident=False)
        return None

    def correct(self, tokens: list[str]) -> tuple[list[str], list[Correction]]:
        """Apply confident corrections; return the new tokens and every correction or suggestion found."""
        out, found = [], []
        for t in tokens:
            c = self.correct_token(t)
            if c is not None:
                found.append(c)
                out.append(c.corrected if c.confident else t)
            else:
                out.append(t)
        return out, found
