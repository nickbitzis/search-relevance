"""Query category classifier: word and character n-gram TF-IDF into multinomial logistic regression.

Training labels come from the query log, the way a real team gets them: the category of the product a shopper
clicked. Those labels are noisy (people click a sibling category or something random), and the model is trained on
the noise, not on the hidden intent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .catalog import LEAF_BY_KEY, LEAF_KEYS

LEAF_THRESHOLD = 0.6  # filter to the leaf above this probability
MID_THRESHOLD = 0.75  # otherwise filter to the parent if the leaves under it add up to this


@dataclass
class CategoryPrediction:
    top: list[tuple[str, float]]  # best leaves with probabilities
    probs: dict[str, float]  # every leaf
    level: str | None  # "leaf", "mid", or None (no filter)
    value: str | None

    def mid_prob(self, mid: str) -> float:
        return sum(p for k, p in self.probs.items() if LEAF_BY_KEY[k].mid == mid)

    def as_dict(self) -> dict:
        return {"top": [{"leaf": k, "p": round(p, 3)} for k, p in self.top], "filter_level": self.level,
                "filter_value": self.value}


class QueryClassifier:
    def __init__(self, seed: int = 0):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
        self.model = LogisticRegression(C=6.0, max_iter=400, random_state=seed)
        self.classes: list[str] = []
        self._cache: dict[str, CategoryPrediction] = {}

    def _features(self, texts: list[str]):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts: list[str], labels: list[str], weights: list[float] | None = None) -> QueryClassifier:
        self.word.fit(texts)
        self.char.fit(texts)
        self.model.fit(self._features(texts), labels, sample_weight=None if weights is None else np.asarray(weights))
        self.classes = [str(c) for c in self.model.classes_]
        self._coef = self.model.coef_.T
        self._intercept = self.model.intercept_
        self._cache.clear()
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        """Softmax over classes, computed directly from the fitted weights (much faster than predict_proba
        for one query at a time)."""
        z = np.asarray(self._features(texts) @ self._coef) + self._intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, text: str) -> CategoryPrediction:
        if not text.strip() or not self.classes:
            return CategoryPrediction([], {}, None, None)
        hit = self._cache.get(text)
        if hit is None:
            row = self.proba([text])[0]
            hit = self._decide({c: float(v) for c, v in zip(self.classes, row, strict=True)})
            if len(self._cache) > 20_000:
                self._cache.clear()
            self._cache[text] = hit
        return hit

    @staticmethod
    def _decide(probs: dict[str, float]) -> CategoryPrediction:
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        level = value = None
        if top[0][1] >= LEAF_THRESHOLD:
            level, value = "leaf", top[0][0]
        else:
            mids: dict[str, float] = {}
            for k, v in probs.items():
                mids[LEAF_BY_KEY[k].mid] = mids.get(LEAF_BY_KEY[k].mid, 0.0) + v
            best_mid = max(mids, key=mids.get)
            if mids[best_mid] >= MID_THRESHOLD:
                level, value = "mid", best_mid
        return CategoryPrediction(top, probs, level, value)


def click_label(rng: np.random.Generator, leaf: str | None, mid: str | None) -> str | None:
    """Category of the clicked product for one logged search: right 85% of the time, a sibling 10%, random 5%."""
    if leaf is None and mid is None:
        return None
    if leaf is None:
        options = [k for k in LEAF_KEYS if LEAF_BY_KEY[k].mid == mid]
        return options[int(rng.integers(len(options)))]
    r = rng.random()
    if r < 0.85:
        return leaf
    siblings = [k for k in LEAF_KEYS if LEAF_BY_KEY[k].mid == LEAF_BY_KEY[leaf].mid and k != leaf]
    if r < 0.95 and siblings:
        return siblings[int(rng.integers(len(siblings)))]
    return LEAF_KEYS[int(rng.integers(len(LEAF_KEYS)))]
