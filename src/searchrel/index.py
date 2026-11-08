"""Inverted index with per-field BM25.

Each field has its own postings, document lengths, and average length, so a word in a short title counts for more
than the same word in a long description. A document's score is the sum over fields of boost * BM25, which keeps
the per-field scores available as ranking features.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .catalog import LEAF_BY_KEY, Product
from .text import tokenize

FIELD_BOOSTS = {"title": 3.0, "brand": 2.0, "category": 2.0, "attrs": 1.0, "description": 0.5}
K1, B = 1.2, 0.75


def product_fields(p: Product) -> dict[str, str]:
    leaf = LEAF_BY_KEY[p.leaf]
    attrs = [p.color, p.material, {"men": "mens", "women": "womens", "kids": "kids", "unisex": "unisex"}[p.gender]]
    if p.waterproof_mm:
        attrs.append("waterproof")
    return {"title": p.title, "brand": p.brand, "category": f"{p.top} {p.mid} {leaf.name}", "attrs": " ".join(attrs),
            "description": p.description}


def bm25_idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_tf(tf: np.ndarray | float, dl: np.ndarray | float, avgdl: float, k1: float = K1, b: float = B):
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


class InvertedIndex:
    def __init__(self, docs: list[dict[str, str]], boosts: dict[str, float] | None = None):
        self.boosts = dict(boosts or FIELD_BOOSTS)
        self.n = len(docs)
        self.postings: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self.doc_len: dict[str, np.ndarray] = {}
        self.avgdl: dict[str, float] = {}
        for f in self.boosts:
            lists: dict[str, tuple[list[int], list[int]]] = {}
            lens = np.zeros(self.n, dtype=np.float32)
            for i, d in enumerate(docs):
                toks = tokenize(d.get(f, ""))
                lens[i] = len(toks)
                for t, c in Counter(toks).items():
                    ids, tfs = lists.setdefault(t, ([], []))
                    ids.append(i)
                    tfs.append(c)
            self.postings[f] = {t: (np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.float32))
                                for t, (ids, tfs) in lists.items()}
            self.doc_len[f] = lens
            self.avgdl[f] = float(lens.mean()) if self.n else 1.0
        self._mask_cache: dict[tuple[str, str | None], np.ndarray] = {}

    @classmethod
    def from_products(cls, products: list[Product]) -> InvertedIndex:
        return cls([product_fields(p) for p in products])

    def df(self, term: str, field: str) -> int:
        p = self.postings[field].get(term)
        return 0 if p is None else len(p[0])

    def has_term(self, term: str, field: str | None = None) -> np.ndarray:
        """Boolean mask of documents containing the term in one field, or in any field."""
        key = (term, field)
        m = self._mask_cache.get(key)
        if m is None:
            m = np.zeros(self.n, dtype=bool)
            for f in ([field] if field else self.boosts):
                p = self.postings[f].get(term)
                if p is not None:
                    m[p[0]] = True
            if len(self._mask_cache) > 50_000:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m

    def field_scores(self, terms: dict[str, float]) -> dict[str, np.ndarray]:
        """Unboosted BM25 per field for every document, with query term weights applied."""
        out = {}
        for f in self.boosts:
            s = np.zeros(self.n, dtype=np.float64)
            for t, w in terms.items():
                p = self.postings[f].get(t)
                if p is None:
                    continue
                ids, tfs = p
                idf = bm25_idf(self.n, len(ids))
                s[ids] += w * idf * bm25_tf(tfs, self.doc_len[f][ids], self.avgdl[f])
            out[f] = s
        return out

    def score(self, terms: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        per = self.field_scores(terms)
        total = np.zeros(self.n, dtype=np.float64)
        for f, s in per.items():
            total += self.boosts[f] * s
        return total, per
