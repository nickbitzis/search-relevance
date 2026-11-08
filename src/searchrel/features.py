"""Ranking features for (query, candidate) pairs. Every feature is computed from the parse, the retrieval result,
and observable catalog fields. Margin is kept out of the relevance model and used only as an optional business blend.
"""

from __future__ import annotations

import math

import numpy as np

from .attributes import color_values, material_values
from .catalog import LEAF_BY_KEY, Product
from .index import InvertedIndex
from .retrieval import Retrieval

FEATURES = (
    "bm25_title", "bm25_brand", "bm25_category", "bm25_attrs", "bm25_description", "bm25_total", "bm25_rank_recip",
    "synonym_score", "title_coverage", "category_prob_leaf", "category_prob_mid", "brand_match", "attr_matches",
    "attr_mismatches", "price_ratio", "log_units_90d", "rating", "log_reviews", "in_stock", "fallback_steps",
)
FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES)}


def leaf_median_prices(products: list[Product]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for p in products:
        by.setdefault(p.leaf, []).append(p.price)
    return {k: float(np.median(v)) for k, v in by.items()}


def attribute_agreement(p: Product, f) -> tuple[int, int]:
    """(matches, mismatches) between a product and the attributes the shopper asked for."""
    ok = bad = 0
    checks = []
    if f.gender:
        checks.append(p.gender in (f.gender, "unisex"))
    if f.color:
        checks.append(p.color in color_values(f.color))
    if f.size and p.sizes:
        checks.append(f.size in p.sizes)
    if f.material:
        checks.append(p.material in material_values(f.material))
    if f.waterproof_min:
        checks.append(p.waterproof_mm >= f.waterproof_min)
    if f.price_max is not None:
        checks.append(p.price <= f.price_max)
    if f.price_min is not None:
        checks.append(p.price >= f.price_min)
    for c in checks:
        ok += c
        bad += not c
    return ok, bad


def compute_features(r: Retrieval, products: list[Product], index: InvertedIndex,
                     median_price: dict[str, float]) -> np.ndarray:
    n = len(r.ids)
    X = np.zeros((n, len(FEATURES)), dtype=np.float64)
    if n == 0:
        return X
    parse = r.parse
    asked = parse.filters  # what the shopper asked for, before any relaxation
    probs = parse.category.probs
    mid_prob = {}
    for k, v in probs.items():
        mid_prob[LEAF_BY_KEY[k].mid] = mid_prob.get(LEAF_BY_KEY[k].mid, 0.0) + v
    title_cov = np.zeros(n)
    if parse.concepts:
        for c in parse.concepts:
            hit = np.zeros(n, dtype=bool)
            for alt in c.alternatives:
                m = np.ones(n, dtype=bool)
                for t in alt:
                    m &= index.has_term(t, "title")[r.ids]
                hit |= m
            title_cov += hit
        title_cov /= len(parse.concepts)
    X[:, 0] = r.field_scores["title"]
    X[:, 1] = r.field_scores["brand"]
    X[:, 2] = r.field_scores["category"]
    X[:, 3] = r.field_scores["attrs"]
    X[:, 4] = r.field_scores["description"]
    X[:, 5] = r.scores
    X[:, 6] = 1.0 / (1.0 + np.arange(n))
    X[:, 7] = r.syn_scores
    X[:, 8] = title_cov
    X[:, 19] = len(r.steps)
    for j, row in enumerate(r.ids):
        p = products[row]
        X[j, 9] = probs.get(p.leaf, 0.0) if probs else -1.0
        X[j, 10] = mid_prob.get(p.mid, 0.0) if probs else -1.0
        X[j, 11] = 0.5 if not asked.brand else float(p.brand == asked.brand)
        X[j, 12], X[j, 13] = attribute_agreement(p, asked)
        cap = asked.price_max
        X[j, 14] = p.price / cap if cap else p.price / median_price.get(p.leaf, p.price)
        X[j, 15] = math.log1p(p.units_90d)
        X[j, 16] = p.rating
        X[j, 17] = math.log1p(p.reviews)
        X[j, 18] = float(p.inventory > 0)
    return X

