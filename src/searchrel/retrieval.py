"""Candidate retrieval: concept matching with minimum-should-match, structured filters, BM25 ordering, and a
null-result fallback that relaxes the query one step at a time and records every step.

Fallback order when nothing matches:
  1. relax extracted filters, least important first (size, color, material, waterproof, price, gender, brand)
  2. apply "did you mean" spelling suggestions
  3. broaden the predicted category: leaf to parent, then no category
  4. match any concept instead of most of them
Filters the shopper picked from facets are never relaxed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .attributes import RELAX_ORDER, Filters, color_values, material_values
from .catalog import BRAND_NAMES, COLORS, GENDERS, LEAF_BY_KEY, LEAF_KEYS, MATERIALS, MIDS, SIZE_BIT, CatalogArrays
from .index import InvertedIndex
from .understanding import QueryParse

PRICE_BUCKETS = ((0, 50), (50, 100), (100, 200), (200, 400), (400, 100_000))


def required_matches(n_concepts: int) -> int:
    """Minimum should match: all concepts for one or two, 75% (rounded up) for more."""
    return n_concepts if n_concepts <= 2 else math.ceil(0.75 * n_concepts)


def filter_mask(a: CatalogArrays, f: Filters, category: tuple[str, str] | None = None) -> np.ndarray:
    m = np.ones(a.n, dtype=bool)
    if f.gender:
        m &= np.isin(a.gender, [GENDERS.index(f.gender), GENDERS.index("unisex")])
    if f.color:
        m &= np.isin(a.color, [COLORS.index(c) for c in color_values(f.color)])
    if f.size:
        m &= (a.size_mask == 0) | ((a.size_mask & SIZE_BIT[f.size]) != 0)
    if f.material:
        m &= np.isin(a.material, [MATERIALS.index(x) for x in material_values(f.material)])
    if f.waterproof_min:
        m &= a.waterproof >= f.waterproof_min
    if f.price_min is not None:
        m &= a.price >= f.price_min
    if f.price_max is not None:
        m &= a.price <= f.price_max
    if f.brand:
        m &= a.brand == BRAND_NAMES.index(f.brand)
    if category:
        level, value = category
        m &= (a.leaf == LEAF_KEYS.index(value)) if level == "leaf" else (a.mid == MIDS.index(value))
    return m


@dataclass
class Retrieval:
    parse: QueryParse
    ids: np.ndarray  # top k candidate rows, BM25 order
    scores: np.ndarray
    field_scores: dict[str, np.ndarray]  # per field, for the candidates
    syn_scores: np.ndarray
    hits: np.ndarray  # every matching row (for facets)
    filters: Filters
    category: tuple[str, str] | None
    steps: list[str] = field(default_factory=list)
    zero_initial: bool = False
    any_match: bool = False

    @property
    def n_hits(self) -> int:
        return len(self.hits)


class Retriever:
    def __init__(self, index: InvertedIndex, arrays: CatalogArrays, popularity: np.ndarray,
                 reparse: Callable[[str], QueryParse] | None = None):
        self.index = index
        self.a = arrays
        self.pop = popularity  # small tie breaker: log units sold
        self.reparse = reparse

    def concept_counts(self, parse: QueryParse) -> np.ndarray:
        counts = np.zeros(self.a.n, dtype=np.int16)
        for c in parse.concepts:
            cm = np.zeros(self.a.n, dtype=bool)
            for alt in c.alternatives:
                am = np.ones(self.a.n, dtype=bool)
                for t in alt:
                    am &= self.index.has_term(t)
                cm |= am
            counts += cm
        return counts

    def retrieve(self, parse: QueryParse, k: int = 100, fallback: bool = True, explicit: Filters | None = None,
                 explicit_category: tuple[str, str] | None = None) -> Retrieval:
        hard = filter_mask(self.a, explicit or Filters(), explicit_category)
        filters = parse.filters
        category = (parse.category.level, parse.category.value) if parse.category.level else None
        if explicit_category:
            category = None  # the shopper's own category choice replaces the predicted one
        steps: list[str] = []
        any_match = False

        def run(p: QueryParse) -> np.ndarray:
            counts = self.concept_counts(p)
            need = 1 if any_match else required_matches(len(p.concepts))
            return (counts >= need) & hard & filter_mask(self.a, filters, category)

        mask = run(parse)
        zero_initial = not mask.any()
        if zero_initial and fallback:
            for name in RELAX_ORDER:
                if mask.any():
                    break
                if getattr(filters, name) is not None:
                    filters = filters.without(name)
                    steps.append(f"relaxed {name.replace('_', ' ')}")
                    mask = run(parse)
            if not mask.any() and parse.suggestions and self.reparse is not None:
                fixed = self.reparse(parse.raw)
                steps.append("applied spelling suggestion " + ", ".join(
                    f"{c.original} -> {c.corrected}" for c in parse.suggestions))
                parse = QueryParse(**{**fixed.__dict__, "filters": filters, "category": parse.category})
                mask = run(parse)
            if not mask.any() and category and category[0] == "leaf":
                category = ("mid", LEAF_BY_KEY[category[1]].mid)
                steps.append(f"broadened category to {category[1]}")
                mask = run(parse)
            if not mask.any() and category:
                category = None
                steps.append("dropped category")
                mask = run(parse)
            if not mask.any() and len(parse.concepts) > 1:
                any_match = True
                steps.append("matched any term")
                mask = run(parse)
        total, per = self.index.score(parse.terms())
        syn = self.index.score(parse.synonym_terms())[0] if parse.synonym_terms() else np.zeros(self.a.n)
        hits = np.flatnonzero(mask)
        key = total[hits] + 1e-4 * self.pop[hits]
        if len(hits) > k:
            part = np.argpartition(-key, k - 1)[:k]
            order = part[np.argsort(-key[part], kind="stable")]
        else:
            order = np.argsort(-key, kind="stable")
        ids = hits[order]
        return Retrieval(parse=parse, ids=ids, scores=total[ids], field_scores={f: s[ids] for f, s in per.items()},
                         syn_scores=syn[ids], hits=hits, filters=filters, category=category, steps=steps,
                         zero_initial=zero_initial, any_match=any_match)


def facets(a: CatalogArrays, rows: np.ndarray, top: int = 8) -> dict[str, list[dict]]:
    """Counts over every matching product, not just the first page."""
    out: dict[str, list[dict]] = {}
    for name, col, labels in (("category", a.leaf, LEAF_KEYS), ("brand", a.brand, BRAND_NAMES),
                              ("color", a.color, COLORS), ("gender", a.gender, GENDERS)):
        c = np.bincount(col[rows], minlength=len(labels))
        order = np.argsort(-c, kind="stable")
        vals = [{"value": labels[i], "count": int(c[i])} for i in order[:top] if c[i] > 0]
        if name == "category":
            for v in vals:
                v["label"] = LEAF_BY_KEY[v["value"]].name
        out[name] = vals
    prices = a.price[rows]
    out["price"] = [{"value": f"{lo}-{hi}" if hi < 100_000 else f"{lo}+", "min": lo, "max": hi,
                     "count": int(((prices >= lo) & (prices < hi)).sum())} for lo, hi in PRICE_BUCKETS]
    return out
