"""Builds the whole stack from a seed: catalog, query log, index, spelling vocabulary, category classifier,
judgments, features, the served ranker, and click logs. Serves search and the query debugger.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .attributes import LEXICON, Filters
from .catalog import LEAF_BY_KEY, LEAF_KEYS, MIDS, CatalogArrays, Product, generate_catalog
from .classifier import QueryClassifier, click_label
from .features import FEATURES, compute_features, leaf_median_prices
from .index import FIELD_BOOSTS, InvertedIndex, product_fields
from .ltr import Ranker, rank_order
from .oracle import grades as oracle_grades
from .querylog import QueryLog, generate_query_log, select_judged, token_counts
from .retrieval import Retrieval, Retriever, facets
from .spelling import SpellCorrector
from .synonyms import SYNONYM_WORDS
from .text import STOPWORDS, raw_tokens
from .understanding import QueryParse, QueryUnderstanding


@dataclass
class EngineConfig:
    n_products: int = 20_000
    n_searches: int = 50_000
    n_judged: int = 600
    seed: int = 7
    pool_k: int = 100  # candidates retrieved for reranking and judged per system
    folds: int = 5
    classifier_holdout: float = 0.2


@dataclass
class JudgedQuery:
    qi: int  # index into the query log's distinct strings
    text: str
    segment: str
    typo: bool
    attribute: bool
    weight: int  # searches in the log
    grades: np.ndarray  # hidden grade of every product (ideal DCG and recall denominators)
    base: Retrieval  # BM25 only
    full: Retrieval  # BM25 plus query understanding, with fallback
    X: np.ndarray  # features for full.ids
    judgments: dict[int, int] = field(default_factory=dict)  # pooled candidates -> grade

    @property
    def n_relevant(self) -> int:
        return int((self.grades >= 2).sum())


class Engine:
    def __init__(self, cfg: EngineConfig | None = None, train: bool = True):
        t0 = time.perf_counter()
        self.cfg = cfg = cfg or EngineConfig()
        self.timings: dict[str, float] = {}
        self.products: list[Product] = generate_catalog(cfg.n_products, seed=cfg.seed + 4)
        self.by_id = {p.id: i for i, p in enumerate(self.products)}
        self.arrays = CatalogArrays(self.products)
        self.log: QueryLog = generate_query_log(self.products, cfg.n_searches, seed=cfg.seed + 16)
        self._tick("data", t0)
        self.index = InvertedIndex.from_products(self.products)
        self.popularity = np.log1p(np.array([p.units_90d for p in self.products], dtype=np.float64))
        self.median_price = leaf_median_prices(self.products)
        self.speller = SpellCorrector(self._vocabulary())
        self._tick("index", t0)
        self.qu = QueryUnderstanding(self.speller, None)
        self.classifier, self.classifier_holdout = self._train_classifier()
        self.qu.classifier = self.classifier
        self.retriever = Retriever(self.index, self.arrays, self.popularity,
                                   reparse=lambda raw: self.qu.parse(raw, apply_suggestions=True))
        self._tick("query_understanding", t0)
        self.judged: list[JudgedQuery] = self._build_judged()
        self._tick("judgments", t0)
        self.ranker: Ranker | None = None
        self.oof: dict[str, list[np.ndarray]] = {}
        self.clicks = None
        if train:
            from .evaluate import train_rankers

            train_rankers(self)
            self._tick("ranking", t0)
        self.startup_seconds = round(time.perf_counter() - t0, 2)

    def _tick(self, name: str, t0: float) -> None:
        self.timings[name] = round(time.perf_counter() - t0 - sum(self.timings.values()), 2)

    # ---- construction -------------------------------------------------------------------------------------------

    def _vocabulary(self) -> dict[str, int]:
        """Catalog token counts plus frequent query log tokens, plus every word the parser and synonym table know."""
        freq: Counter = Counter()
        for p in self.products:
            for text in product_fields(p).values():
                freq.update(raw_tokens(text))
        for t, c in token_counts(self.log).items():
            if c >= 20:
                freq[t] += c
        for w in LEXICON | SYNONYM_WORDS | STOPWORDS:
            freq[w] = max(freq[w], 50)
        return {w: c for w, c in freq.items() if c >= 2 and not w.isdigit()}

    def _train_classifier(self) -> tuple[QueryClassifier, set[int]]:
        rng = np.random.default_rng(self.cfg.seed + 101)
        n = len(self.log.strings)
        holdout = set(np.flatnonzero(rng.random(n) < self.cfg.classifier_holdout).tolist())
        texts: dict[int, str] = {}
        pairs: Counter = Counter()
        for qi in self.log.searches:
            qi = int(qi)
            if qi in holdout:
                continue
            it = self.log.intents[qi]
            label = click_label(rng, it.leaf, it.mid)
            if label is None:
                continue
            if qi not in texts:
                texts[qi] = self.qu.parse(self.log.strings[qi]).category_text
            if texts[qi]:
                pairs[(texts[qi], label)] += 1
        X = [t for t, _ in pairs]
        y = [lab for _, lab in pairs]
        w = [math.log1p(c) for c in pairs.values()]
        return QueryClassifier(seed=self.cfg.seed).fit(X, y, w), holdout

    def _build_judged(self) -> list[JudgedQuery]:
        out = []
        for qi in select_judged(self.log, self.cfg.n_judged, seed=self.cfg.seed + 5):
            text = self.log.strings[qi]
            it = self.log.intents[qi]
            g = oracle_grades(it, self.arrays)
            base = self.retriever.retrieve(QueryUnderstanding.baseline(text), k=self.cfg.pool_k, fallback=False)
            full = self.retriever.retrieve(self.qu.parse(text), k=self.cfg.pool_k)
            X = compute_features(full, self.products, self.index, self.median_price)
            jq = JudgedQuery(qi=qi, text=text, segment=self.log.segment(qi), typo=self.log.typo_of[qi] is not None,
                             attribute=it.n_attributes > 0, weight=int(self.log.counts[qi]), grades=g, base=base,
                             full=full, X=X)
            pool = set(base.ids.tolist()) | set(full.ids.tolist())
            jq.judgments = {int(d): int(g[d]) for d in pool}
            out.append(jq)
        return out

    # ---- serving ------------------------------------------------------------------------------------------------

    def parse(self, q: str, mode: str = "full") -> QueryParse:
        return QueryUnderstanding.baseline(q) if mode == "bm25" else self.qu.parse(q)

    def retrieve(self, q: str, ranker: str = "ltr", explicit: Filters | None = None,
                 explicit_category: tuple[str, str] | None = None) -> tuple[Retrieval, np.ndarray, np.ndarray]:
        """Returns the retrieval, the final order (indices into retrieval.ids), and the LTR scores (or BM25)."""
        mode = "bm25" if ranker == "bm25" else "full"
        r = self.retriever.retrieve(self.parse(q, mode), k=self.cfg.pool_k, fallback=mode != "bm25",
                                    explicit=explicit, explicit_category=explicit_category)
        if ranker == "ltr" and self.ranker is not None and len(r.ids):
            X = compute_features(r, self.products, self.index, self.median_price)
            s = self.ranker.predict(X)
            return r, rank_order(s), s
        return r, np.arange(len(r.ids)), r.scores

    def search(self, q: str, ranker: str = "ltr", explicit: Filters | None = None,
               category: str | None = None, limit: int = 24) -> dict:
        cat = None
        if category:
            cat = ("leaf", category) if category in LEAF_BY_KEY else ("mid", category) if category in MIDS else None
        r, order, scores = self.retrieve(q, ranker, explicit, cat)
        results = []
        for pos, j in enumerate(order[:limit]):
            p = self.products[r.ids[j]]
            results.append(p.public() | {"rank": pos + 1, "bm25_rank": int(j) + 1, "score": round(float(scores[j]), 4)})
        parse = r.parse
        applied = [c for c in parse.corrections if c.confident]
        return {
            "query": q, "ranker": ranker, "total": r.n_hits, "results": results,
            "facets": facets(self.arrays, r.hits),
            "spelling": {"showing_results_for": parse.corrected_query if applied else None,
                         "did_you_mean": (" ".join({c.original: c.corrected for c in parse.suggestions}.get(t, t)
                                                   for t in parse.tokens) if parse.suggestions else None)},
            "filters": r.filters.active(), "category_filter": list(r.category) if r.category else None,
            "fallback": r.steps, "zero_results_before_fallback": r.zero_initial,
            "parse": parse.as_dict(),
        }

    def debug(self, q: str, top: int = 20) -> dict:
        """Everything that happened to one query: parse, filters, fallback, candidates, features, final ranking."""
        t0 = time.perf_counter()
        r, order, scores = self.retrieve(q, "ltr")
        X = compute_features(r, self.products, self.index, self.median_price)
        contrib = self.ranker.contributions(X[order[:top]]) if self.ranker is not None and len(r.ids) else None
        judged = next((j for j in self.judged if j.text == q.strip().lower()), None)
        rows = []
        for pos, j in enumerate(order[:top]):
            p = self.products[r.ids[j]]
            feats = {f: round(float(X[j, k]), 4) for k, f in enumerate(FEATURES)}
            row = {"rank": pos + 1, "bm25_rank": int(j) + 1, "id": p.id, "title": p.title, "category": p.category_path,
                   "price": p.price, "ltr_score": round(float(scores[j]), 4), "bm25": round(float(r.scores[j]), 3),
                   "features": feats}
            if contrib is not None:
                c = contrib[pos]
                top_k = np.argsort(-np.abs(c))[:4]
                row["why"] = [{"feature": FEATURES[k], "value": feats[FEATURES[k]],
                               "contribution": round(float(c[k]), 3)} for k in top_k]
            if judged is not None:
                row["grade"] = int(judged.grades[r.ids[j]])
            rows.append(row)
        out = {
            "query": q, "parse": r.parse.as_dict(), "filters_applied": r.filters.active(),
            "category_filter": list(r.category) if r.category else None, "fallback_steps": r.steps,
            "zero_results_before_fallback": r.zero_initial, "matched_any_term": r.any_match,
            "candidates": r.n_hits, "reranked": len(r.ids), "field_boosts": FIELD_BOOSTS,
            "ranker": {"backend": self.ranker.backend if self.ranker else None, "features": list(FEATURES)},
            "results": rows, "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if judged is not None:
            from .evaluate import query_metrics

            out["judged"] = {"segment": judged.segment, "typo": judged.typo, "n_relevant": judged.n_relevant,
                             "metrics": query_metrics(self, judged)}
        return out

    def judged_examples(self, n: int = 8) -> list[dict]:
        """Judged queries where learning to rank changed NDCG@10 the most, for the debugger."""
        if not hasattr(self, "_judged_examples"):
            from .evaluate import query_metrics

            rows = []
            for jq in self.judged:
                if jq.n_relevant == 0 or not self.oof:
                    continue
                m = query_metrics(self, jq)
                rows.append({"query": jq.text, "segment": "typo" if jq.typo else jq.segment,
                             "bm25_qu": m["bm25_qu"]["ndcg@10"], "ltr": m["ltr"]["ndcg@10"]})
            rows.sort(key=lambda r: r["bm25_qu"] - r["ltr"])
            self._judged_examples = rows[:n]
        return self._judged_examples

    def sample_queries(self) -> dict:
        """Example queries for the UI, one per interesting case."""
        return {"judged": self.judged_examples(),
                "examples": ["mens blue waterproof jacket under 200", "waterproof shell", "hikign boots", "puffy",
                             "quillback", "womens merino base layer top size m", "pink kids climbing harness under 20",
                             "tent repair kit", "headlamp", "anorak"],
                "leaves": [{"key": k, "name": LEAF_BY_KEY[k].name} for k in LEAF_KEYS]}
