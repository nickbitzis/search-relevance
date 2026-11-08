"""Query understanding: price parsing, spelling, synonym expansion, attribute extraction, and category classification,
in that order. The output is a QueryParse that retrieval, features, and the debugger all read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .attributes import Filters, extract_price, match_attribute
from .classifier import CategoryPrediction, QueryClassifier
from .spelling import Correction, SpellCorrector
from .synonyms import Expansion, find_synonyms
from .text import STOPWORDS, normalize, raw_tokens, stem, tokenize

SYNONYM_WEIGHT = 0.7  # query weight of terms that only come from a synonym


@dataclass
class Concept:
    """One thing the shopper asked for. A document matches it if it contains every token of any alternative."""

    text: str
    alternatives: list[tuple[str, ...]]
    synonym: bool = False


@dataclass
class QueryParse:
    raw: str
    normalized: str
    tokens: list[str]
    concepts: list[Concept]
    filters: Filters = field(default_factory=Filters)
    corrections: list[Correction] = field(default_factory=list)
    expansions: list[Expansion] = field(default_factory=list)
    attribute_spans: list[tuple[str, str, object]] = field(default_factory=list)  # (slot, text, value)
    category: CategoryPrediction = field(default_factory=lambda: CategoryPrediction([], {}, None, None))
    category_text: str = ""
    mode: str = "full"

    @property
    def suggestions(self) -> list[Correction]:
        return [c for c in self.corrections if not c.confident]

    @property
    def corrected_query(self) -> str:
        return " ".join(self.tokens)

    def terms(self) -> dict[str, float]:
        """Weighted query terms for BM25. The shopper's own words weigh 1.0, synonym-only words 0.7."""
        w: dict[str, float] = {}
        for c in self.concepts:
            for k, alt in enumerate(c.alternatives):
                for t in alt:
                    w[t] = max(w.get(t, 0.0), 1.0 if k == 0 else SYNONYM_WEIGHT)
        return w

    def synonym_terms(self) -> dict[str, float]:
        own = {t for c in self.concepts for t in c.alternatives[0]}
        return {t: wt for t, wt in self.terms().items() if t not in own}

    def as_dict(self) -> dict:
        return {
            "raw": self.raw, "normalized": self.normalized, "mode": self.mode, "corrected": self.corrected_query,
            "corrections": [{"from": c.original, "to": c.corrected, "distance": c.distance, "applied": c.confident}
                            for c in self.corrections],
            "synonyms": [{"phrase": e.phrase, "adds": e.canonical} for e in self.expansions],
            "attributes": [{"slot": s, "text": t, "value": v} for s, t, v in self.attribute_spans],
            "filters": self.filters.active(),
            "concepts": [{"text": c.text, "alternatives": [" ".join(a) for a in c.alternatives]}
                         for c in self.concepts],
            "category": self.category.as_dict(), "terms": {k: round(v, 2) for k, v in self.terms().items()},
        }


def _stems(phrase: str) -> tuple[str, ...]:
    return tuple(tokenize(phrase))


class QueryUnderstanding:
    def __init__(self, speller: SpellCorrector | None = None, classifier: QueryClassifier | None = None, *,
                 spelling: bool = True, synonyms: bool = True, attributes: bool = True, category: bool = True):
        self.speller = speller
        self.classifier = classifier
        self.spelling = spelling
        self.synonyms = synonyms
        self.attributes = attributes
        self.category = category

    @staticmethod
    def baseline(raw: str) -> QueryParse:
        """No understanding at all: every non-stopword token is a required concept."""
        toks = tokenize(raw)
        concepts = [Concept(t, [(t,)]) for t in dict.fromkeys(toks)]
        return QueryParse(raw=raw, normalized=normalize(raw), tokens=raw_tokens(raw), concepts=concepts,
                          mode="baseline")

    def parse(self, raw: str, apply_suggestions: bool = False) -> QueryParse:
        filters = Filters()
        spans: list[tuple[str, str, object]] = []
        text = raw.lower()
        corrections: list[Correction] = []
        if self.spelling and self.speller is not None:
            # Spelling runs first, word by word, so "uder $50" is fixed before the price parser reads it.
            words = text.split()
            for k, w in enumerate(words):
                core = normalize(w)
                if not core.isalpha():
                    continue
                c = self.speller.correct_token(core)
                if c is not None:
                    corrections.append(c)
                    if c.confident or apply_suggestions:
                        words[k] = c.corrected
            text = " ".join(words)
        if self.attributes:
            lo, hi, text, found = extract_price(text)
            filters.price_min, filters.price_max = lo, hi
            for f in found:
                spans.append(("price", f, {"min": lo, "max": hi}))
        toks = raw_tokens(text)
        concepts: list[Concept] = []
        expansions: list[Expansion] = []
        i = 0
        while i < len(toks):
            hit = find_synonyms(toks, i) if self.synonyms else None
            if hit:
                exp, n = hit
                own, canon = _stems(exp.phrase), _stems(exp.canonical)
                alts = [own] if own == canon else [own, canon]
                concepts.append(Concept(exp.phrase, alts, synonym=True))
                expansions.append(exp)
                i += n
                continue
            m = match_attribute(toks, i) if self.attributes else None
            if m:
                slot, value, n = m
                if getattr(filters, slot) is None:
                    setattr(filters, slot, value)
                spans.append((slot, " ".join(toks[i:i + n]), value))
                i += n
                continue
            t = toks[i]
            if t not in STOPWORDS:
                s = stem(t)
                if not any(c.alternatives[0] == (s,) for c in concepts):
                    concepts.append(Concept(t, [(s,)]))
            i += 1
        words = [c.text for c in concepts] + [" ".join(c.alternatives[1]) for c in concepts if len(c.alternatives) > 1]
        category_text = " ".join(words)
        parse = QueryParse(raw=raw, normalized=normalize(raw), tokens=toks, concepts=concepts, filters=filters,
                           corrections=corrections, expansions=expansions, attribute_spans=spans,
                           category_text=category_text)
        if self.category and self.classifier is not None:
            parse.category = self.classifier.predict(category_text)
        return parse
