"""Tokenization shared by the index, the query parser, and the spelling vocabulary."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and the for of with in on to by from at or is are my best good new".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, fold apostrophes so men's becomes mens, and turn other punctuation into spaces."""
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(_TOKEN.findall(text))


def stem(token: str) -> str:
    """A light plural stemmer: jackets -> jacket, batteries -> battery, dress stays dress."""
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def raw_tokens(text: str) -> list[str]:
    """Normalized tokens before stemming and stopword removal (used by spelling and attribute parsing)."""
    return normalize(text).split()


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Index-time and query-time analyzer: normalize, drop stopwords, stem."""
    return [stem(t) for t in raw_tokens(text) if keep_stopwords or t not in STOPWORDS]
