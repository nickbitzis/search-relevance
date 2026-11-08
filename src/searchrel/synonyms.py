"""Curated synonym table: how shoppers say it -> how the catalog says it.

Keys and values are normalized phrases (lowercase, no punctuation). Expansion is additive: the shopper's own words
stay in the query, and the catalog phrase is added as an alternative with a lower weight. Phrases are matched
longest first, so "waterproof shell" wins over a bare "waterproof".
"""

from __future__ import annotations

from dataclasses import dataclass

SYNONYMS: dict[str, str] = {
    "waterproof shell": "rain jacket", "rain shell": "rain jacket", "hardshell": "rain jacket",
    "hard shell": "rain jacket", "raincoat": "rain jacket", "rain coat": "rain jacket",
    "puffer": "insulated jacket", "puffer jacket": "insulated jacket", "puffy": "insulated jacket",
    "puffy jacket": "insulated jacket", "down coat": "down jacket",
    "soft shell": "softshell jacket", "softshell": "softshell jacket",
    "trekking pants": "hiking pant", "zip off pants": "convertible pant",
    "waterproof pants": "rain pant", "rain trousers": "rain pant", "overtrousers": "rain pant",
    "leggings": "tights",
    "long underwear": "base layer", "long johns": "base layer bottom", "thermals": "base layer",
    "tshirt": "tee", "t shirt": "tee", "tee shirt": "tee",
    "sun hoodie": "sun shirt",
    "trekking boots": "hiking boot", "walking boots": "hiking boot", "hikers": "hiking boot",
    "trail shoes": "trail running shoe", "trail runners": "trail running shoe",
    "flip flops": "sandal",
    "mummy bag": "sleeping bag",
    "sleeping mat": "sleeping pad", "camping mat": "sleeping pad", "air mattress": "air pad",
    "camping stove": "camp stove", "burner": "stove",
    "mess kit": "cook set", "pots and pans": "cook set",
    "camping chair": "camp chair",
    "day pack": "daypack", "rucksack": "backpacking pack",
    "water pack": "hydration pack",
    "head torch": "headlamp", "head lamp": "headlamp",
    "water flask": "water bottle", "thermos": "insulated bottle",
    "walking sticks": "trekking pole", "trekking sticks": "trekking pole",
    "toque": "beanie", "winter hat": "beanie",
}
MAX_PHRASE = max(len(k.split()) for k in SYNONYMS)
SYNONYM_WORDS = frozenset(w for k, v in SYNONYMS.items() for w in (k + " " + v).split())


@dataclass(frozen=True)
class Expansion:
    phrase: str  # what the shopper typed
    canonical: str  # catalog phrase added as an alternative


def find_synonyms(tokens: list[str], start: int) -> tuple[Expansion, int] | None:
    """Longest synonym phrase starting at tokens[start]; returns the expansion and how many tokens it used."""
    for n in range(min(MAX_PHRASE, len(tokens) - start), 0, -1):
        phrase = " ".join(tokens[start:start + n])
        if phrase in SYNONYMS:
            return Expansion(phrase, SYNONYMS[phrase]), n
    return None


def expand(tokens: list[str]) -> list[Expansion]:
    """All synonym expansions in a token list, scanning left to right with longest match."""
    out, i = [], 0
    while i < len(tokens):
        hit = find_synonyms(tokens, i)
        if hit:
            out.append(hit[0])
            i += hit[1]
        else:
            i += 1
    return out
