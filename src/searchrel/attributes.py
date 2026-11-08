"""Attribute extraction: turn words like "mens", "navy", "size m", "merino", "waterproof", and "under 200" into
structured filters, and take them out of the text query so BM25 does not try to match "under" or "200".
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .catalog import BRAND_NAMES, COLORS
from .text import normalize

GENDER_WORDS = {
    "men": "men", "mens": "men", "man": "men", "male": "men", "womens": "women", "women": "women", "woman": "women",
    "ladies": "women", "female": "women", "kids": "kids", "kid": "kids", "childrens": "kids", "children": "kids",
    "youth": "kids", "boys": "kids", "girls": "kids", "unisex": "unisex",
    "for men": "men", "for women": "women", "for kids": "kids",
}
COLOR_WORDS = {c: c for c in COLORS} | {
    "grey": "gray", "charcoal": "gray", "olive": "green", "tan": "brown", "maroon": "red", "crimson": "red",
    "cream": "white", "dark blue": "navy",
}
COLOR_FAMILY = {"blue": ("blue", "navy")}
MATERIAL_WORDS = {
    "merino": "merino wool", "merino wool": "merino wool", "wool": "wool", "leather": "leather", "nylon": "nylon",
    "cotton": "cotton", "titanium": "titanium", "aluminum": "aluminum", "aluminium": "aluminum", "carbon": "carbon",
    "synthetic": "synthetic", "polyester": "polyester", "suede": "suede",
}
MATERIAL_FAMILY = {"wool": ("wool", "merino wool")}
SIZE_WORDS = {"xs": "XS", "s": "S", "m": "M", "l": "L", "xl": "XL", "xxl": "XXL", "small": "S", "medium": "M",
              "large": "L", "extra large": "XL"} | {str(n): str(n) for n in range(5, 15)}
STANDALONE_SIZES = {"xs": "XS", "xl": "XL", "xxl": "XXL", "extra large": "XL"}
BRAND_WORDS = {normalize(b): b for b in BRAND_NAMES}
WATERPROOF_WORDS = {"waterproof": 1}

# Every word the attribute parser understands, so the spelling corrector never "fixes" them.
LEXICON = frozenset(w for table in (GENDER_WORDS, COLOR_WORDS, MATERIAL_WORDS, SIZE_WORDS, BRAND_WORDS,
                                    WATERPROOF_WORDS) for k in table for w in k.split()) | {
    "size", "under", "below", "over", "above", "between", "less", "more", "than", "least", "dollars", "bucks", "max",
    "cheaper", "up", "to", "mm"}

# Importance order for the null-result fallback: the first one is relaxed first.
RELAX_ORDER = ("size", "color", "material", "waterproof_min", "price_min", "price_max", "gender", "brand")


@dataclass
class Filters:
    gender: str | None = None
    color: str | None = None
    size: str | None = None
    material: str | None = None
    waterproof_min: int | None = None
    price_min: float | None = None
    price_max: float | None = None
    brand: str | None = None

    def active(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def without(self, name: str) -> Filters:
        f = Filters(**asdict(self))
        setattr(f, name, None)
        return f

    def merged(self, override: Filters) -> Filters:
        """Explicit filters (facet clicks) win over extracted ones."""
        d = asdict(self)
        d.update(override.active())
        return Filters(**d)


_NUM = r"\$?(\d+(?:\.\d+)?)"
PRICE_PATTERNS = (
    ("between", re.compile(rf"\bbetween {_NUM} (?:and|to|-) {_NUM}(?: dollars| bucks)?")),
    ("range", re.compile(rf"(?<![\w-]){_NUM} ?(?:-|to) ?{_NUM}(?: dollars| bucks)?(?![\w-])")),
    ("max", re.compile(rf"(?:\bunder|\bbelow|\bless than|\bcheaper than|\bup to|\bmax|<) ?{_NUM}(?: dollars| bucks)?")),
    ("min", re.compile(rf"(?:\bover|\babove|\bmore than|\bat least|>) ?{_NUM}(?: dollars| bucks)?")),
)


def extract_price(text: str) -> tuple[float | None, float | None, str, list[str]]:
    """Price constraints from raw lowercase text. Returns (min, max, text with the matches removed, matches)."""
    text = text.lower()
    lo = hi = None
    found: list[str] = []
    for kind, pat in PRICE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        nums = [float(g) for g in m.groups()]
        if kind in ("between", "range"):
            a, b = sorted(nums)
            if a < 10:  # "2-3 person" is not a price
                continue
            lo, hi = a, b
        elif kind == "max":
            hi = nums[0]
        else:
            lo = nums[0]
        found.append(m.group(0).strip())
        text = text[:m.start()] + " " + text[m.end():]
    return lo, hi, text, found


_WP_K = re.compile(r"^(\d{1,2})k$")
_WP_MM = re.compile(r"^(\d{4,5})mm$")


def match_attribute(tokens: list[str], i: int) -> tuple[str, object, int] | None:
    """Try to read one attribute starting at tokens[i]. Returns (slot, value, tokens used) or None."""
    two = " ".join(tokens[i:i + 2]) if i + 1 < len(tokens) else None
    three = " ".join(tokens[i:i + 3]) if i + 2 < len(tokens) else None
    tok = tokens[i]
    for phrase, n in ((three, 3), (two, 2), (tok, 1)):
        if phrase is None:
            continue
        if phrase in BRAND_WORDS:
            return "brand", BRAND_WORDS[phrase], n
        if phrase in GENDER_WORDS:
            return "gender", GENDER_WORDS[phrase], n
        if phrase in COLOR_WORDS:
            return "color", COLOR_WORDS[phrase], n
        if phrase in MATERIAL_WORDS:
            return "material", MATERIAL_WORDS[phrase], n
        if phrase in STANDALONE_SIZES:
            return "size", STANDALONE_SIZES[phrase], n
    if tok == "size" and i + 1 < len(tokens):
        nxt2 = " ".join(tokens[i + 1:i + 3])
        if nxt2 in SIZE_WORDS:
            return "size", SIZE_WORDS[nxt2], 3
        if tokens[i + 1] in SIZE_WORDS:
            return "size", SIZE_WORDS[tokens[i + 1]], 2
    if tok in WATERPROOF_WORDS:
        return "waterproof_min", 1, 1
    m = _WP_K.match(tok) or _WP_MM.match(tok)
    if m:
        v = int(m.group(1))
        v = v * 1000 if v < 100 else v
        if 1000 <= v <= 40000:
            used = 2 if i + 1 < len(tokens) and tokens[i + 1] == "waterproof" else 1
            return "waterproof_min", v, used
    if tok.isdigit() and i + 1 < len(tokens) and tokens[i + 1] == "mm" and 1000 <= int(tok) <= 40000:
        return "waterproof_min", int(tok), 2
    return None


def color_values(color: str) -> tuple[str, ...]:
    return COLOR_FAMILY.get(color, (color,))


def material_values(material: str) -> tuple[str, ...]:
    return MATERIAL_FAMILY.get(material, (material,))
