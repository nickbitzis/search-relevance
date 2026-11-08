"""Synthetic query log: about 50,000 searches over a few thousand distinct query strings with a Zipf shaped
head, torso, and tail. Each distinct string comes from a hidden intent, so every query has ground truth for
spelling, attributes, category, and relevance.

The log includes the things that make product search hard: synonyms the catalog never uses ("waterproof shell"),
synonyms nobody put in the table ("anorak"), attribute-laden queries ("mens blue waterproof jacket under 200"),
brand queries, typos, overconstrained queries, and things the store does not sell ("hammock").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .catalog import BRAND_NAMES, LEAF_BY_KEY, LEAVES, Product
from .oracle import Intent

LEAF_PHRASES: dict[str, tuple[str, ...]] = {
    "rain_jacket": ("rain jacket", "rain jackets", "waterproof shell", "rain shell", "raincoat", "hardshell", "anorak",
                    "rain slicker"),
    "insulated_jacket": ("down jacket", "insulated jacket", "puffer jacket", "puffy", "puffer", "parka",
                         "synthetic insulated jacket"),
    "fleece_jacket": ("fleece jacket", "fleece", "fleece hoodie", "fleece pullover"),
    "softshell_jacket": ("softshell jacket", "soft shell", "softshell"),
    "hiking_pants": ("hiking pants", "trekking pants", "trail pants", "convertible pants", "zip off pants"),
    "rain_pants": ("rain pants", "waterproof pants", "rain trousers", "overtrousers"),
    "tights": ("tights", "leggings", "running tights"),
    "base_top": ("base layer top", "thermal top", "base layer crew", "long underwear top"),
    "base_bottom": ("base layer bottoms", "long johns", "thermal bottoms", "base layer pants"),
    "tee": ("t shirt", "tshirt", "tee", "graphic tee", "tee shirt"),
    "hiking_shirt": ("hiking shirt", "sun shirt", "sun hoodie", "button up shirt"),
    "hiking_boots": ("hiking boots", "hiking boot", "trekking boots", "walking boots", "backpacking boots", "hikers"),
    "winter_boots": ("winter boots", "snow boots", "insulated boots"),
    "trail_runners": ("trail running shoes", "trail runners", "trail shoes", "trail running shoe"),
    "approach_shoes": ("approach shoes",),
    "sandals": ("sandals", "sport sandals", "river sandals", "flip flops"),
    "tents": ("tent", "backpacking tent", "camping tent", "2 person tent", "dome tent"),
    "sleeping_bags": ("sleeping bag", "mummy bag", "down sleeping bag", "sleeping bags"),
    "sleeping_pads": ("sleeping pad", "sleeping mat", "air mattress", "camping mat", "foam pad"),
    "stoves": ("camp stove", "backpacking stove", "camping stove", "canister stove", "burner"),
    "cookware": ("cook set", "mess kit", "camp mug", "cook pot", "pots and pans"),
    "chairs": ("camp chair", "camping chair", "folding chair"),
    "daypacks": ("daypack", "day pack", "hiking daypack"),
    "backpacking_packs": ("backpacking pack", "trekking pack", "rucksack", "backpacking backpack"),
    "hydration_packs": ("hydration pack", "hydration vest", "water pack"),
    "harnesses": ("climbing harness", "harness"),
    "ropes": ("climbing rope", "dynamic rope", "rope"),
    "helmets": ("climbing helmet",),
    "climbing_shoes": ("climbing shoes", "climbing shoe"),
    "gloves": ("gloves", "mittens", "liner gloves"),
    "hats": ("beanie", "sun hat", "winter hat", "toque", "cap"),
    "socks": ("hiking socks", "wool socks", "crew socks", "socks"),
    "headlamps": ("headlamp", "head lamp", "head torch", "rechargeable headlamp"),
    "bottles": ("water bottle", "insulated bottle", "thermos", "water flask"),
    "filters": ("water filter", "water purifier"),
    "poles": ("trekking poles", "hiking poles", "walking sticks", "trekking sticks"),
    "gear_care": ("tent repair kit", "seam sealer", "boot wax", "waterproofing"),
}
MID_PHRASES = {"Jackets": ("jacket", "jackets", "coat"), "Pants": ("pants",), "Base Layers": ("base layer", "thermals"),
               "Shirts": ("shirt", "shirts"), "Boots": ("boots",), "Shoes": ("shoes",),
               "Packs": ("backpack", "backpacks", "pack"), "Climbing Gear": ("climbing gear",)}
NULL_PHRASES = ("hammock", "kayak", "snowshoes", "crampons", "ski goggles", "bear canister", "camp pillow",
                "snow pants", "bike shorts", "avalanche beacon", "fishing rod", "paddle board", "ice axe", "bivy sack",
                "dog leash", "trekking umbrella")
PHRASE_MATERIAL = {"down": "down", "wool": "wool"}  # implied by the phrase itself ("down jacket", "wool socks")
GENDER_FORMS = {"men": ("mens", "men's", "men", "mens"), "women": ("womens", "women's", "women", "ladies"),
                "kids": ("kids", "kids'", "youth", "boys", "girls")}
GENDER_SUFFIX = {"men": "for men", "women": "for women", "kids": "for kids"}
COLOR_FORMS = {"gray": ("gray", "grey", "charcoal"), "green": ("green", "olive"), "brown": ("brown", "tan"),
               "navy": ("navy", "dark blue"), "red": ("red", "maroon")}
QUERY_COLORS = ("black", "blue", "gray", "green", "red", "orange", "navy", "brown", "white", "yellow", "purple", "pink")
NICE_PRICES = (20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 175, 200, 250, 300, 350, 400, 500)
KEYBOARD = "qwertyuiop asdfghjkl zxcvbnm"


@dataclass
class QueryLog:
    strings: list[str]  # distinct query strings
    intents: list[Intent]  # hidden intent per distinct string
    typo_of: list[str | None]  # clean string if this string is a typo variant
    searches: np.ndarray  # index into strings, one per logged search
    counts: np.ndarray  # searches per distinct string

    def segment(self, qi: int) -> str:
        c = self.counts[qi]
        return "head" if c >= HEAD_MIN else "torso" if c >= TORSO_MIN else "tail"


HEAD_MIN, TORSO_MIN = 100, 10


def _nice_price(rng: np.random.Generator, base: float, low: float = 0.6, high: float = 1.6) -> int:
    target = base * float(rng.uniform(low, high))
    return min(NICE_PRICES, key=lambda p: abs(p - target))


def _render(rng: np.random.Generator, phrase: str, it: Intent, implied_material: bool) -> str:
    pre: list[str] = []
    post: list[str] = []
    if it.gender:
        if rng.random() < 0.25:
            post.append(GENDER_SUFFIX[it.gender])
        else:
            forms = GENDER_FORMS[it.gender]
            pre.append(forms[int(rng.integers(len(forms)))])
    if it.color:
        forms = COLOR_FORMS.get(it.color, (it.color,))
        pre.append(forms[int(rng.integers(len(forms)))])
    if it.material and not implied_material:
        pre.append("merino" if it.material == "merino wool" and rng.random() < 0.6 else it.material)
    if it.waterproof_min:
        pre.append("waterproof" if it.waterproof_min == 1 else f"{it.waterproof_min // 1000}k waterproof")
    rng.shuffle(pre)
    if it.size:
        if it.size.isdigit():
            post.append(f"size {it.size}")
        else:
            word = {"S": "small", "M": "medium", "L": "large"}.get(it.size)
            if it.size in ("XS", "XL", "XXL") and rng.random() < 0.4:
                post.append(it.size.lower())
            else:
                post.append(f"size {word if word and rng.random() < 0.4 else it.size.lower()}")
    if it.price_min is not None and it.price_max is not None:
        lo, hi = int(it.price_min), int(it.price_max)
        post.append(f"between {lo} and {hi}" if rng.random() < 0.5 else f"{lo}-{hi}")
    elif it.price_max is not None:
        p = int(it.price_max)
        post.append(str(rng.choice([f"under {p}", f"under ${p}", f"below {p}", f"less than {p}"])))
    elif it.price_min is not None:
        post.append(f"over {int(it.price_min)}")
    rng.shuffle(post)
    head = [it.brand.lower()] if it.brand and phrase else []
    return " ".join(head + pre + ([phrase] if phrase else [it.brand.lower()]) + post)


def _sample_attrs(rng: np.random.Generator, leaf_key: str | None, k: int, brand_leaves: dict[str, set[str]]) -> dict:
    leaf = LEAF_BY_KEY[leaf_key] if leaf_key else None
    options = ["color", "price", "brand"]
    if leaf is None or leaf.gendered:
        options.append("gender")
    if leaf is not None and leaf.size_kind != "none":
        options.append("size")
    if leaf is not None and len(leaf.materials) > 1:
        options.append("material")
    if leaf is None or leaf.waterproof != "never" or rng.random() < 0.05:
        options.append("waterproof")
    chosen = [options[j] for j in rng.permutation(len(options))[:k]]
    out: dict = {}
    base = leaf.price if leaf else 150
    for slot in chosen:
        if slot == "color":
            out["color"] = QUERY_COLORS[_pick(rng, _COLOR_CUM)]
        elif slot == "gender":
            out["gender"] = ("men", "women", "kids")[_pick(rng, _GENDER_CUM)]
        elif slot == "size":
            if leaf.size_kind == "shoe":
                out["size"] = str(int(rng.integers(6, 13)))
            else:
                out["size"] = ("XS", "S", "M", "L", "XL", "XXL")[_pick(rng, _SIZE_CUM)]
        elif slot == "material":
            out["material"] = leaf.materials[int(rng.integers(len(leaf.materials)))]
        elif slot == "waterproof":
            out["waterproof_min"] = 1 if rng.random() < 0.85 else int(rng.choice([10000, 20000]))
        elif slot == "brand":
            makers = [b for b in BRAND_NAMES if leaf_key in brand_leaves.get(b, set())] if leaf_key else BRAND_NAMES
            pool = makers if makers and rng.random() < 0.85 else BRAND_NAMES
            out["brand"] = pool[int(rng.integers(len(pool)))]
        elif slot == "price":
            r = rng.random()
            if r < 0.75:
                out["price_max"] = float(_nice_price(rng, base, 0.15 if rng.random() < 0.1 else 0.6))
            elif r < 0.9:
                lo = _nice_price(rng, base, 0.5, 0.9)
                hi = min((p for p in NICE_PRICES if p > lo), default=lo * 2)
                hi = max(hi, _nice_price(rng, base, 1.0, 1.6))
                out["price_min"], out["price_max"] = float(lo), float(hi)
            else:
                out["price_min"] = float(_nice_price(rng, base, 1.0, 1.5))
    return out


_COLOR_P = np.array([0.2, 0.16, 0.12, 0.11, 0.08, 0.06, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
_COLOR_CUM = np.cumsum(_COLOR_P / _COLOR_P.sum())
_GENDER_CUM = np.cumsum([0.47, 0.47, 0.06])
_SIZE_CUM = np.cumsum([0.05, 0.2, 0.3, 0.25, 0.15, 0.05])


def _pick(rng: np.random.Generator, cum: np.ndarray) -> int:
    return min(int(np.searchsorted(cum, rng.random() * cum[-1])), len(cum) - 1)


def _typo(rng: np.random.Generator, q: str) -> str | None:
    words = q.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return None
    i = idx[int(rng.integers(len(idx)))]
    w = words[i]
    op = rng.integers(4)
    j = int(rng.integers(1, len(w)))
    letter = KEYBOARD.replace(" ", "")[int(rng.integers(26))]
    if op == 0:
        w = w[:j] + w[j + 1:]  # deletion
    elif op == 1:
        w = w[:j] + letter + w[j:]  # insertion
    elif op == 2:
        w = w[:j] + letter + w[j + 1:]  # substitution
    elif j < len(w) - 1:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # transposition
    else:
        w = w[:j - 1] + w[j] + w[j - 1]
    if w == words[i]:
        return None
    words[i] = w
    return " ".join(words)


def generate_query_log(products: list[Product], n_searches: int = 50_000, seed: int = 23,
                       typo_rate: float = 0.07) -> QueryLog:
    rng = np.random.default_rng(seed)
    brand_leaves: dict[str, set[str]] = {}
    for p in products:
        brand_leaves.setdefault(p.brand, set()).add(p.leaf)
    candidates: list[tuple[float, str, Intent]] = []  # (popularity rank key, string, intent)

    def add(key: float, phrase: str, leaf: str | None, mid: str | None, attrs: dict):
        implied = next((m for w, m in PHRASE_MATERIAL.items() if phrase.startswith(w + " ")), None)
        if implied and "material" not in attrs:
            attrs = attrs | {"material": implied}
        it = Intent(leaf=leaf, mid=mid, **attrs)
        candidates.append((key + float(rng.normal(0, 0.3)), _render(rng, phrase, it, implied is not None), it))

    for lf in LEAVES:
        for pos, ph in enumerate(LEAF_PHRASES[lf.key]):
            add(0.3 * pos, ph, lf.key, None, {})
    for mid, phrases in MID_PHRASES.items():
        for ph in phrases:
            add(0.4, ph, None, mid, {})
    for b in BRAND_NAMES:
        add(1.2, "", None, None, {"brand": b})
    for ph in NULL_PHRASES:
        add(1.6, ph, None, None, {})
    leaf_keys = [lf.key for lf in LEAVES]
    weights = np.array([lf.share for lf in LEAVES])
    weights = weights / weights.sum()
    for k, n_draw, key in ((1, 2600, 1.0), (2, 3500, 1.9), (3, 2400, 2.4), (4, 700, 2.8)):
        for _ in range(n_draw):
            r = rng.random()
            if r < 0.85:
                leaf = leaf_keys[int(rng.choice(len(leaf_keys), p=weights))]
                phrases = LEAF_PHRASES[leaf]
                ph = phrases[min(int(rng.geometric(0.45)) - 1, len(phrases) - 1)]
                add(key, ph, leaf, None, _sample_attrs(rng, leaf, k, brand_leaves))
            elif r < 0.97:
                mid = list(MID_PHRASES)[int(rng.integers(len(MID_PHRASES)))]
                ph = MID_PHRASES[mid][int(rng.integers(len(MID_PHRASES[mid])))]
                add(key, ph, None, mid, _sample_attrs(rng, None, k, brand_leaves))
            else:
                add(key, NULL_PHRASES[int(rng.integers(len(NULL_PHRASES)))], None, None,
                    {k2: v for k2, v in _sample_attrs(rng, None, k, brand_leaves).items() if k2 != "brand"})
    candidates.sort(key=lambda t: t[0])
    strings: list[str] = []
    intents: list[Intent] = []
    seen: set[str] = set()
    for _, s, it in candidates:
        if s not in seen:
            seen.add(s)
            strings.append(s)
            intents.append(it)
    n_clean = len(strings)
    zipf = 1.0 / (np.arange(1, n_clean + 1) + 6.0) ** 1.08  # Zipf-Mandelbrot: a fat head, not one giant query
    zipf /= zipf.sum()
    draws = rng.choice(n_clean, size=n_searches, p=zipf)
    typo_of: list[str | None] = [None] * n_clean
    index = {s: i for i, s in enumerate(strings)}
    searches = np.empty(n_searches, dtype=np.int32)
    for j, qi in enumerate(draws):
        if rng.random() < typo_rate:
            t = _typo(rng, strings[qi])
            if t is not None:
                if t not in index:
                    index[t] = len(strings)
                    strings.append(t)
                    intents.append(intents[qi])
                    typo_of.append(strings[qi])
                searches[j] = index[t]
                continue
        searches[j] = qi
    counts = np.bincount(searches, minlength=len(strings))
    return QueryLog(strings=strings, intents=intents, typo_of=typo_of, searches=searches, counts=counts)


def token_counts(log: QueryLog) -> Counter:
    """Token frequencies over all logged searches (the frequency prior for spelling)."""
    c: Counter = Counter()
    for qi, n in enumerate(log.counts):
        if n:
            for t in log.strings[qi].replace("'", "").split():
                c[t] += int(n)
    return c


def select_judged(log: QueryLog, n: int = 600, seed: int = 5) -> list[int]:
    """Stratified sample of distinct strings for relevance judgments: head, torso, tail, and typo variants."""
    rng = np.random.default_rng(seed)
    pools: dict[str, list[int]] = {"head": [], "torso": [], "tail": [], "typo": []}
    for qi in range(len(log.strings)):
        if log.counts[qi] == 0:
            continue
        pools["typo" if log.typo_of[qi] else log.segment(qi)].append(qi)
    quota = {"head": int(n * 0.2), "torso": int(n * 0.3), "typo": int(n * 0.2)}
    quota["tail"] = n - sum(quota.values())
    chosen: list[int] = []
    for seg in ("head", "torso", "typo", "tail"):
        pool = pools[seg]
        take = min(quota[seg], len(pool))
        chosen += [int(x) for x in rng.choice(pool, size=take, replace=False)]
    if len(chosen) < n:  # small logs: top up from whatever is left
        rest = [qi for qi in range(len(log.strings)) if log.counts[qi] and qi not in set(chosen)]
        chosen += [int(x) for x in rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False)]
    return sorted(chosen)
