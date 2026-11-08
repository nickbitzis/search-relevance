"""Seeded catalog for Tamarack Row Outfitters, a made-up outdoor gear retailer.

About 20,000 products across a three level category tree (top, mid, leaf), 24 invented brands, and the attributes a
real gear catalog carries: color, sizes, material, gender, waterproof rating, weight, price, rating, reviews,
inventory, and margin. Every product also has a latent quality score that drives rating, reviews, and sales; the
search stack only ever sees the observable fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Leaf:
    key: str
    top: str
    mid: str
    name: str
    nouns: tuple[str, ...]  # title nouns; the first is the canonical one
    price: float
    weight: tuple[int, int]
    gendered: bool
    size_kind: str  # apparel, shoe, none
    waterproof: str  # always, some, never
    materials: tuple[str, ...]
    share: float = 1.0  # relative number of products


LEAVES: tuple[Leaf, ...] = (
    Leaf("rain_jacket", "Clothing", "Jackets", "Rain Jackets", ("Rain Jacket", "Rain Jacket", "Waterproof Jacket"),
         180, (250, 600), True, "apparel", "always", ("nylon", "polyester", "recycled nylon"), 2.0),
    Leaf("insulated_jacket", "Clothing", "Jackets", "Insulated Jackets",
         ("Down Jacket", "Insulated Jacket", "Synthetic Insulated Jacket"), 220, (300, 900), True, "apparel", "some",
         ("down", "synthetic", "polyester"), 2.0),
    Leaf("fleece_jacket", "Clothing", "Jackets", "Fleece Jackets",
         ("Fleece Jacket", "Fleece Hoodie", "Fleece Pullover"),
         90, (250, 600), True, "apparel", "never", ("fleece", "polyester", "recycled polyester"), 1.6),
    Leaf("softshell_jacket", "Clothing", "Jackets", "Softshell Jackets", ("Softshell Jacket",), 140, (300, 600), True,
         "apparel", "some", ("nylon", "polyester", "stretch nylon"), 1.0),
    Leaf("hiking_pants", "Clothing", "Pants", "Hiking Pants", ("Hiking Pant", "Trail Pant", "Convertible Pant"), 80,
         (250, 450), True, "apparel", "never", ("nylon", "polyester", "cotton"), 1.6),
    Leaf("rain_pants", "Clothing", "Pants", "Rain Pants", ("Rain Pant",), 110, (200, 400), True, "apparel", "always",
         ("nylon", "polyester"), 0.8),
    Leaf("tights", "Clothing", "Pants", "Tights", ("Tights", "Running Tights"), 70, (150, 300), True, "apparel",
         "never", ("polyester", "merino wool", "nylon"), 0.8),
    Leaf("base_top", "Clothing", "Base Layers", "Base Layer Tops", ("Base Layer Top", "Base Layer Crew", "Thermal Top"),
         75, (150, 300), True, "apparel", "never", ("merino wool", "polyester", "synthetic"), 1.2),
    Leaf("base_bottom", "Clothing", "Base Layers", "Base Layer Bottoms", ("Base Layer Bottom", "Thermal Bottom"), 70,
         (150, 300), True, "apparel", "never", ("merino wool", "polyester", "synthetic"), 0.9),
    Leaf("tee", "Clothing", "Shirts", "T-Shirts", ("Tee", "T-Shirt", "Graphic Tee"), 35, (120, 250), True, "apparel",
         "never", ("cotton", "merino wool", "polyester"), 1.4),
    Leaf("hiking_shirt", "Clothing", "Shirts", "Hiking Shirts", ("Hiking Shirt", "Sun Shirt", "Button-Up Shirt"), 65,
         (150, 300), True, "apparel", "never", ("nylon", "polyester", "cotton"), 1.0),
    Leaf("hiking_boots", "Footwear", "Boots", "Hiking Boots", ("Hiking Boot", "Mid Hiking Boot", "Backpacking Boot"),
         170, (900, 1600), True, "shoe", "some", ("leather", "nylon", "suede"), 1.6),
    Leaf("winter_boots", "Footwear", "Boots", "Winter Boots", ("Winter Boot", "Snow Boot", "Insulated Boot"), 160,
         (1000, 1900), True, "shoe", "always", ("leather", "nylon", "rubber"), 0.9),
    Leaf("trail_runners", "Footwear", "Shoes", "Trail Running Shoes", ("Trail Running Shoe", "Trail Runner"), 140,
         (500, 800), True, "shoe", "some", ("mesh", "nylon"), 1.4),
    Leaf("approach_shoes", "Footwear", "Shoes", "Approach Shoes", ("Approach Shoe",), 150, (600, 900), True, "shoe",
         "never", ("suede", "leather", "mesh"), 0.5),
    Leaf("sandals", "Footwear", "Shoes", "Sandals", ("Sandal", "Sport Sandal", "River Sandal"), 70, (300, 700), True,
         "shoe", "never", ("rubber", "nylon", "leather"), 0.8),
    Leaf("tents", "Camping", "Shelter", "Tents", ("Tent", "Backpacking Tent", "Camping Tent", "Dome Tent"), 300,
         (1200, 4000), False, "none", "always", ("nylon", "polyester", "ripstop nylon"), 1.3),
    Leaf("sleeping_bags", "Camping", "Sleep", "Sleeping Bags", ("Sleeping Bag", "Mummy Sleeping Bag"), 220,
         (600, 1800), False, "none", "some", ("down", "synthetic"), 1.1),
    Leaf("sleeping_pads", "Camping", "Sleep", "Sleeping Pads", ("Sleeping Pad", "Air Pad", "Foam Pad"), 110,
         (300, 900), False, "none", "never", ("nylon", "foam"), 0.8),
    Leaf("stoves", "Camping", "Kitchen", "Camp Stoves", ("Camp Stove", "Backpacking Stove", "Canister Stove"), 80,
         (80, 2500), False, "none", "never", ("aluminum", "steel", "titanium"), 0.6),
    Leaf("cookware", "Camping", "Kitchen", "Cookware", ("Cook Set", "Cook Pot", "Camp Mug"), 45, (100, 700), False,
         "none", "never", ("aluminum", "titanium", "steel"), 0.7),
    Leaf("chairs", "Camping", "Furniture", "Camp Chairs", ("Camp Chair", "Folding Chair"), 70, (500, 2500), False,
         "none", "never", ("aluminum", "polyester", "steel"), 0.5),
    Leaf("daypacks", "Packs", "Packs", "Daypacks", ("Daypack", "Hiking Daypack"), 90, (400, 1000), False, "none",
         "some", ("nylon", "polyester", "recycled nylon"), 1.2),
    Leaf("backpacking_packs", "Packs", "Packs", "Backpacking Packs", ("Backpacking Pack", "Trekking Pack"), 240,
         (1200, 2400), True, "none", "some", ("nylon", "polyester"), 0.9),
    Leaf("hydration_packs", "Packs", "Packs", "Hydration Packs", ("Hydration Pack", "Hydration Vest"), 90, (200, 600),
         False, "none", "never", ("nylon", "polyester"), 0.6),
    Leaf("harnesses", "Climbing", "Climbing Gear", "Climbing Harnesses", ("Climbing Harness",), 75, (300, 450), True,
         "apparel", "never", ("nylon", "polyester"), 0.5),
    Leaf("ropes", "Climbing", "Climbing Gear", "Climbing Ropes", ("Climbing Rope", "Dynamic Rope"), 200, (3000, 4500),
         False, "none", "some", ("nylon",), 0.4),
    Leaf("helmets", "Climbing", "Climbing Gear", "Climbing Helmets", ("Climbing Helmet",), 80, (200, 400), False,
         "none", "never", ("polycarbonate", "foam"), 0.4),
    Leaf("climbing_shoes", "Climbing", "Climbing Gear", "Climbing Shoes", ("Climbing Shoe",), 150, (400, 600), True,
         "shoe", "never", ("leather", "synthetic", "suede"), 0.5),
    Leaf("gloves", "Accessories", "Clothing Accessories", "Gloves", ("Glove", "Liner Glove", "Mitten"), 45, (50, 250),
         True, "apparel", "some", ("fleece", "leather", "wool", "polyester"), 1.0),
    Leaf("hats", "Accessories", "Clothing Accessories", "Hats", ("Beanie", "Sun Hat", "Cap"), 30, (50, 150), False,
         "none", "never", ("wool", "cotton", "polyester", "merino wool"), 1.0),
    Leaf("socks", "Accessories", "Clothing Accessories", "Socks", ("Hiking Sock", "Wool Sock", "Crew Sock"), 22,
         (40, 120), True, "apparel", "never", ("merino wool", "wool", "nylon"), 1.1),
    Leaf("headlamps", "Accessories", "Electronics", "Headlamps", ("Headlamp", "Rechargeable Headlamp"), 45, (50, 200),
         False, "none", "some", ("plastic", "aluminum"), 0.6),
    Leaf("bottles", "Accessories", "Water", "Water Bottles", ("Water Bottle", "Insulated Bottle"), 30, (150, 500),
         False, "none", "never", ("steel", "plastic", "aluminum"), 0.7),
    Leaf("filters", "Accessories", "Water", "Water Filters", ("Water Filter", "Water Purifier"), 60, (50, 400), False,
         "none", "never", ("plastic",), 0.4),
    Leaf("poles", "Accessories", "Trekking Poles", "Trekking Poles", ("Trekking Pole", "Hiking Pole"), 110,
         (200, 600), False, "none", "never", ("aluminum", "carbon"), 0.5),
    Leaf("gear_care", "Accessories", "Gear Care", "Gear Care and Repair", ("Repair Kit",), 18, (30, 300), False,
         "none", "never", ("plastic",), 1.0),
)
LEAF_BY_KEY = {lf.key: lf for lf in LEAVES}
LEAF_KEYS = [lf.key for lf in LEAVES]
MIDS = sorted({lf.mid for lf in LEAVES})
TOPS = sorted({lf.top for lf in LEAVES})

# Gear care items name the product they care for, which is a classic BM25 trap: "Rain Jacket Repair Tape" contains
# every word of "rain jacket" and is not what the shopper wants.
GEAR_CARE = (
    ("rain_jacket", "Rain Jacket Wash-In Waterproofing"), ("rain_jacket", "Rain Jacket Repair Tape"),
    ("insulated_jacket", "Down Jacket Wash"), ("tents", "Tent Seam Sealer"), ("tents", "Tent Repair Kit"),
    ("tents", "Tent Footprint"), ("hiking_boots", "Hiking Boot Wax"), ("hiking_boots", "Hiking Boot Laces"),
    ("sleeping_pads", "Sleeping Pad Repair Kit"), ("sleeping_bags", "Sleeping Bag Liner"),
    ("stoves", "Stove Fuel Canister"), ("ropes", "Climbing Rope Bag"), ("daypacks", "Daypack Rain Cover"),
)

BRANDS: tuple[tuple[str, float], ...] = (  # name, price tier
    ("Quillback", 1.45), ("Halvard", 1.3), ("Brackenfell", 1.0), ("Larchwood", 0.95), ("Scree Supply", 0.8),
    ("Eskerline", 1.15), ("Moraine Co", 0.9), ("Cairnworth", 1.25), ("Fjeldmark", 1.5), ("Owlbrook", 0.75),
    ("Pikaworks", 0.85), ("Ravelin", 1.1), ("Glissade", 1.2), ("Talusworks", 1.0), ("Verglas", 1.4),
    ("Corrie and Tarn", 1.05), ("Firnline", 1.35), ("Kettlehole", 0.8), ("Duneholt", 0.9), ("Marrowstone", 1.0),
    ("Tinderbox Trail", 0.7), ("Graupel", 1.15), ("Sastrugi", 1.3), ("Lochan", 0.95),
)
BRAND_NAMES = [b for b, _ in BRANDS]

COLORS = ("black", "blue", "gray", "green", "red", "orange", "navy", "brown", "white", "yellow", "purple", "pink")
COLOR_WEIGHTS = (0.2, 0.14, 0.13, 0.12, 0.08, 0.06, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02)
GENDERS = ("men", "women", "kids", "unisex")
GENDER_LABEL = {"men": "Men's", "women": "Women's", "kids": "Kids'", "unisex": ""}
MATERIALS = sorted({m for lf in LEAVES for m in lf.materials})
APPAREL_SIZES = ("XS", "S", "M", "L", "XL", "XXL")
SHOE_SIZES = tuple(str(s) for s in range(5, 15))
SIZES = APPAREL_SIZES + SHOE_SIZES
SIZE_BIT = {s: 1 << i for i, s in enumerate(SIZES)}

NOUN_MATERIAL = {"Down": "down", "Synthetic": "synthetic", "Wool": "wool", "Foam": "foam", "Fleece": "fleece"}

MODELS = (
    "Cascade", "Aspen", "Tundra", "Ridge", "Granite", "Ember", "Alpine", "Summit", "Vista", "Talon", "Drift", "Nimbus",
    "Zephyr", "Boulder", "Glacier", "Sierra", "Juniper", "Canyon", "Mesa", "Fjord", "Basin", "Crest", "Pinnacle",
    "Haven", "Torrent", "Gale", "Squall", "Rime", "Moss", "Larch", "Spruce", "Cedar", "Alder", "Hemlock", "Birch",
    "Quartz", "Flint", "Shale", "Basalt", "Obsidian", "Comet", "Orbit", "Arc", "Vector", "Pulse", "Trek", "Rover",
    "Nomad", "Scout", "Ranger", "Voyager", "Trailhead", "Highpoint", "Overlook", "Traverse", "Ascent", "Pioneer",
    "Horizon", "Kestrel Point", "Bramble",
)
SUFFIXES = ("", "", "", " 2", " Pro", " Lite", " UL", " Plus", " II", " Hybrid")
USES = ("day hikes", "backpacking trips", "alpine starts", "wet shoulder seasons", "trail running", "car camping",
        "travel", "everyday wear", "cold mornings", "long approaches")


@dataclass
class Product:
    id: str
    title: str
    brand: str
    leaf: str
    mid: str
    top: str
    color: str
    sizes: tuple[str, ...]
    material: str
    gender: str
    waterproof_mm: int
    weight_g: int
    price: float
    rating: float
    reviews: int
    inventory: int
    margin: float
    units_90d: int
    description: str
    refers_to: str = ""  # gear care: the leaf this item is for
    quality: float = field(default=0.0, repr=False)  # latent, hidden from search

    @property
    def category_path(self) -> str:
        return f"{self.top} > {self.mid} > {LEAF_BY_KEY[self.leaf].name}"

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "brand": self.brand, "category": self.category_path,
                "leaf": self.leaf, "color": self.color, "sizes": list(self.sizes), "material": self.material,
                "gender": self.gender, "waterproof_mm": self.waterproof_mm, "weight_g": self.weight_g,
                "price": self.price, "rating": self.rating, "reviews": self.reviews, "in_stock": self.inventory > 0,
                "inventory": self.inventory}


def _sizes(rng: np.random.Generator, leaf: Leaf, gender: str) -> tuple[str, ...]:
    if leaf.size_kind == "apparel":
        base = APPAREL_SIZES[1:4] if gender == "kids" else APPAREL_SIZES
        lo = int(rng.integers(0, 2))
        hi = len(base) - int(rng.integers(0, 2))
        sizes = [s for s in base[lo:hi] if rng.random() > 0.12]  # some sizes sold out
        return tuple(sizes or base[lo:lo + 1])
    if leaf.size_kind == "shoe":
        start = {"women": 0, "men": 2, "kids": 0}.get(gender, 1)
        span = SHOE_SIZES[start:start + 8]
        return tuple(s for s in span if rng.random() > 0.15) or span[:1]
    return ()


def _description(rng: np.random.Generator, leaf: Leaf, p: Product) -> str:
    noun = leaf.nouns[0].lower()
    parts = [f"A {p.material} {noun} built for {USES[int(rng.integers(len(USES)))]}."]
    if p.waterproof_mm:
        parts.append(f"Waterproof to {p.waterproof_mm} mm with taped seams.")
    parts.append(f"Weighs {p.weight_g} g.")
    if rng.random() < 0.35:  # cross-sell copy mentions other categories, which pollutes the description field
        other = LEAVES[int(rng.integers(len(LEAVES) - 1))]
        parts.append(f"Pairs well with our {other.nouns[0].lower()}s.")
    if rng.random() < 0.3:
        parts.append(f"Available in {p.color} and other colors.")
    return " ".join(parts)


def generate_catalog(n: int = 20_000, seed: int = 11) -> list[Product]:
    rng = np.random.default_rng(seed)
    shares = np.array([lf.share for lf in LEAVES])
    shares = shares / shares.sum()
    # Each brand makes 8 to 20 leaves (gear care is made by four of them).
    makes = {b: set(rng.choice(LEAF_KEYS[:-1], size=int(rng.integers(8, 21)), replace=False)) for b in BRAND_NAMES}
    for b in BRAND_NAMES[:4]:
        makes[b].add("gear_care")
    brand_pop = {b: rng.normal(0, 0.5) for b in BRAND_NAMES}
    tier = dict(BRANDS)
    leaf_idx = rng.choice(len(LEAVES), size=n, p=shares)
    makers = {lf.key: [b for b in BRAND_NAMES if lf.key in makes[b]] or BRAND_NAMES for lf in LEAVES}
    qs = rng.normal(0, 1, size=n)
    gender_draw = rng.choice(4, size=n, p=[0.42, 0.42, 0.06, 0.10])
    color_draw = rng.choice(len(COLORS), size=n, p=np.array(COLOR_WEIGHTS) / sum(COLOR_WEIGHTS))
    products: list[Product] = []
    for i in range(n):
        leaf = LEAVES[leaf_idx[i]]
        brands = makers[leaf.key]
        brand = brands[int(rng.integers(len(brands)))]
        q = float(qs[i])
        gender = GENDERS[int(gender_draw[i])] if leaf.gendered else "unisex"
        color = COLORS[int(color_draw[i])]
        material = leaf.materials[int(rng.integers(len(leaf.materials)))]
        refers = ""
        if leaf.key == "gear_care":
            refers, noun = GEAR_CARE[int(rng.integers(len(GEAR_CARE)))]
            title = f"{brand} {noun}"
        else:
            noun = leaf.nouns[int(rng.integers(len(leaf.nouns)))]
            for word, mat in NOUN_MATERIAL.items():  # a "Down Jacket" is made of down
                if word in noun:
                    material = mat
            model = MODELS[int(rng.integers(len(MODELS)))] + SUFFIXES[int(rng.integers(len(SUFFIXES)))]
            label = GENDER_LABEL[gender]
            title = f"{brand} {model} {label + ' ' if label else ''}{noun}"
            if rng.random() < 0.4:
                title += f", {color.title()}"
        if leaf.waterproof == "always":
            wp = int(rng.choice([10000, 15000, 20000, 28000]))
        elif leaf.waterproof == "some" and rng.random() < 0.5:
            wp = int(rng.choice([5000, 10000, 20000]))
        else:
            wp = 0
        price = leaf.price * tier[brand] * math.exp(rng.normal(0, 0.25)) * (1 + 0.12 * q)
        if gender == "kids":
            price *= 0.6
        price = max(4.95, round(price) - 0.05)
        rating = min(5.0, max(1.5, round(3.95 + 0.45 * q + rng.normal(0, 0.35), 1)))
        reviews = int(rng.poisson(math.exp(2.4 + 0.7 * q + brand_pop[brand])))
        units = int(rng.poisson(math.exp(1.6 + 0.9 * q + brand_pop[brand])))
        inventory = 0 if rng.random() < 0.12 else int(rng.poisson(25)) + 1
        margin = round(min(0.65, max(0.12, 0.2 + 0.18 * tier[brand] + rng.normal(0, 0.05))), 3)
        p = Product(id=f"p{i:05d}", title=title, brand=brand, leaf=leaf.key, mid=leaf.mid, top=leaf.top, color=color,
                    sizes=(), material=material, gender=gender, waterproof_mm=wp,
                    weight_g=int(rng.integers(*leaf.weight)), price=price, rating=rating, reviews=reviews,
                    inventory=inventory, margin=margin, units_90d=units, description="", refers_to=refers, quality=q)
        p.sizes = _sizes(rng, leaf, gender)
        p.description = _description(rng, leaf, p)
        products.append(p)
    return products


class CatalogArrays:
    """Column view of the catalog for vectorized filters and the hidden relevance model."""

    def __init__(self, products: list[Product]):
        li = {k: i for i, k in enumerate(LEAF_KEYS)}
        self.n = len(products)
        self.leaf = np.array([li[p.leaf] for p in products], dtype=np.int16)
        self.mid = np.array([MIDS.index(p.mid) for p in products], dtype=np.int16)
        self.brand = np.array([BRAND_NAMES.index(p.brand) for p in products], dtype=np.int16)
        self.color = np.array([COLORS.index(p.color) for p in products], dtype=np.int16)
        self.gender = np.array([GENDERS.index(p.gender) for p in products], dtype=np.int16)
        self.material = np.array([MATERIALS.index(p.material) for p in products], dtype=np.int16)
        self.waterproof = np.array([p.waterproof_mm for p in products], dtype=np.int32)
        self.price = np.array([p.price for p in products], dtype=np.float64)
        self.size_mask = np.array([sum(SIZE_BIT[s] for s in p.sizes) for p in products], dtype=np.int64)
        self.in_stock = np.array([p.inventory > 0 for p in products])
        self.rating = np.array([p.rating for p in products], dtype=np.float64)
        self.refers = np.array([li[p.refers_to] if p.refers_to else -1 for p in products], dtype=np.int16)
