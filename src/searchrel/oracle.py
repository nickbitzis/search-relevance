"""The hidden relevance model. It plays the role of trained human judges: given the shopper's true intent, it grades
any product 0 (irrelevant), 1 (related), 2 (relevant, something is off), or 3 (exactly right).

Search code never imports this module. Only data generation, judgments, and the click simulator do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributes import color_values, material_values
from .catalog import BRAND_NAMES, COLORS, GENDERS, LEAF_BY_KEY, LEAF_KEYS, MATERIALS, MIDS, SIZE_BIT, CatalogArrays


@dataclass(frozen=True)
class Intent:
    """What the shopper actually wants. Category is a leaf, a mid level node, or neither (brand only or no match)."""

    leaf: str | None = None
    mid: str | None = None
    brand: str | None = None
    gender: str | None = None
    color: str | None = None
    size: str | None = None
    material: str | None = None
    waterproof_min: int | None = None
    price_min: float | None = None
    price_max: float | None = None

    @property
    def n_attributes(self) -> int:
        return sum(v is not None for v in (self.gender, self.color, self.size, self.material, self.waterproof_min,
                                           self.price_min, self.price_max))

    @property
    def category_mid(self) -> str | None:
        return LEAF_BY_KEY[self.leaf].mid if self.leaf else self.mid


def grades(intent: Intent, a: CatalogArrays) -> np.ndarray:
    """Graded relevance of every product for one intent, as an int8 array."""
    if intent.leaf:
        li = LEAF_KEYS.index(intent.leaf)
        mi = MIDS.index(LEAF_BY_KEY[intent.leaf].mid)
        topical = np.where(a.leaf == li, 3, np.where((a.mid == mi) | (a.refers == li), 1, 0))
    elif intent.mid:
        topical = np.where(a.mid == MIDS.index(intent.mid), 3, 0)
    elif intent.brand:
        topical = np.where(a.brand == BRAND_NAMES.index(intent.brand), 3, 0)
    else:
        return np.zeros(a.n, dtype=np.int8)
    pen = np.zeros(a.n, dtype=np.int16)
    if intent.brand and (intent.leaf or intent.mid):
        pen += 2 * (a.brand != BRAND_NAMES.index(intent.brand))
    if intent.gender:
        unisex = a.gender == GENDERS.index("unisex")
        same = a.gender == GENDERS.index(intent.gender)
        pen += np.where(same, 0, np.where(unisex, 1 if intent.gender == "kids" else 0, 2))
    if intent.color:
        ok = np.isin(a.color, [COLORS.index(c) for c in color_values(intent.color)])
        pen += ~ok
    if intent.size:
        sized = a.size_mask != 0
        pen += sized & ((a.size_mask & SIZE_BIT[intent.size]) == 0)
    if intent.material:
        ok = np.isin(a.material, [MATERIALS.index(m) for m in material_values(intent.material)])
        pen += ~ok
    if intent.waterproof_min:
        pen += np.where(a.waterproof >= intent.waterproof_min, 0, np.where(a.waterproof > 0, 1, 2))
    if intent.price_max is not None:
        pen += np.where(a.price > intent.price_max * 1.1, 2, np.where(a.price > intent.price_max, 1, 0))
    if intent.price_min is not None:
        pen += a.price < intent.price_min
    pen += ~a.in_stock  # judges were told an out of stock item is less useful
    pen += a.rating < 3.0
    g = np.where(topical > 0, np.clip(topical - pen, 0, 3), 0)
    return g.astype(np.int8)
