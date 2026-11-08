import numpy as np

from searchrel.attributes import Filters
from searchrel.classifier import CategoryPrediction
from searchrel.features import FEATURE_INDEX, compute_features
from searchrel.retrieval import facets, filter_mask
from searchrel.spelling import SpellCorrector
from searchrel.understanding import QueryUnderstanding

from .conftest import build_retriever

QU = QueryUnderstanding()


def titles(products, r):
    return [products[i].title for i in r.ids]


def test_bm25_puts_the_jacket_above_the_repair_tape(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("rain jacket"))
    assert "Repair Tape" not in titles(tiny_catalog, r)[0]
    assert r.n_hits == 5  # three jackets, the tape, and the tent whose copy says "rain jackets"
    assert "Tent" not in titles(tiny_catalog, r)[0]


def test_filters_from_attributes(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("womens rain jacket"))
    assert titles(tiny_catalog, r)[0] == "Halvard Crest Women's Rain Jacket"
    assert all(tiny_catalog[i].gender in ("women", "unisex") for i in r.ids)
    r = R.retrieve(QU.parse("blue rain jacket"))
    assert {tiny_catalog[i].color for i in r.ids} == {"blue", "navy"}  # blue includes navy


def test_filter_mask_semantics(tiny_catalog):
    _, _, a = build_retriever(tiny_catalog)
    assert filter_mask(a, Filters(gender="men")).tolist() == [True, False, True, True, True, True]  # unisex passes
    assert filter_mask(a, Filters(size="XL")).tolist() == [False, False, True, False, True, True]  # sizeless pass
    assert filter_mask(a, Filters(price_max=150)).sum() == 2
    assert filter_mask(a, Filters(waterproof_min=1)).sum() == 4
    assert filter_mask(a, Filters(), ("mid", "Jackets")).sum() == 4


def test_fallback_relaxes_least_important_filter_first(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    # No red men's jacket in size XL: size goes first, then color.
    r = R.retrieve(QU.parse("mens red rain jacket xl"))
    assert r.zero_initial
    assert r.steps[:2] == ["relaxed size", "relaxed color"]
    assert r.n_hits > 0


def test_fallback_stops_as_soon_as_results_appear(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("orange rain jacket under 200"))
    assert r.steps == ["relaxed color"]
    assert r.filters.active() == {"price_max": 200.0}


def test_no_fallback_for_baseline(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QueryUnderstanding.baseline("kayak"), fallback=False)
    assert r.n_hits == 0 and r.steps == []


def test_fallback_order_spelling_then_category_then_any_term(tiny_catalog):
    speller = SpellCorrector({"tent": 500, "tint": 1, "rain": 100, "jacket": 100, "backpacking": 50})
    qu = QueryUnderstanding(speller, None)
    R, _, _ = build_retriever(tiny_catalog, reparse=lambda raw: qu.parse(raw, apply_suggestions=True))
    p = qu.parse("tint")
    assert p.suggestions and p.tokens == ["tint"]
    r = R.retrieve(p)
    assert r.steps == ["applied spelling suggestion tint -> tent"]
    assert titles(tiny_catalog, r) == ["Owlbrook Trek Backpacking Tent"]

    p = qu.parse("rain jacket")
    p.category = CategoryPrediction([("ropes", 0.9)], {"ropes": 0.9}, "leaf", "ropes")
    r = R.retrieve(p)
    assert r.steps == ["broadened category to Climbing Gear", "dropped category"]

    r = R.retrieve(qu.parse("rain kayak"))
    assert r.steps == ["matched any term"] and r.any_match


def test_explicit_filters_are_never_relaxed(tiny_catalog):
    R, _, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("rain jacket"), explicit=Filters(brand="Verglas"))
    assert r.steps == ["matched any term"]  # the fallback loosens the text, never the shopper's brand pick
    assert {tiny_catalog[i].brand for i in r.ids} == {"Verglas"}


def test_facets_count_every_hit(tiny_catalog):
    R, _, a = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("jacket"))
    f = facets(a, r.hits)
    assert sum(v["count"] for v in f["category"]) == r.n_hits
    assert f["brand"][0] == {"value": "Quillback", "count": 3}


def test_feature_computation(tiny_catalog):
    R, idx, _ = build_retriever(tiny_catalog)
    r = R.retrieve(QU.parse("quillback mens blue rain jacket under 200"))
    X = compute_features(r, tiny_catalog, idx, {"rain_jacket": 150.0, "gear_care": 12.0, "insulated_jacket": 260})
    rows = {tiny_catalog[i].id: X[k] for k, i in enumerate(r.ids)}
    best = rows["p00000"]  # Quillback, men, blue, 180, in stock
    assert best[FEATURE_INDEX["brand_match"]] == 1.0
    assert best[FEATURE_INDEX["attr_matches"]] == 3 and best[FEATURE_INDEX["attr_mismatches"]] == 0
    assert best[FEATURE_INDEX["price_ratio"]] == 180 / 200
    assert best[FEATURE_INDEX["in_stock"]] == 1.0
    assert best[FEATURE_INDEX["title_coverage"]] == 1.0
    assert rows["p00005"][FEATURE_INDEX["in_stock"]] == 0.0
    assert np.all(X[:, FEATURE_INDEX["bm25_total"]] >= 0)
    assert X[0, FEATURE_INDEX["bm25_rank_recip"]] == 1.0
