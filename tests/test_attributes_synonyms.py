import pytest

from searchrel.attributes import Filters, extract_price
from searchrel.synonyms import expand, find_synonyms
from searchrel.understanding import SYNONYM_WEIGHT, QueryUnderstanding

QU = QueryUnderstanding()


@pytest.mark.parametrize("text,lo,hi", [
    ("rain jacket under 200", None, 200), ("tent under $150", None, 150), ("boots below 100", None, 100),
    ("pack less than 80 dollars", None, 80), ("stove up to 60", None, 60), ("jacket between 100 and 200", 100, 200),
    ("jacket 100-200", 100, 200), ("tent $150 to $300", 150, 300), ("boots over 150", 150, None),
    ("2 person tent", None, None), ("2-3 person tent", None, None),
])
def test_price_phrasings(text, lo, hi):
    got_lo, got_hi, _, _ = extract_price(text)
    assert (got_lo, got_hi) == (lo, hi)


@pytest.mark.parametrize("query,expected", [
    ("mens blue waterproof jacket under 200",
     {"gender": "men", "color": "blue", "waterproof_min": 1, "price_max": 200.0}),
    ("men's rain jacket", {"gender": "men"}),
    ("rain jacket for women", {"gender": "women"}),
    ("ladies hiking boots", {"gender": "women"}),
    ("youth climbing harness", {"gender": "kids"}),
    ("grey fleece", {"color": "gray"}),
    ("dark blue down jacket", {"color": "navy"}),
    ("olive hiking pants", {"color": "green"}),
    ("hiking shirt size m", {"size": "M"}),
    ("base layer top size medium", {"size": "M"}),
    ("rain jacket xl", {"size": "XL"}),
    ("trail runners size 10", {"size": "10"}),
    ("merino base layer", {"material": "merino wool"}),
    ("wool socks", {"material": "wool"}),
    ("leather hiking boots", {"material": "leather"}),
    ("20k waterproof rain jacket", {"waterproof_min": 20000}),
    ("quillback rain jacket", {"brand": "Quillback"}),
    ("scree supply tent", {"brand": "Scree Supply"}),
    ("corrie and tarn fleece", {"brand": "Corrie and Tarn"}),
    ("womens merino base layer top size s under 90",
     {"gender": "women", "material": "merino wool", "size": "S", "price_max": 90.0}),
    ("tent", {}),
])
def test_attribute_extraction(query, expected):
    assert QU.parse(query).filters.active() == expected


def test_extracted_words_leave_the_text_query():
    p = QU.parse("mens blue waterproof jacket under 200")
    assert [c.text for c in p.concepts] == ["jacket"]


def test_filters_without_and_merged():
    f = Filters(color="blue", size="M")
    assert f.without("color").active() == {"size": "M"}
    assert f.merged(Filters(color="red")).active() == {"color": "red", "size": "M"}


def test_longest_synonym_wins():
    exp, n = find_synonyms(["waterproof", "shell", "mens"], 0)
    assert (exp.canonical, n) == ("rain jacket", 2)
    assert [e.canonical for e in expand(["puffy", "jacket", "head", "torch"])] == ["insulated jacket", "headlamp"]


def test_synonym_adds_alternative_with_lower_weight():
    p = QU.parse("waterproof shell")
    c = p.concepts[0]
    assert c.alternatives == [("waterproof", "shell"), ("rain", "jacket")]
    t = p.terms()
    assert t["shell"] == 1.0 and t["rain"] == SYNONYM_WEIGHT
    assert p.filters.active() == {}  # "waterproof" belongs to the phrase, not a filter


def test_synonyms_off_keeps_words_literal():
    p = QueryUnderstanding(synonyms=False).parse("puffy")
    assert p.concepts[0].alternatives == [("puffy",)]


def test_baseline_has_no_understanding():
    p = QueryUnderstanding.baseline("mens blue jacket under 200")
    assert [c.text for c in p.concepts] == ["men", "blue", "jacket", "under", "200"]
    assert p.filters.active() == {}
