import numpy as np
import pytest

from searchrel.index import InvertedIndex, bm25_idf, bm25_tf
from searchrel.retrieval import required_matches
from searchrel.text import normalize, stem, tokenize


def test_normalize_folds_apostrophes_and_punctuation():
    assert normalize("Men's Rain-Jacket, $200!") == "mens rain jacket 200"


@pytest.mark.parametrize("word,expected", [("jackets", "jacket"), ("batteries", "battery"), ("dress", "dress"),
                                           ("boots", "boot"), ("mens", "men"), ("gas", "gas"), ("200", "200"),
                                           ("cactus", "cactus")])
def test_stem(word, expected):
    assert stem(word) == expected


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("a tent for the backpacking trips") == ["tent", "backpacking", "trip"]
    assert "for" in tokenize("for men", keep_stopwords=True)


def test_idf_decreases_with_document_frequency():
    assert bm25_idf(1000, 1) > bm25_idf(1000, 10) > bm25_idf(1000, 500) > 0


def test_tf_saturates_and_length_normalizes():
    one, two, ten = (bm25_tf(t, 5, 5) for t in (1, 2, 10))
    assert one < two < ten < 2.2  # bounded by k1 + 1
    assert bm25_tf(1, 2, 5) > bm25_tf(1, 10, 5)  # same tf counts more in a short field


def test_field_scores_and_boosts():
    idx = InvertedIndex([{"title": "rain jacket", "description": "jacket"},
                         {"title": "tent", "description": "rain rain rain cover for a tent"}],
                        boosts={"title": 3.0, "description": 1.0})
    total, per = idx.score({"rain": 1.0})
    assert per["title"][0] > 0 and per["title"][1] == 0
    assert per["description"][1] > 0
    assert total[0] > total[1]  # one title hit beats three description hits with a 3x boost


def test_absent_term_scores_zero_and_weights_scale():
    idx = InvertedIndex([{"title": "rain jacket"}, {"title": "tent"}], boosts={"title": 1.0})
    assert idx.score({"kayak": 1.0})[0].sum() == 0
    a = idx.score({"rain": 1.0})[0][0]
    b = idx.score({"rain": 0.5})[0][0]
    assert b == pytest.approx(a / 2)


def test_multi_term_ranking_and_has_term():
    idx = InvertedIndex([{"title": "rain jacket"}, {"title": "rain pant"}, {"title": "down jacket"}],
                        boosts={"title": 1.0})
    total, _ = idx.score({"rain": 1.0, "jacket": 1.0})
    assert int(np.argmax(total)) == 0
    assert idx.has_term("jacket").tolist() == [True, False, True]
    assert idx.has_term("jacket", "title").tolist() == [True, False, True]


@pytest.mark.parametrize("n,need", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (6, 5)])
def test_minimum_should_match(n, need):
    assert required_matches(n) == need
