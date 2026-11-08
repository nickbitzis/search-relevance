import pytest

from searchrel.spelling import SpellCorrector, edit_distance

VOCAB = {"tent": 900, "jacket": 1200, "rain": 1000, "boots": 600, "hiking": 800, "headlamp": 300, "sleeping": 500,
         "tint": 2, "quillback": 150, "bag": 700}


@pytest.mark.parametrize("a,b,d", [("tent", "tent", 0), ("tnet", "tent", 1), ("jaket", "jacket", 1),
                                   ("jackte", "jacket", 1), ("hikng", "hiking", 1), ("headlmap", "headlamp", 1),
                                   ("sleping", "sleeping", 1), ("bots", "boots", 1), ("abc", "xyz", 3),
                                   ("", "abc", 3)])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == d


@pytest.fixture()
def sc():
    return SpellCorrector(VOCAB)


@pytest.mark.parametrize("typo,fixed", [("tnet", "tent"), ("jakcet", "jacket"), ("hikign", "hiking"),
                                        ("headlam", "headlamp"), ("quilback", "quillback"), ("sleepnig", "sleeping")])
def test_unknown_tokens_are_corrected(sc, typo, fixed):
    c = sc.correct_token(typo)
    assert c is not None and c.corrected == fixed and c.confident


def test_known_digits_and_short_tokens_are_left_alone(sc):
    assert sc.correct_token("tent") is None
    assert sc.correct_token("200") is None
    assert sc.correct_token("xl") is None
    assert sc.correct_token("20k") is None


def test_rare_known_word_gets_a_suggestion_not_a_fix(sc):
    toks, found = sc.correct(["tint"])
    assert toks == ["tint"]
    assert found[0].corrected == "tent" and not found[0].confident


def test_frequency_prior_breaks_ties():
    sc = SpellCorrector({"bat": 5, "bag": 500})
    assert sc.correct_token("bax").corrected == "bag"


def test_far_words_are_not_corrected(sc):
    assert sc.correct_token("kayak") is None


def test_spelling_runs_before_price_parsing():
    from searchrel.understanding import QueryUnderstanding

    qu = QueryUnderstanding(SpellCorrector(VOCAB | {"under": 50, "shirt": 300, "black": 400}), None)
    p = qu.parse("black t shirt uder $50")
    assert p.filters.price_max == 50
    assert [(c.original, c.corrected) for c in p.corrections] == [("uder", "under")]
