import pytest
from fastapi.testclient import TestClient

from searchrel.app import create_app
from searchrel.evaluate import offline_report, understanding_report, zero_results_report


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_search_returns_results_and_facets(client):
    r = client.get("/api/search", params={"q": "mens rain jacket under 200"}).json()
    assert r["total"] > 0 and r["results"]
    assert r["filters"]["gender"] == "men" and r["filters"]["price_max"] == 200
    assert all(x["price"] <= 200 for x in r["results"])
    assert {"category", "brand", "color", "gender", "price"} <= set(r["facets"])


def test_search_spelling(client):
    r = client.get("/api/search", params={"q": "sleepnig bag"}).json()
    assert r["spelling"]["showing_results_for"] == "sleeping bag"


def test_search_facet_filters_and_rankers(client):
    for ranker in ("ltr", "bm25_qu", "bm25"):
        r = client.get("/api/search", params={"q": "tent", "ranker": ranker, "color": "green"}).json()
        assert all(x["color"] == "green" for x in r["results"])
    r = client.get("/api/search", params={"q": "jacket", "category": "rain_jacket"}).json()
    assert all(x["leaf"] == "rain_jacket" for x in r["results"])


def test_search_validation(client):
    assert client.get("/api/search", params={"q": "x", "ranker": "nope"}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "brand": "Nope"}).status_code == 422
    assert client.get("/api/products/p99999999").status_code == 404


def test_debug_endpoint(client, engine):
    q = engine.judged[0].text
    r = client.get("/api/debug", params={"q": q, "top": 5}).json()
    assert r["parse"]["raw"] == q
    assert len(r["results"]) <= 5
    if r["results"]:
        row = r["results"][0]
        assert set(row["features"]) and row["why"] and "grade" in row
    assert "judged" in r


def test_eval_and_metrics(client):
    r = client.get("/api/eval").json()
    assert r["offline"]["segments"]["all"]["queries"] > 0
    assert r["interleaving"][0]["sample_size_interleaving"]
    m = client.get("/metrics").text
    assert "searchrel_requests_total" in m and "searchrel_searches_total" in m
    assert "searchrel_offline_ndcg10" in m


def test_full_pipeline_beats_bm25_on_the_judged_set(engine):
    seg = offline_report(engine)["segments"]
    assert seg["all"]["ltr"]["ndcg@10"] > seg["all"]["bm25"]["ndcg@10"]
    assert seg["all"]["bm25_qu"]["ndcg@10"] > seg["all"]["bm25"]["ndcg@10"]


def test_fallback_lowers_zero_results(engine):
    z = zero_results_report(engine)
    assert z["bm25_qu_after_fallback"]["queries"] <= z["bm25_qu_before_fallback"]["queries"]
    assert z["bm25_qu_after_fallback"]["queries"] < z["bm25"]["queries"]


def test_understanding_report_is_sane(engine):
    u = understanding_report(engine)
    assert u["spelling"]["corrected"] > 0.5
    assert u["spelling"]["false_corrections"] < 0.05
    assert u["attributes"]["gender"]["precision"] > 0.9
    assert u["synonyms"]["catalog_noun_covered_with"] > u["synonyms"]["catalog_noun_covered_without"]
    assert u["category"]["top1_accuracy"] > 0.6
