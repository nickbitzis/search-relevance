"""FastAPI service: search with facets, the query debugger, evaluation results, health, metrics, and the UI."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from .attributes import Filters
from .catalog import BRAND_NAMES, COLORS, GENDERS, LEAF_BY_KEY, MIDS
from .engine import Engine, EngineConfig
from .logs import get_logger, log_event
from .metrics import Metrics

STATIC = Path(__file__).parent / "static"
REPORTS = Path(os.environ.get("SEARCHREL_REPORTS", "reports"))
RANKERS = ("ltr", "bm25_qu", "bm25")


def config_from_env() -> EngineConfig:
    return EngineConfig(n_products=int(os.environ.get("SEARCHREL_PRODUCTS", "20000")),
                        n_searches=int(os.environ.get("SEARCHREL_SEARCHES", "50000")),
                        n_judged=int(os.environ.get("SEARCHREL_JUDGED", "600")),
                        seed=int(os.environ.get("SEARCHREL_SEED", "7")))


def create_app(engine: Engine | None = None) -> FastAPI:
    logger = get_logger()
    engine = engine or Engine(config_from_env())
    log_event(logger, "engine_ready", products=len(engine.products), searches=len(engine.log.searches),
              distinct_queries=len(engine.log.strings), judged=len(engine.judged),
              backend=engine.ranker.backend if engine.ranker else None, startup_seconds=engine.startup_seconds,
              timings=engine.timings)
    metrics = Metrics()
    app = FastAPI(title="search-relevance", version="1.0.0")
    app.state.engine = engine
    report_lock = threading.Lock()
    report_cache: dict = {}

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        metrics.observe(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(logger, "request", request_id=rid, method=request.method, route=path,
                      status=response.status_code, ms=round(dt * 1000, 1), query=request.query_params.get("q"))
        return response

    def evaluation() -> dict:
        """The saved report if one exists for this configuration, otherwise computed once from the engine."""
        with report_lock:
            if not report_cache:
                saved = REPORTS / "evaluation.json"
                if saved.exists():
                    data = json.loads(saved.read_text())
                    if data.get("config") == engine.cfg.__dict__:
                        report_cache.update(data)
                if not report_cache:
                    from .evaluate import full_report

                    report_cache.update(full_report(engine, interleave_n=4000))
            return report_cache

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "products": len(engine.products), "judged_queries": len(engine.judged),
                "ranker": engine.ranker.backend if engine.ranker else None, "startup_seconds": engine.startup_seconds}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        gauges = {
            "searchrel_catalog_products": ("Products in the index", [({}, len(engine.products))]),
            "searchrel_judged_queries": ("Queries with relevance judgments", [({}, len(engine.judged))]),
            "searchrel_ranker_train_rows": ("Rows the served ranker was trained on",
                                            [({"backend": engine.ranker.backend}, engine.ranker.train_rows)]
                                            if engine.ranker else []),
            "searchrel_startup_seconds": ("Time to build data, index, and models", [({}, engine.startup_seconds)]),
        }
        if report_cache:
            seg = report_cache["offline"]["segments"]["all"]
            gauges["searchrel_offline_ndcg10"] = ("Offline NDCG@10 on the judged set",
                                                  [({"system": s}, seg[s]["ndcg@10"]) for s in RANKERS])
        return metrics.render(gauges)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/search")
    def search(q: str = "", ranker: str = "ltr", category: str | None = None, brand: str | None = None,
               color: str | None = None, gender: str | None = None,
               price_min: Annotated[float | None, Query(ge=0)] = None,
               price_max: Annotated[float | None, Query(ge=0)] = None,
               limit: Annotated[int, Query(ge=1, le=100)] = 24):
        if ranker not in RANKERS:
            raise HTTPException(422, f"ranker must be one of {', '.join(RANKERS)}")
        if category and category not in LEAF_BY_KEY and category not in MIDS:
            raise HTTPException(422, f"unknown category {category}")
        if brand and brand not in BRAND_NAMES:
            raise HTTPException(422, f"unknown brand {brand}")
        if color and color not in COLORS:
            raise HTTPException(422, f"unknown color {color}")
        if gender and gender not in GENDERS:
            raise HTTPException(422, f"unknown gender {gender}")
        explicit = Filters(brand=brand, color=color, gender=gender, price_min=price_min, price_max=price_max)
        out = engine.search(q, ranker=ranker, explicit=explicit, category=category, limit=limit)
        corr = out["parse"]["corrections"]
        metrics.search(ranker, out["zero_results_before_fallback"], out["total"] == 0, out["fallback"],
                       sum(c["applied"] for c in corr), sum(not c["applied"] for c in corr))
        return out

    @app.get("/api/debug")
    def debug(q: str, top: Annotated[int, Query(ge=1, le=100)] = 20):
        return engine.debug(q, top=top)

    @app.get("/api/eval")
    def eval_results():
        return evaluation()

    @app.get("/api/examples")
    def examples():
        return engine.sample_queries()

    @app.get("/api/products/{product_id}")
    def product(product_id: str):
        i = engine.by_id.get(product_id)
        if i is None:
            raise HTTPException(404, f"no product {product_id}")
        p = engine.products[i]
        return p.public() | {"description": p.description, "category_path": p.category_path}

    return app
