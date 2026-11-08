"""Training and evaluation: cross-validated learning to rank on judgments and on clicks, offline metrics by
segment, query understanding accuracy, zero-result rates, the margin trade-off, and simulated interleaving.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .catalog import LEAF_BY_KEY
from .clickmodel import ATTRACT_BY_GRADE, ClickLog, PBMFit, ctr_labels, examination, fit_pbm, simulate_clicks
from .interleave import ab_sample_size, credit, interleaving_sample_size, sign_test_p, team_draft
from .ir import ndcg, recall_at, reciprocal_rank
from .ltr import Ranker, rank_order
from .text import tokenize
from .understanding import QueryUnderstanding

SYSTEMS = ("bm25", "bm25_qu", "ltr")
SYSTEM_LABELS = {"bm25": "BM25 only", "bm25_qu": "BM25 + query understanding", "ltr": "+ learning to rank"}
SEGMENTS = ("all", "head", "torso", "tail", "typo", "attribute")
LABEL_SOURCES = ("judgments", "clicks_raw", "clicks_pbm")
RANDOMIZED_SHARE = 0.1  # share of logged sessions with a shuffled top 10
GRADE_CUTS = (ATTRACT_BY_GRADE[:-1] + ATTRACT_BY_GRADE[1:]) / 2  # attractiveness -> grade 0..3


@dataclass
class ClickData:
    log: ClickLog
    pbm: PBMFit
    ctr: dict[tuple[int, int], float]
    theta_true: np.ndarray
    sessions: int


def _production_sessions(engine, rng: np.random.Generator) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """One session per logged search of a judged query, shown in BM25 + QU order. 10% of sessions shuffle the top
    10 and another 40% swap adjacent pairs: the position variation a click model needs to separate position from
    relevance. Production search teams run a small randomization bucket for exactly this reason."""
    by_qi = {jq.qi: i for i, jq in enumerate(engine.judged)}
    sessions = []
    for qi in engine.log.searches:
        i = by_qi.get(int(qi))
        if i is None:
            continue
        jq = engine.judged[i]
        docs = jq.full.ids[:10].copy()
        r = rng.random()
        if r < RANDOMIZED_SHARE:
            rng.shuffle(docs)  # a small randomization bucket: the top 10 in random order
        elif r < 0.5:
            # Overlapping pairings (0-1, 2-3, ... or 1-2, 3-4, ...) chain every rank to its neighbors.
            for k in range(int(rng.integers(2)), len(docs) - 1, 2):
                if rng.random() < 0.5:
                    docs[k], docs[k + 1] = docs[k + 1], docs[k]
        sessions.append((i, docs, jq.grades[docs].astype(np.int64)))
    return sessions


def _labels(engine, source: str, clicks: ClickData | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """(row mask, label) per judged query for one label source."""
    out = []
    for i, jq in enumerate(engine.judged):
        ids = jq.full.ids
        if source == "judgments":
            out.append((np.ones(len(ids), dtype=bool), jq.grades[ids].astype(np.float64)))
            continue
        table = clicks.ctr if source == "clicks_raw" else clicks.pbm.alpha
        vals = np.array([table.get((i, int(d)), np.nan) for d in ids])
        mask = ~np.isnan(vals)
        out.append((mask, np.digitize(np.nan_to_num(vals), GRADE_CUTS).astype(np.float64)))
    return out


def _fit(engine, labels, members: list[int], seed: int) -> Ranker | None:
    Xs, ys, groups = [], [], []
    for i in members:
        mask, y = labels[i]
        if mask.sum() < 2:
            continue
        Xs.append(engine.judged[i].X[mask])
        ys.append(y[mask])
        groups.append(int(mask.sum()))
    if not Xs:
        return None
    return Ranker(seed=seed).fit(np.vstack(Xs), np.concatenate(ys), groups)


def train_rankers(engine) -> None:
    cfg = engine.cfg
    rng = np.random.default_rng(cfg.seed + 202)
    n = len(engine.judged)
    fold = rng.permutation(np.arange(n) % cfg.folds)
    theta = examination(10)
    sessions = _production_sessions(engine, rng)
    log = simulate_clicks(rng, sessions, theta)
    pbm = fit_pbm(log)
    engine.clicks = ClickData(log=log, pbm=pbm, ctr=ctr_labels(log), theta_true=theta, sessions=len(sessions))
    engine.fold_of = fold
    for source in LABEL_SOURCES:
        labels = _labels(engine, source, engine.clicks)
        oof = [np.zeros(len(jq.full.ids)) for jq in engine.judged]
        for f in range(cfg.folds):
            model = _fit(engine, labels, [i for i in range(n) if fold[i] != f], seed=cfg.seed + f)
            for i in np.flatnonzero(fold == f):
                if model is not None and len(engine.judged[i].X):
                    oof[i] = model.predict(engine.judged[i].X)
        engine.oof[source] = oof
    engine.ranker = _fit(engine, _labels(engine, "judgments", None), list(range(n)), seed=cfg.seed)


def ranking(engine, i: int, system: str) -> np.ndarray:
    jq = engine.judged[i]
    if system == "bm25":
        return jq.base.ids
    if system == "bm25_qu":
        return jq.full.ids
    source = {"ltr": "judgments"}.get(system, system.replace("ltr_", ""))
    return jq.full.ids[rank_order(engine.oof[source][i])]


def _metrics(jq, ids: np.ndarray) -> dict:
    g = jq.grades[ids]
    return {"ndcg@10": ndcg(g, jq.grades[jq.grades > 0], 10), "mrr": reciprocal_rank(g),
            "recall@50": recall_at(g, jq.n_relevant, 50)}


def query_metrics(engine, jq) -> dict:
    i = engine.judged.index(jq)
    return {s: {k: round(v, 4) for k, v in _metrics(jq, ranking(engine, i, s)).items()} for s in SYSTEMS}


def _in_segment(jq, seg: str) -> bool:
    if seg == "all":
        return True
    if seg == "typo":
        return jq.typo
    if seg == "attribute":
        return jq.attribute and not jq.typo
    return not jq.typo and jq.segment == seg


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def offline_report(engine, systems=SYSTEMS) -> dict:
    evaluable = [i for i, jq in enumerate(engine.judged) if jq.n_relevant > 0]
    per = {s: {i: _metrics(engine.judged[i], ranking(engine, i, s)) for i in evaluable} for s in systems}
    table: dict = {}
    for seg in SEGMENTS:
        members = [i for i in evaluable if _in_segment(engine.judged[i], seg)]
        table[seg] = {"queries": len(members)}
        for s in systems:
            table[seg][s] = {m: round(float(np.mean([per[s][i][m] for i in members])), 4) if members else None
                             for m in ("ndcg@10", "mrr", "recall@50")}
    w = np.array([engine.judged[i].weight for i in evaluable], dtype=np.float64)
    weighted = {s: round(float(np.average([per[s][i]["ndcg@10"] for i in evaluable], weights=w)), 4)
                for s in systems}
    rng = np.random.default_rng(engine.cfg.seed + 303)
    nd = {s: np.array([per[s][i]["ndcg@10"] for i in evaluable]) for s in systems}
    out = {"segments": table, "traffic_weighted_ndcg@10": weighted, "evaluable_queries": len(evaluable),
           "judged_queries": len(engine.judged), "no_relevant_products": len(engine.judged) - len(evaluable),
           "judgments": sum(len(jq.judgments) for jq in engine.judged)}
    if "ltr" in systems:
        out["ci95_ndcg@10"] = {"bm25_qu_minus_bm25": _bootstrap_diff(nd["bm25"], nd["bm25_qu"], rng),
                               "ltr_minus_bm25_qu": _bootstrap_diff(nd["bm25_qu"], nd["ltr"], rng),
                               "ltr_minus_bm25": _bootstrap_diff(nd["bm25"], nd["ltr"], rng)}
    return out


def click_report(engine) -> dict:
    evaluable = [i for i, jq in enumerate(engine.judged) if jq.n_relevant > 0]
    res = {}
    for source in LABEL_SOURCES:
        name = "ltr" if source == "judgments" else f"ltr_{source}"
        vals = [_metrics(engine.judged[i], ranking(engine, i, name)) for i in evaluable]
        res[source] = {m: round(float(np.mean([v[m] for v in vals])), 4) for m in ("ndcg@10", "mrr")}
    c = engine.clicks
    labeled = sum(1 for i, jq in enumerate(engine.judged) for d in jq.full.ids if (i, int(d)) in c.pbm.alpha)
    return {"by_label_source": res, "sessions": c.sessions, "impressions": int(len(c.log.click)),
            "clicks": int(c.log.click.sum()), "labeled_pairs": labeled,
            "examination_true": [round(float(x), 3) for x in c.theta_true],
            "examination_estimated": [round(float(x), 3) for x in c.pbm.theta]}


def zero_results_report(engine) -> dict:
    J = engine.judged
    w = np.array([jq.weight for jq in J], dtype=np.float64)

    def rate(flags):
        f = np.array(flags, dtype=np.float64)
        return {"queries": round(float(f.mean()), 4), "traffic": round(float(np.average(f, weights=w)), 4)}

    steps = defaultdict(int)
    for jq in J:
        for s in jq.full.steps:
            steps[s.split(" ")[0] + " " + s.split(" ")[1] if s.startswith("relaxed") else s.split(" ->")[0]] += 1
    return {"bm25": rate([jq.base.n_hits == 0 for jq in J]),
            "bm25_qu_before_fallback": rate([jq.full.zero_initial for jq in J]),
            "bm25_qu_after_fallback": rate([jq.full.n_hits == 0 for jq in J]),
            "fallback_steps": dict(sorted(steps.items(), key=lambda kv: -kv[1])[:10])}


def margin_report(engine, lambdas=(0.0, 0.1, 0.25, 0.5, 1.0)) -> list[dict]:
    """Blend a z-scored margin into the LTR score and see what it costs in NDCG."""
    margins = np.array([p.margin for p in engine.products])
    mu, sd = margins.mean(), margins.std()
    evaluable = [i for i, jq in enumerate(engine.judged) if jq.n_relevant > 0]
    out = []
    for lam in lambdas:
        nds, mg = [], []
        for i in evaluable:
            jq = engine.judged[i]
            s = engine.oof["judgments"][i]
            if len(s) == 0:
                nds.append(0.0)
                continue
            sd_s = s.std() or 1.0
            blended = s / sd_s + lam * (margins[jq.full.ids] - mu) / sd
            ids = jq.full.ids[rank_order(blended)]
            nds.append(ndcg(jq.grades[ids], jq.grades[jq.grades > 0], 10))
            mg.append(margins[ids[:10]].mean())
        out.append({"lambda": lam, "ndcg@10": round(float(np.mean(nds)), 4),
                    "top10_margin": round(float(np.mean(mg)), 4)})
    return out


def understanding_report(engine) -> dict:
    log = engine.log
    hold = sorted(engine.classifier_holdout)
    qu = engine.qu
    # Spelling: typo variants should parse to the same tokens as their clean string.
    typo = [qi for qi in hold if log.typo_of[qi]]
    fixed = sum(qu.parse(log.strings[qi]).tokens == qu.parse(log.typo_of[qi]).tokens for qi in typo)
    clean = [qi for qi in hold if not log.typo_of[qi]]
    false_fix = sum(any(c.confident for c in qu.parse(log.strings[qi]).corrections) for qi in clean)
    # Attributes: per slot precision and recall against the hidden intent.
    slots = ("gender", "color", "size", "material", "waterproof_min", "price_min", "price_max", "brand")
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    exact = 0
    syn_on = syn_off = n_leaf = 0
    no_syn = QueryUnderstanding(engine.speller, None, synonyms=False, category=False)
    cat_ok = cat_n = filt_n = filt_ok = 0
    for qi in clean:
        it = log.intents[qi]
        p = qu.parse(log.strings[qi])
        got = p.filters.active()
        want = {s: getattr(it, s) for s in slots if getattr(it, s) is not None}
        for s in slots:
            g, t = got.get(s), want.get(s)
            if g is not None and t is not None and g == t:
                tp[s] += 1
            else:
                fp[s] += g is not None
                fn[s] += t is not None
        exact += got == want
        if it.leaf:
            canon = set(tokenize(LEAF_BY_KEY[it.leaf].nouns[0]))
            n_leaf += 1
            syn_on += canon <= set(p.terms())
            syn_off += canon <= set(no_syn.parse(log.strings[qi]).terms())
            if p.category.top:
                cat_n += 1
                cat_ok += p.category.top[0][0] == it.leaf
        if p.category.level and (it.leaf or it.mid):
            filt_n += 1
            mid = it.category_mid
            filt_ok += (p.category.value == it.leaf) if p.category.level == "leaf" else (p.category.value == mid)
    attr = {s: {"precision": round(tp[s] / (tp[s] + fp[s]), 3) if tp[s] + fp[s] else None,
                "recall": round(tp[s] / (tp[s] + fn[s]), 3) if tp[s] + fn[s] else None,
                "support": tp[s] + fn[s]} for s in slots}
    return {
        "sample_queries": len(hold),
        "spelling": {"typo_queries": len(typo), "corrected": round(fixed / max(len(typo), 1), 3),
                     "clean_queries": len(clean), "false_corrections": round(false_fix / max(len(clean), 1), 4)},
        "attributes": attr, "attributes_exact_match": round(exact / max(len(clean), 1), 3),
        "synonyms": {"leaf_queries": n_leaf, "catalog_noun_covered_with": round(syn_on / max(n_leaf, 1), 3),
                     "catalog_noun_covered_without": round(syn_off / max(n_leaf, 1), 3)},
        "category": {"top1_accuracy": round(cat_ok / max(cat_n, 1), 3), "queries": cat_n,
                     "filter_rate": round(filt_n / max(len(clean), 1), 3),
                     "filter_precision": round(filt_ok / max(filt_n, 1), 3)},
    }


def _user_clicks(rng: np.random.Generator, grades: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """One shopper on one result page. Shoppers differ: click propensity is Gamma distributed (mean 1, heavy
    spread) and patience varies, and a shopper who clicks a good result usually stops. This between-user variance
    is what makes A/B tests on clicks slow, and what interleaving cancels by showing one user both rankers."""
    k = len(grades)
    propensity = rng.gamma(0.6, 1 / 0.6)
    patience = rng.uniform(0.6, 1.6)
    p = np.clip(theta[:k] ** patience * ATTRACT_BY_GRADE[grades] * propensity, 0, 1)
    clicks = np.zeros(k, dtype=bool)
    for r in range(k):
        if rng.random() < p[r]:
            clicks[r] = True
            if grades[r] >= 2 and rng.random() < 0.6:
                break
    return clicks


def _simulate(engine, a: str, b: str, n: int, rng: np.random.Generator) -> dict:
    evaluable = [i for i, jq in enumerate(engine.judged) if len(jq.full.ids)]
    w = np.array([engine.judged[i].weight for i in evaluable], dtype=np.float64)
    picks = rng.choice(evaluable, size=(2, n), p=w / w.sum())  # interleaving traffic, A/B traffic
    theta = engine.clicks.theta_true
    il, ma, mb, sa, sb = (np.zeros(n) for _ in range(5))
    for t in range(n):
        i = picks[0, t]
        jq = engine.judged[i]
        ra, rb = ranking(engine, i, a)[:10].tolist(), ranking(engine, i, b)[:10].tolist()
        shown, team = team_draft(ra, rb, rng)
        il[t] = credit(team, _user_clicks(rng, jq.grades[shown].astype(np.int64), theta)) if shown else 0
        # A/B: each arm is a separate search by a separate shopper. Arm B replays arm A's search and shopper
        # (common random numbers), so the mean difference is measured precisely while each arm's spread is the
        # spread of independent sessions.
        j = picks[1, t]
        seed = int(rng.integers(2**32))
        for system, clicks_out, any_out in ((a, ma, sa), (b, mb, sb)):
            docs = ranking(engine, j, system)[:10]
            g = engine.judged[j].grades[docs].astype(np.int64)
            c = _user_clicks(np.random.default_rng(seed), g, theta) if len(docs) else np.zeros(0, dtype=bool)
            clicks_out[t] = c.sum()
            any_out[t] = float(c.any())
    return {"interleave": il, "clicks_a": ma, "clicks_b": mb, "any_a": sa, "any_b": sb}


def interleaving_report(engine, pairs=(("bm25_qu", "ltr"),), n: int = 20_000,
                        power_runs: int = 300) -> list[dict]:
    rng = np.random.default_rng(engine.cfg.seed + 404)
    out = []
    for a, b in pairs:
        sim = _simulate(engine, a, b, n, rng)
        il = sim["interleave"]
        n_il = interleaving_sample_size(il)
        n_ab_clicks = ab_sample_size(sim["clicks_a"], sim["clicks_b"])
        n_ab_rate = ab_sample_size(sim["any_a"], sim["any_b"])
        n_ab = min(n_ab_clicks, n_ab_rate)
        # Empirical power at the computed sizes: resample impressions and count how often the test detects B > A.
        power = None
        if np.isfinite(n_il):
            m = int(np.ceil(n_il))
            samples = (rng.choice(il, size=m) for _ in range(power_runs))
            hits = sum(sign_test_p(x) < 0.05 and x.mean() > 0 for x in samples)
            power = round(hits / power_runs, 3)
        out.append({
            "a": a, "b": b, "impressions_simulated": n,
            "b_wins": round(float(np.mean(il > 0)), 4), "a_wins": round(float(np.mean(il < 0)), 4),
            "ties": round(float(np.mean(il == 0)), 4), "preference": round(float(il.mean()), 4),
            "p_value": sign_test_p(il),
            "ab_clicks_per_search": [round(float(sim["clicks_a"].mean()), 4), round(float(sim["clicks_b"].mean()), 4)],
            "ab_click_rate": [round(float(sim["any_a"].mean()), 4), round(float(sim["any_b"].mean()), 4)],
            "sample_size_interleaving": int(np.ceil(n_il)) if np.isfinite(n_il) else None,
            "sample_size_ab_clicks": int(np.ceil(n_ab_clicks)) if np.isfinite(n_ab_clicks) else None,
            "sample_size_ab_click_rate": int(np.ceil(n_ab_rate)) if np.isfinite(n_ab_rate) else None,
            "ab_to_interleaving_ratio": (round(float(n_ab / n_il), 1)
                                         if np.isfinite(n_il) and np.isfinite(n_ab) else None),
            "empirical_power_at_interleaving_size": power,
        })
    return out


def full_report(engine, interleave_n: int = 20_000) -> dict:
    off = offline_report(engine)
    return {
        "config": engine.cfg.__dict__, "ranker_backend": engine.ranker.backend if engine.ranker else None,
        "startup_seconds": engine.startup_seconds, "offline": off, "zero_results": zero_results_report(engine),
        "clicks": click_report(engine), "margin_tradeoff": margin_report(engine),
        "understanding": understanding_report(engine),
        "interleaving": interleaving_report(engine, n=interleave_n),
        "gate": {"ltr_ndcg@10": off["segments"]["all"]["ltr"]["ndcg@10"],
                 "bm25_ndcg@10": off["segments"]["all"]["bm25"]["ndcg@10"],
                 "passed": off["segments"]["all"]["ltr"]["ndcg@10"] > off["segments"]["all"]["bm25"]["ndcg@10"]},
    }
