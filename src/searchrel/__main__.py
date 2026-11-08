"""Command line.

    uv run searchrel serve                 # API and UI on http://127.0.0.1:8846
    uv run searchrel eval                  # full evaluation, writes reports/evaluation.json and prints the tables
    uv run searchrel eval --gate           # exit 1 if full pipeline NDCG@10 is not above BM25 only
    uv run searchrel debug "mens blue waterproof jacket under 200"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def tables(report: dict) -> str:
    from .evaluate import SEGMENTS, SYSTEM_LABELS

    seg = report["offline"]["segments"]
    lines = ["| System | " + " | ".join(f"{s} ({seg[s]['queries']})" for s in SEGMENTS) + " |",
             "|---|" + "---|" * len(SEGMENTS)]
    for metric in ("ndcg@10", "mrr", "recall@50"):
        for sys_key, label in SYSTEM_LABELS.items():
            vals = " | ".join(f"{seg[s][sys_key][metric]:.3f}" for s in SEGMENTS)
            lines.append(f"| {metric} {label} | {vals} |")
    z = report["zero_results"]
    lines += ["", "| Zero results (judged set) | Queries | Traffic weighted |", "|---|---|---|"]
    for k, label in (("bm25", "BM25 only"), ("bm25_qu_before_fallback", "BM25 + QU, before fallback"),
                     ("bm25_qu_after_fallback", "BM25 + QU, after fallback")):
        lines.append(f"| {label} | {z[k]['queries']:.1%} | {z[k]['traffic']:.1%} |")
    c = report["clicks"]["by_label_source"]
    lines += ["", "| LTR trained on | NDCG@10 | MRR |", "|---|---|---|"]
    for k, label in (("judgments", "Judgments"), ("clicks_raw", "Raw clicks (CTR)"),
                     ("clicks_pbm", "Clicks debiased with PBM")):
        lines.append(f"| {label} | {c[k]['ndcg@10']:.3f} | {c[k]['mrr']:.3f} |")
    lines += ["", "| A vs B | B wins | A wins | Ties | Interleaving n | A/B n (clicks) | A/B n (click rate) | Ratio |",
              "|---|---|---|---|---|---|---|---|"]
    for r in report["interleaving"]:
        lines.append(f"| {r['a']} vs {r['b']} | {r['b_wins']:.1%} | {r['a_wins']:.1%} | {r['ties']:.1%} | "
                     f"{r['sample_size_interleaving']:,} | {r['sample_size_ab_clicks']:,} | "
                     f"{r['sample_size_ab_click_rate']:,} | {r['ab_to_interleaving_ratio']}x |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="searchrel")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8846")))
    e = sub.add_parser("eval")
    e.add_argument("--gate", action="store_true", help="fail if full pipeline NDCG@10 <= BM25 only")
    e.add_argument("--out", default="reports")
    e.add_argument("--interleave-n", type=int, default=20_000)
    d = sub.add_parser("debug")
    d.add_argument("query")
    d.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("searchrel.app:create_app", factory=True, host=args.host, port=args.port, access_log=False)
        return 0

    from .app import config_from_env
    from .engine import Engine

    engine = Engine(config_from_env())
    if args.cmd == "debug":
        print(json.dumps(engine.debug(args.query, top=args.top), indent=2, default=str))
        return 0

    from .evaluate import full_report

    report = full_report(engine, interleave_n=args.interleave_n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(tables(report))
    g = report["gate"]
    print(f"\nGate: full pipeline NDCG@10 {g['ltr_ndcg@10']:.4f} vs BM25 only {g['bm25_ndcg@10']:.4f}")
    if args.gate and not g["passed"]:
        print("GATE FAILED: full pipeline NDCG@10 is not above BM25 only", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
