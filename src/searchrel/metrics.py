"""Prometheus text exposition without a client library."""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(d.items())) + "}" if d else ""


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = defaultdict(int)  # (route, status)
        self.latency = defaultdict(lambda: [0] * len(BUCKETS))
        self.latency_sum = defaultdict(float)
        self.latency_count = defaultdict(int)
        self.searches = defaultdict(int)  # ranker
        self.zero_results = defaultdict(int)  # stage: before_fallback, after_fallback
        self.fallback_steps = defaultdict(int)  # step kind
        self.spelling = defaultdict(int)  # applied, suggested

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[(route, str(status))] += 1
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.latency[route][i] += 1
            self.latency_sum[route] += seconds
            self.latency_count[route] += 1

    def search(self, ranker: str, zero_before: bool, zero_after: bool, steps: list[str], applied: int,
               suggested: int) -> None:
        with self.lock:
            self.searches[ranker] += 1
            if zero_before:
                self.zero_results["before_fallback"] += 1
            if zero_after:
                self.zero_results["after_fallback"] += 1
            for s in steps:
                self.fallback_steps[" ".join(s.split(" ")[:2]).replace(" spelling", "")] += 1
            self.spelling["applied"] += applied
            self.spelling["suggested"] += suggested

    def render(self, gauges: dict[str, tuple[str, list[tuple[dict, float]]]]) -> str:
        lines = ["# HELP searchrel_requests_total HTTP requests by route and status",
                 "# TYPE searchrel_requests_total counter"]
        with self.lock:
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f"searchrel_requests_total{_labels({'route': route, 'status': status})} {n}")
            lines += ["# HELP searchrel_request_seconds Request latency", "# TYPE searchrel_request_seconds histogram"]
            for route in sorted(self.latency_count):
                for b, n in zip(BUCKETS, self.latency[route], strict=True):
                    lines.append(f"searchrel_request_seconds_bucket{_labels({'route': route, 'le': b})} {n}")
                lines.append(f"searchrel_request_seconds_bucket{_labels({'route': route, 'le': '+Inf'})} "
                             f"{self.latency_count[route]}")
                lines.append(f"searchrel_request_seconds_sum{_labels({'route': route})} {self.latency_sum[route]:.6f}")
                lines.append(f"searchrel_request_seconds_count{_labels({'route': route})} {self.latency_count[route]}")
            for name, help_text, label, table in (
                ("searchrel_searches_total", "Searches served by ranker", "ranker", self.searches),
                ("searchrel_zero_results_total", "Searches with zero results", "stage", self.zero_results),
                ("searchrel_fallback_steps_total", "Null-result fallback steps taken", "step", self.fallback_steps),
                ("searchrel_spelling_corrections_total", "Spelling corrections", "kind", self.spelling),
            ):
                lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
                for k, n in sorted(table.items()):
                    lines.append(f"{name}{_labels({label: k})} {n}")
        for name, (help_text, samples) in gauges.items():
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
            for labels, value in samples:
                lines.append(f"{name}{_labels(labels)} {value}")
        return "\n".join(lines) + "\n"
