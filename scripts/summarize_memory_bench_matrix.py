#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


CSV_DIGITS = 6
PERCENT_SCALE = 100.0


@dataclass
class Row:
    metric: str = ""
    storage: str = ""
    dim: str = ""
    capacity: str = ""
    query_search: str = ""
    perf_csv: str = ""
    benchmark: str = ""
    ns_per_op: str = ""
    ticks_per_op: str = ""
    extra_name: str = ""
    extra_value: str = ""
    counters: Dict[str, float] = field(default_factory=dict)


_RE_BLOCK = re.compile(
    r"^## metric=(?P<metric>\S+)\s+storage=(?P<storage>\S+)\s+dim=(?P<dim>\S+)\s+"
    r"capacity=(?P<capacity>\S+)\s+query_search=(?P<query_search>\S*)\s*$"
)
_RE_PERF = re.compile(r"^# perf_stat:\s+(?P<path>.+\.csv)\s*$")
_RE_RESULT = re.compile(
    r"^(?P<benchmark>features\.memory\s+[^\s].*?)\s{2,}"
    r"(?P<mops>[0-9]+\.[0-9]+)\s+Mops/s\s+"
    r"(?P<ns>[0-9]+\.[0-9]+)\s+ns/op\s+"
    r"(?P<ticks>[0-9]+\.[0-9]+)\s+ticks/op\b"
    r"(?:.*?\s+(?P<extra_name>[A-Za-z0-9_]+)=\s*(?P<extra_value>[0-9]+\.[0-9]+))?"
)
_RE_LATENCY = re.compile(
    r"^(?P<benchmark>features\.memory\s+[^\s].*?)\s{2,}"
    r"p50=\s*(?P<p50>[0-9]+\.[0-9]+)\s+ns\s+"
    r"p95=\s*(?P<p95>[0-9]+\.[0-9]+)\s+ns\s+"
    r"p99=\s*(?P<p99>[0-9]+\.[0-9]+)\s+ns\s+"
    r"max=\s*(?P<max>[0-9]+\.[0-9]+)\s+ns\b"
)


def _flush(rows: List[Row], cur: Optional[Row]) -> None:
    if cur is None:
        return
    if cur.metric or cur.benchmark or cur.perf_csv:
        rows.append(cur)


def parse_log(lines: Iterable[str]) -> List[Row]:
    rows: List[Row] = []
    cur: Optional[Row] = None

    for raw in lines:
        line = raw.rstrip("\n")

        m = _RE_BLOCK.match(line)
        if m:
            _flush(rows, cur)
            cur = Row(
                metric=m.group("metric"),
                storage=m.group("storage"),
                dim=m.group("dim"),
                capacity=m.group("capacity"),
                query_search=m.group("query_search"),
            )
            continue

        if cur is None:
            continue

        m = _RE_PERF.match(line)
        if m:
            cur.perf_csv = m.group("path").strip()
            continue

        m = _RE_RESULT.match(line)
        if m:
            cur.benchmark = m.group("benchmark").strip()
            cur.ns_per_op = m.group("ns")
            cur.ticks_per_op = m.group("ticks")
            cur.extra_name = m.group("extra_name") or ""
            cur.extra_value = m.group("extra_value") or ""
            continue

        m = _RE_LATENCY.match(line)
        if m:
            cur.benchmark = m.group("benchmark").strip()
            cur.extra_name = "p50_ns"
            cur.extra_value = m.group("p50")
            cur.counters["p95_ns"] = float(m.group("p95"))
            cur.counters["p99_ns"] = float(m.group("p99"))
            cur.counters["max_ns"] = float(m.group("max"))
            continue

    _flush(rows, cur)
    return rows


def _counter_value(raw: str) -> Optional[float]:
    text = raw.strip().replace(",", "")
    if not text or text.startswith("<not"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_perf_csv(path: str) -> Dict[str, float]:
    counters: Dict[str, float] = {}
    if not path or not os.path.exists(path):
        return counters

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for fields in csv.reader(f):
            if len(fields) < 3:
                continue
            value = _counter_value(fields[0])
            if value is None:
                continue
            event = fields[2].strip()
            if event:
                counters[event] = value
    return counters


def attach_counters(rows: List[Row]) -> None:
    for row in rows:
        if row.perf_csv:
            row.counters.update(_read_perf_csv(row.perf_csv))


def _ratio(counters: Dict[str, float], left: str, right: str) -> str:
    a = counters.get(left, 0.0)
    b = counters.get(right, 0.0)
    if b <= 0.0:
        return ""
    return f"{a / b:.{CSV_DIGITS}f}"


def _percent(counters: Dict[str, float], left: str, right: str) -> str:
    a = counters.get(left, 0.0)
    b = counters.get(right, 0.0)
    if b <= 0.0:
        return ""
    return f"{(a * PERCENT_SCALE) / b:.{CSV_DIGITS}f}"


def _counter(row: Row, key: str) -> str:
    value = row.counters.get(key)
    if value is None:
        return ""
    return f"{value:.0f}"


def write_csv(rows: List[Row], out_path: str) -> None:
    fields = [
        "metric",
        "storage",
        "dim",
        "capacity",
        "query_search",
        "benchmark",
        "ns_per_op",
        "ticks_per_op",
        "extra_name",
        "extra_value",
        "ipc",
        "branch_miss_pct",
        "cache_miss_pct",
        "llc_load_miss_pct",
        "l1d_load_miss_pct",
        "dtlb_load_miss_pct",
        "cycles",
        "instructions",
        "branches",
        "branch_misses",
        "cache_references",
        "cache_misses",
        "llc_loads",
        "llc_load_misses",
        "l1d_loads",
        "l1d_load_misses",
        "dtlb_loads",
        "dtlb_load_misses",
        "p95_ns",
        "p99_ns",
        "max_ns",
        "perf_csv",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counters = row.counters
            writer.writerow(
                {
                    "metric": row.metric,
                    "storage": row.storage,
                    "dim": row.dim,
                    "capacity": row.capacity,
                    "query_search": row.query_search,
                    "benchmark": row.benchmark,
                    "ns_per_op": row.ns_per_op,
                    "ticks_per_op": row.ticks_per_op,
                    "extra_name": row.extra_name,
                    "extra_value": row.extra_value,
                    "ipc": _ratio(counters, "instructions", "cycles"),
                    "branch_miss_pct": _percent(counters, "branch-misses", "branches"),
                    "cache_miss_pct": _percent(counters, "cache-misses", "cache-references"),
                    "llc_load_miss_pct": _percent(counters, "LLC-load-misses", "LLC-loads"),
                    "l1d_load_miss_pct": _percent(counters, "L1-dcache-load-misses", "L1-dcache-loads"),
                    "dtlb_load_miss_pct": _percent(counters, "dTLB-load-misses", "dTLB-loads"),
                    "cycles": _counter(row, "cycles"),
                    "instructions": _counter(row, "instructions"),
                    "branches": _counter(row, "branches"),
                    "branch_misses": _counter(row, "branch-misses"),
                    "cache_references": _counter(row, "cache-references"),
                    "cache_misses": _counter(row, "cache-misses"),
                    "llc_loads": _counter(row, "LLC-loads"),
                    "llc_load_misses": _counter(row, "LLC-load-misses"),
                    "l1d_loads": _counter(row, "L1-dcache-loads"),
                    "l1d_load_misses": _counter(row, "L1-dcache-load-misses"),
                    "dtlb_loads": _counter(row, "dTLB-loads"),
                    "dtlb_load_misses": _counter(row, "dTLB-load-misses"),
                    "p95_ns": counters.get("p95_ns", ""),
                    "p99_ns": counters.get("p99_ns", ""),
                    "max_ns": counters.get("max_ns", ""),
                    "perf_csv": row.perf_csv,
                }
            )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize Astral memory bench-matrix output.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input log from scripts/run_memory_bench_matrix.sh")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    parser.add_argument("--require-rows", action="store_true", help="Fail if no benchmark rows are found")
    args = parser.parse_args(argv)

    with open(args.in_path, "r", encoding="utf-8", errors="replace") as f:
        rows = parse_log(f)
    attach_counters(rows)

    if args.require_rows and not rows:
        print(f"[memory-summary] no rows found in {args.in_path}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    write_csv(rows, args.out_path)
    print(f"[memory-summary] rows={len(rows)} out={args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
