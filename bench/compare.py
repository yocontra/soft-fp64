#!/usr/bin/env python3
"""Compare a soft-fp64 microbench JSON run against a committed baseline.

Exits 0 if no op regresses beyond the threshold, 1 otherwise.

Usage:
    compare.py <current.json> <baseline.json> [--threshold=0.05]

Input schema (both files):
    {
      "schema": "soft-fp64.bench.v1",
      "results": [
        {"name": str, "ns_per_op": float, "mops_per_sec": float, "iterations": int},
        ...
      ]
    }

Matching is by `name`. Regression metric is ns_per_op (lower is better):
    delta = (current_ns - baseline_ns) / baseline_ns
A positive delta greater than the threshold is a regression (failure).
A negative delta with absolute value greater than the threshold is an
improvement (informational only).

Ops with baseline ns_per_op < NOISE_FLOOR_NS are skipped — tiny timings
dominated by measurement jitter can swing ±20% run-to-run even on a
thermally stable machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_ID = "soft-fp64.bench.v1"
NOISE_FLOOR_NS = 2.0

# Cheap-op carveout threshold. Ops with a baseline below this get exempted
# from the percentage gate when `--cheap-op-absolute-budget` is set and the
# absolute ns/op delta is within that budget. Rationale: on Apple Silicon,
# `SOFT_FP64_FENV=tls` adds a structural ~5 ns/op from the `__tlv_get_addr`
# roundtrip at every raise site; on a sub-15 ns op the floor dominates the
# percentage and the percentage stops being the right signal. Ops at or
# above this baseline still gate on the percentage only.
CHEAP_OP_NS_THRESHOLD = 15.0


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load a bench JSON file and return {name: result_obj}."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except OSError as exc:
        sys.stderr.write(f"error: cannot read {path}: {exc}\n")
        sys.exit(2)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error: invalid JSON in {path}: {exc}\n")
        sys.exit(2)

    schema = doc.get("schema")
    if schema != SCHEMA_ID:
        sys.stderr.write(
            f"error: {path} has schema {schema!r}, expected {SCHEMA_ID!r}\n"
        )
        sys.exit(2)

    results = doc.get("results")
    if not isinstance(results, list):
        sys.stderr.write(f"error: {path} missing 'results' array\n")
        sys.exit(2)

    by_name: dict[str, dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        by_name[name] = entry
    return by_name


def fmt_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a soft-fp64 bench JSON against a baseline."
    )
    parser.add_argument("current", type=Path, help="Fresh bench JSON to test.")
    parser.add_argument("baseline", type=Path, help="Committed baseline JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Fractional ns/op change to flag (default 0.05 = 5%%).",
    )
    parser.add_argument(
        "--cheap-op-absolute-budget",
        type=float,
        default=0.0,
        help=(
            "Absolute ns/op budget (default 0.0 = off) that exempts ops with "
            f"a baseline < {CHEAP_OP_NS_THRESHOLD:.1f} ns from the percentage "
            "gate, provided abs(current - baseline) <= budget. Ops with a "
            f"baseline >= {CHEAP_OP_NS_THRESHOLD:.1f} ns always gate on the "
            "percentage rule. A cheap op whose absolute delta also exceeds "
            "the budget still reports as a regression."
        ),
    )
    args = parser.parse_args()

    if args.threshold <= 0:
        sys.stderr.write("error: --threshold must be > 0\n")
        return 2

    if args.cheap_op_absolute_budget < 0:
        sys.stderr.write("error: --cheap-op-absolute-budget must be >= 0\n")
        return 2

    current = load_results(args.current)
    baseline = load_results(args.baseline)

    regressions: list[tuple[str, float, float, float]] = []
    improvements: list[tuple[str, float, float, float]] = []
    within: list[tuple[str, float, float, float]] = []
    exempted_cheap: list[tuple[str, float, float, float]] = []
    skipped_noise: list[str] = []
    missing_in_current: list[str] = []
    new_ops: list[str] = []

    cheap_budget_enabled = args.cheap_op_absolute_budget > 0.0

    for name, base in baseline.items():
        base_ns = base.get("ns_per_op")
        if not isinstance(base_ns, (int, float)):
            continue
        if name not in current:
            missing_in_current.append(name)
            continue
        if base_ns < NOISE_FLOOR_NS:
            skipped_noise.append(name)
            continue
        cur_ns = current[name].get("ns_per_op")
        if not isinstance(cur_ns, (int, float)):
            missing_in_current.append(name)
            continue
        delta = (cur_ns - base_ns) / base_ns
        row = (name, base_ns, cur_ns, delta)
        if delta > args.threshold:
            # Cheap-op carveout: only applies to ops with a sub-threshold
            # baseline AND only when the absolute ns/op delta is within the
            # requested budget. Belt-and-suspenders — a cheap op that
            # breaches both the percentage and the absolute budget still
            # reports as a regression.
            if (
                cheap_budget_enabled
                and base_ns < CHEAP_OP_NS_THRESHOLD
                and abs(cur_ns - base_ns) <= args.cheap_op_absolute_budget
            ):
                exempted_cheap.append(row)
            else:
                regressions.append(row)
        elif delta < -args.threshold:
            improvements.append(row)
        else:
            within.append(row)

    for name in current:
        if name not in baseline:
            new_ops.append(name)

    all_rows = regressions + improvements + within + exempted_cheap
    all_rows.sort(key=lambda r: abs(r[3]), reverse=True)
    exempt_keys = {r[0] for r in exempted_cheap}

    out = sys.stdout.write
    out(f"# soft-fp64 bench comparison\n\n")
    out(f"- current:   `{args.current}`\n")
    out(f"- baseline:  `{args.baseline}`\n")
    out(f"- threshold: {args.threshold * 100:.2f}%\n")
    out(f"- noise floor: {NOISE_FLOOR_NS:.1f} ns/op (smaller baselines skipped)\n")
    if cheap_budget_enabled:
        out(
            f"- cheap-op carveout: baseline < {CHEAP_OP_NS_THRESHOLD:.1f} ns "
            f"exempt when |delta| <= {args.cheap_op_absolute_budget:.2f} ns\n"
        )
    out("\n")

    if all_rows:
        out("| op | baseline ns/op | current ns/op | delta % |\n")
        out("|---|---:|---:|---:|\n")
        for name, base_ns, cur_ns, delta in all_rows:
            marker = ""
            if name in exempt_keys:
                marker = " _exempt (cheap op)_"
            elif delta > args.threshold:
                marker = " **REGRESSION**"
            elif delta < -args.threshold:
                marker = " _improvement_"
            out(
                f"| {name} | {base_ns:.4f} | {cur_ns:.4f} | "
                f"{fmt_delta(delta)}{marker} |\n"
            )
        out("\n")

    if regressions:
        out(f"## Regressions ({len(regressions)})\n\n")
        for name, base_ns, cur_ns, delta in regressions:
            out(
                f"- `{name}`: {base_ns:.4f} -> {cur_ns:.4f} ns/op "
                f"({fmt_delta(delta)})\n"
            )
        out("\n")

    if improvements:
        out(f"## Improvements ({len(improvements)}) — informational\n\n")
        for name, base_ns, cur_ns, delta in improvements:
            out(
                f"- `{name}`: {base_ns:.4f} -> {cur_ns:.4f} ns/op "
                f"({fmt_delta(delta)})\n"
            )
        out("\n")

    if exempted_cheap:
        out(
            f"## Exempted cheap ops ({len(exempted_cheap)}) — "
            f"baseline < {CHEAP_OP_NS_THRESHOLD:.1f} ns, "
            f"|delta| <= {args.cheap_op_absolute_budget:.2f} ns\n\n"
        )
        for name, base_ns, cur_ns, delta in exempted_cheap:
            abs_delta = cur_ns - base_ns
            sign = "+" if abs_delta >= 0 else ""
            out(
                f"- `{name}`: {base_ns:.4f} -> {cur_ns:.4f} ns/op "
                f"({fmt_delta(delta)}, {sign}{abs_delta:.4f} ns)\n"
            )
        out("\n")

    if skipped_noise:
        out(
            f"## Skipped (baseline < {NOISE_FLOOR_NS:.1f} ns/op, noise floor)\n\n"
        )
        for name in skipped_noise:
            out(f"- `{name}`\n")
        out("\n")

    if missing_in_current:
        out("## Missing in current run (not compared)\n\n")
        for name in missing_in_current:
            out(f"- `{name}`\n")
        out("\n")

    if new_ops:
        out("## New ops not in baseline (informational)\n\n")
        for name in new_ops:
            out(f"- `{name}`\n")
        out("\n")

    if regressions:
        out(
            f"**FAIL**: {len(regressions)} op(s) regressed beyond "
            f"{args.threshold * 100:.2f}%.\n"
        )
        return 1

    out("**PASS**: no regressions beyond threshold.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
