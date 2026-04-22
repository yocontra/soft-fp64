#!/usr/bin/env python3
"""README sweep-claim parity check.

Every numeric coverage window claimed in README.md (e.g. ``x ∈ [1e-100,
1e100]``, ``y ∈ [-5, 5]``) must be traceable to a sweep call site in
the test source. The reviewer that prompted this guard found that the
"x-wide: x ∈ [1e-100, 1e100]" claim was paired with a linear sweep that
sampled almost exclusively at ~1e100 — the lower 99 decades were
unexercised, so the README claim was not actually tested.

This script does not check sampling strategy (the trivial-match gate in
``record()`` does that). It checks that the (lo, hi) numbers quoted in
the README appear, as a numeric pair in that order and in proximity, in
the test source. Numbers are normalised before matching so that ``-50``
in the README matches ``-50.0`` in the source.

Failures are printed with ``file:line`` for both the README claim and
the proximity-search target, suitable for prek hook output. Exit
non-zero on any unmatched claim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
TEST_SOURCES = [
    REPO / "tests" / "mpfr" / "test_mpfr_diff.cpp",
    REPO / "tests" / "test_coverage_mpfr.cpp",
    REPO / "tests" / "test_transcendental_1ulp.cpp",
]

# Capture `[lo, hi]` after an `x ∈ ` / `y ∈ ` / `x in ` / `y in ` lead.
# Numbers may be in scientific notation (1e-100, -5e+5), decimal
# (-50.0), integer (-50), or include sign.
NUM = r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
CLAIM = re.compile(
    r"[`*]?[xy]\b\s*(?:∈|in)\s*\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]",
)


def normalise(token: str) -> float:
    return float(token)


def number_appears_in(source: str, value: float) -> list[int]:
    """Return line numbers in `source` containing `value` (as a number)."""
    hits: list[int] = []
    pattern = re.compile(r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for lineno, line in enumerate(source.splitlines(), start=1):
        for token in pattern.findall(line):
            try:
                if normalise(token) == value:
                    hits.append(lineno)
                    break
            except ValueError:
                pass
    return hits


def find_pair_in_proximity(
    source_path: Path, source_text: str, lo: float, hi: float, window: int = 4
) -> tuple[int, str] | None:
    """True if both `lo` and `hi` appear within `window` lines of each
    other in `source_text`, with `lo` preceding `hi` (matches a typical
    `sweep2(..., lo, hi, …)` argument list)."""
    lo_lines = number_appears_in(source_text, lo)
    hi_lines = number_appears_in(source_text, hi)
    for la in lo_lines:
        for lb in hi_lines:
            if 0 <= lb - la <= window:
                # Show both lines for context.
                lines = source_text.splitlines()
                snippet = "\n".join(
                    f"      {source_path.name}:{n}: {lines[n - 1].rstrip()}"
                    for n in range(la, lb + 1)
                    if 1 <= n <= len(lines)
                )
                return (la, snippet)
    return None


def main() -> int:
    if not README.exists():
        print(f"README.md not found at {README}", file=sys.stderr)
        return 2

    readme_text = README.read_text(encoding="utf-8")
    sources = {p: p.read_text(encoding="utf-8") for p in TEST_SOURCES if p.exists()}
    if not sources:
        print("No test sources to check parity against.", file=sys.stderr)
        return 2

    failures: list[str] = []
    checked = 0

    for lineno, line in enumerate(readme_text.splitlines(), start=1):
        for match in CLAIM.finditer(line):
            checked += 1
            lo_token, hi_token = match.group(1), match.group(2)
            try:
                lo = normalise(lo_token)
                hi = normalise(hi_token)
            except ValueError:
                continue
            found_in: list[str] = []
            for path, text in sources.items():
                hit = find_pair_in_proximity(path, text, lo, hi)
                if hit is not None:
                    found_in.append(f"{path.relative_to(REPO)}:{hit[0]}")
            if not found_in:
                failures.append(
                    f"README.md:{lineno}: [{lo_token}, {hi_token}] — no test "
                    f"source contains this pair as adjacent sweep arguments."
                    f"\n      claim: {line.strip()}"
                )

    if failures:
        print(
            f"check_readme_claim_parity: {len(failures)} unmatched claim(s) "
            f"out of {checked}.",
            file=sys.stderr,
        )
        for f in failures:
            print("  " + f, file=sys.stderr)
        print(
            "\n  Each README sweep window must appear as adjacent (lo, hi) "
            "arguments in a test sweep call. Either fix the README to match "
            "the test code or fix the test code to match the README — do "
            "not let the two diverge.",
            file=sys.stderr,
        )
        return 1

    print(
        f"check_readme_claim_parity: clean ({checked} README claims, all "
        f"matched in test source)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
