#!/usr/bin/env bash
# Local pre-push guard — runs the same gates CI enforces on a PR so a
# contributor can catch regressions before `git push`.
#
# Three checks, all fatal on failure:
#   1. Full unfiltered ctest (CLAUDE.md §Test hygiene: no -E / -R filters).
#   2. clang-format-19 --dry-run --Werror over include/ src/ tests/.
#   3. Bench regression gate (bench/compare.py vs bench/baseline.json
#      at --threshold=0.10) iff a current.json snapshot exists.
#
# The bench step is conditional because it needs a Release build with
# SOFT_FP64_BUILD_BENCH=ON and a prior `bench_soft_fp64 --json` run. If
# current.json is missing we print instructions and keep going — bench
# regressions are still caught in CI's bench-regression job.
#
# SPDX-License-Identifier: MIT
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build"

red()  { printf '\033[0;31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[0;33m%s\033[0m\n' "$*"; }

fail=0

# ---- 1. ctest --------------------------------------------------------------
if [[ ! -d "$build" ]]; then
    red "No build directory at $build. Configure + build first:"
    red "    cmake -B build && cmake --build build --parallel"
    exit 1
fi

echo "==> ctest (unfiltered)"
if ctest --test-dir "$build" --output-on-failure; then
    grn "ctest: PASS"
else
    red "ctest: FAIL"
    fail=1
fi

# ---- 2. clang-format-19 ----------------------------------------------------
echo
echo "==> clang-format-19 --dry-run --Werror"

cf=""
for cand in clang-format-19 clang-format; do
    if command -v "$cand" >/dev/null 2>&1; then
        # Check version — lint CI is pinned to 19, so local must match.
        ver=$("$cand" --version | grep -Eo '[0-9]+' | head -1 || true)
        if [[ "$ver" == "19" ]]; then
            cf="$cand"
            break
        fi
    fi
done

if [[ -z "$cf" ]]; then
    ylw "clang-format-19 not found on PATH; skipping format check."
    ylw "Install: brew install clang-format@19 (macOS) or from apt.llvm.org (Linux)."
else
    # Mirror CI's file discovery: include/, src/, tests/, excluding vendored
    # trees (tests/testfloat/vendor, tests/external). No find/xargs here —
    # globs through bash's find for portability with BSD find on macOS.
    mapfile -d '' files < <(
        find "$root/include" "$root/src" "$root/tests" \
             \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) \
             -not -path '*/testfloat/vendor/*' \
             -not -path '*/external/*' \
             -print0
    )
    if "$cf" --dry-run --Werror --style=file "${files[@]}"; then
        grn "clang-format-19: PASS"
    else
        red "clang-format-19: FAIL"
        fail=1
    fi
fi

# ---- 3. Bench regression gate ---------------------------------------------
echo
echo "==> bench/compare.py (threshold 0.10)"

current="$root/current.json"
baseline="$root/bench/baseline.json"

if [[ ! -f "$current" ]]; then
    ylw "No current.json at $current — skipping bench gate."
    ylw "To run bench locally:"
    ylw "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOFT_FP64_BUILD_BENCH=ON"
    ylw "    cmake --build build --target bench_soft_fp64"
    ylw "    ./build/bench/bench_soft_fp64 --json --min-time-ms=500 > current.json"
elif [[ ! -f "$baseline" ]]; then
    red "No baseline.json at $baseline — repo is malformed."
    fail=1
else
    if python3 "$root/bench/compare.py" "$current" "$baseline" --threshold=0.10; then
        grn "bench/compare.py: PASS"
    else
        red "bench/compare.py: FAIL"
        fail=1
    fi
fi

# ---- Summary ---------------------------------------------------------------
echo
if (( fail )); then
    red "pre_push: FAILURES — do not push."
    exit 1
fi
grn "pre_push: all checks PASS."
