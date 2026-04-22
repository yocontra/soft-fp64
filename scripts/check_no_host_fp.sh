#!/usr/bin/env bash
# check_no_host_fp.sh — mechanical guard for the "no hidden host-FPU
# dependency" rule in CLAUDE.md.
#
# Scans every .cpp / .h under src/ (no carve-out — the SLEEF port was
# completed). Flags:
#   - #include <cmath> / <math.h> / <fenv.h> / <cfenv> / <cfloat> / <float.h>
#   - libm / std:: math names called as functions
#   - __builtin_ FP intrinsics (sqrt, fma, sin, cos, ...)
#   - static_cast<double>(integer…) and C-style (double)/(float) casts
#     from an integer-looking token
#   - fp-contract / fast-math escape pragmas (CMakeLists.txt pins
#     -ffp-contract=off globally; source overrides silently undo it)
#   - FP-literal comparisons — raw `x OP <double-literal>` in src/.
#     Must use the sleef::lt_ / gt_ / eq_ / etc helpers (or, in the core,
#     integer bit comparison on bits_of(x)). Integer literal comparisons
#     (e.g. `(q & 2) != 0`) are unaffected because they don't match the
#     FP-literal pattern (must have a decimal point or exponent marker).
#
# Exits non-zero on first finding with file:line:match so prek surfaces
# the exact offender. Comments (// and /* */, including multi-line) are
# stripped before scanning so docstrings and explanatory prose do not
# false-fire on math terminology.
#
# Definition of "FPU dependency" here: the library's ABI is `double`, and
# passing/returning a `double` between `sf64_*` calls is NOT an FPU use —
# the bits are manipulated only via __builtin_bit_cast. What IS an FPU
# use is any of the patterns above: they lower to hardware fp64 ops on
# the host and break bit-exactness on targets without native fp64.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Files to scan: every .cpp / .h under src/, including src/sleef/. The
# sleef port completed the `+ - * / < > == != (double)int` conversion to
# `sf64_*` ABI calls; there is no carve-out.
files=()
while IFS= read -r f; do
    files+=("$f")
done < <(find src -type f \( -name '*.cpp' -o -name '*.h' \) | sort)

if [ "${#files[@]}" -eq 0 ]; then
    echo "check_no_host_fp: no source files found under src/ (top level)" >&2
    exit 2
fi

# Strip C/C++ comments while preserving line numbers by replacing each
# comment character with a space (newlines are kept). This means grep -n
# on the preprocessed stream reports the same line numbers as the
# original file.
preprocess() {
    perl -0777 -pe '
        s{/\*.*?\*/}{ my $m=$&; $m =~ s/[^\n]/ /gs; $m }gse;
        s{//[^\n]*}{ my $m=$&; $m =~ s/./ /gs; $m }ge;
    ' "$1"
}

findings=0
report() {
    findings=$((findings + 1))
    printf '%s\n' "$1"
}

scan() {
    local label="$1" pattern="$2"
    local f pp hit
    for f in "${files[@]}"; do
        pp=$(preprocess "$f")
        hit=$(printf '%s' "$pp" | grep -nE "$pattern" 2>/dev/null || true)
        if [ -n "$hit" ]; then
            while IFS= read -r line; do
                report "[$label] $f:$line"
            done <<< "$hit"
        fi
    done
}

# 1. FP-related standard headers.
scan "fp-header" \
    '#[[:space:]]*include[[:space:]]*[<"](cmath|math\.h|fenv\.h|cfenv|cfloat|float\.h)[>"]'

# 2. libm / std:: math function calls. The name must be at a token
#    boundary: preceded by start-of-line or a non-identifier char, and
#    followed by `(`. This prevents `sf64_sqrt` and `trunc_bits` from
#    matching `sqrt` / `trunc`.
math_names='sqrt|cbrt|hypot|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|exp|exp2|exp10|expm1|log|log2|log10|log1p|pow|fabs|floor|ceil|round|trunc|fma|fmod|remainder|ldexp|frexp|modf|scalbn|scalbln|nearbyint|rint|lround|llround|lrint|llrint|copysign|nextafter|nexttoward|tgamma|lgamma|erf|erfc|fmin|fmax|fdim'
scan "libm-call" \
    "(^|[^A-Za-z0-9_])(std::)?(${math_names})[[:space:]]*\\("

# 3. Compiler FP builtins.
scan "builtin-fp" \
    "__builtin_(${math_names})[[:space:]]*\\("

# 4. static_cast<double/float>(integer-looking).
scan "int-to-double-cast" \
    'static_cast<[[:space:]]*(double|float)[[:space:]]*>[[:space:]]*\([[:space:]]*(-?[0-9]|[a-zA-Z_][a-zA-Z0-9_]*[[:space:]]*[,)]|\([[:space:]]*(u?int(8|16|32|64)_t|int|unsigned|long|short|size_t|ptrdiff_t))'

# 5. C-style casts to double/float followed by an expression head.
scan "c-style-fp-cast" \
    '\((double|float)\)[[:space:]]*(-?[0-9]|[a-zA-Z_])'

# 6. FP-contract / fast-math escape pragmas.
scan "fp-contract-escape" \
    '#[[:space:]]*pragma[[:space:]]+(STDC[[:space:]]+FP_CONTRACT|GCC[[:space:]]+optimize|clang[[:space:]]+fp[[:space:]]+contract)|__attribute__\(\(optimize'

# 7. Raw comparisons against a floating-point literal. FP literal requires
#    an explicit decimal point with digits on both sides OR an exponent
#    marker — so integer-literal comparisons (`(q & 2) != 0`) are skipped.
#    Any match should become a call through sleef::lt_ / gt_ / eq_ / etc
#    (or a bit-level integer comparison on bits_of(x) in the core).
scan "fp-lit-compare" \
    '[a-zA-Z_)][[:space:]]*(<|>|<=|>=|==|!=)[[:space:]]*(-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?|-?[0-9]+[eE][-+]?[0-9]+)\b'

# 8. Raw comparisons involving a double-typed expression where neither
#    operand is an FP literal. Catches the pattern that slipped through
#    rule 7 during the original carve-out sweep — e.g.
#    `if (x.hi < y.hi)`, `if (sf64_fabs(x) < sf64_fabs(y))`. The hooks:
#      - `.hi` / `.lo` / `.dd` — DD struct fields are doubles by definition
#      - `sf64_fabs(…)` / `sf64_neg(…)` / `dd_to_d(…)` — return double
#      - `from_bits(…)` — reinterprets uint64_t bits as double
#    If any of these appears immediately followed by a relational operator,
#    the line is doing a host-FPU comparison and must use the
#    `sleef::lt_/le_/gt_/ge_/eq_/ne_` helpers instead.
#    Heuristic note: integer-typed structs that happen to have `.hi` /
#    `.lo` fields would false-positive here, but the codebase has none.
scan "fp-var-compare" \
    '(\.(hi|lo|dd)|sf64_fabs\([^()]*\)|sf64_neg\([^()]*\)|dd_to_d\([^()]*\)|from_bits\([^()]*\))[[:space:]]*(<=|>=|==|!=|<[^<=]|>[^>=])'

if [ "$findings" -gt 0 ]; then
    echo "" >&2
    echo "check_no_host_fp: $findings finding(s) — see CLAUDE.md § Hard constraints." >&2
    echo "If you believe a finding is a false positive, fix the pattern in" >&2
    echo "scripts/check_no_host_fp.sh rather than suppressing the hit." >&2
    exit 1
fi

echo "check_no_host_fp: clean (${#files[@]} files scanned, full src/ tree)"
