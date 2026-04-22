# Contributing to soft-fp64

Bug-exact integer fp64 has narrow tolerance for shortcuts. Read this
before changing anything in `src/`, `tests/`, `bench/`, or
`.github/workflows/`.

The full integrity contract — what is allowed, what is forbidden, and
what the automated guards check — lives in this file so every
contributor sees the same rules across machines. Local agent
configuration (Claude Code, Cursor, etc.) may add to these rules but
not relax them.

## Build + test

```bash
cmake -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`prek run --all-files` must be clean before pushing. CI runs the same
hooks (clang-format pinned to v19.1.7, the no-host-FP guard, whitespace
hygiene, large-file guard, private-key detection).

## Hard constraints

These are the rules that the design relies on. Breaking any of them is
breaking the library, regardless of whether tests pass.

### No host-FPU dependency in `src/`

`sf64_*` must be bit-correct on a device with no fp64 unit. Every
`+ - * / < > <= >= == != (double)int` in `src/` goes through an
`sf64_*` ABI call — arithmetic via `sf64_add`/`sub`/`mul`/`div`/`fma`,
relational comparisons via the `sleef::lt_/le_/gt_/ge_/eq_/ne_` helpers
in `src/sleef/sleef_common.h` (which wrap `sf64_fcmp` with the LLVM
predicate encoding), and integer→fp via the `sf64_from_i{8,16,32,64}` /
`sf64_from_u{…}` matrix.

The SLEEF transcendentals are a **port** of upstream SLEEF 3.6, not a
verbatim drop. The port completes the host-FPU removal that upstream
does not do. **There is no carve-out.** If upstream SLEEF structure
suggests a raw `double` op, the port rewrites it to `sf64_*`.

The mechanical guard is `scripts/check_no_host_fp.sh`. It scans every
`.cpp` / `.h` under `src/` (no exclusions) for `<cmath>` / `<math.h>`
/ `<fenv.h>` includes, libm / `std::` math calls, `__builtin_` FP
intrinsics, `static_cast<double>(int…)`, C-style `(double)`/`(float)`
casts, fp-contract / fast-math escape pragmas, and raw comparisons
against an FP literal. It runs as the `no-host-fp` hook in
`.pre-commit-config.yaml`. If a finding is a genuine false positive,
fix the regex in the script — do not suppress the hit or add an
allow-list entry.

### No fp-contract escape

`CMakeLists.txt` sets `-ffp-contract=off` globally on the library
target so the compiler never fuses `a*b+c` into `llvm.fma.f64` (which
would silently change bits and create a link cycle since `sf64_fma`
is implemented on top of `sf64_add` / `sf64_mul`). Do not add
`#pragma STDC FP_CONTRACT ON`, `#pragma clang fp contract(fast)`,
`__attribute__((optimize))`, `set_source_files_properties` overrides,
or any other source-level escape. The guard script flags these.

### Public ABI is `sf64_*` only

`-fvisibility=hidden -fvisibility-inlines-hidden` is set on the
library target. Cross-TU internal helpers use the `sf64_internal_`
prefix and `__attribute__((visibility("hidden")))`. The
`install-smoke` CI job greps `nm -g` on `libsoft_fp64.a` and fails
if any non-`sf64_*` symbol escapes (`__clang_call_terminate` is the
only allowed exception).

## Integrity rules

These rules exist because the alternative — silently weakening tests,
laundering numerical regressions through "refactor" commits, or
stretching the definition of "passing" — destroys the value of the
test suite. Reviewers, human or AI, are expected to enforce them.

**Never, under any justification:**

- Widen a ULP tolerance, replace `ASSERT_EQ(bits, ...)` with
  `ASSERT_NEAR` / `EXPECT_DOUBLE_EQ` / `approx_equal`, or flatten NaN
  payloads / signed-zero in a comparator. Bit-exact is bit-exact.
- Modify, comment out, skip, `GTEST_SKIP`, `DISABLED_`-prefix, `#if
  0`, `if (on_ci()) return`, or delete rows from TestFloat / MPFR
  vectors. Move cases between test files to escape a tighter tier.
  Add `// TODO: fix later` or `// known-broken`.
- Shrink a random corpus, change the PRNG seed until the failure
  vanishes, narrow the exhaustive `f32 ↔ f64` range, or silently
  shard.
- Drop MPFR precision below 200 bits, round the reference through
  `double`, or replace an MPFR call with libm.
- Special-case inputs with magic-constant branches keyed to failing
  corpus rows — even dressed up as IEEE-754 classification.
- Call host fp64 from `src/` by any route: `std::sqrt`, libm,
  `__builtin_sqrt` / `__builtin_fma` / `__builtin_sin`, inline asm,
  hardware intrinsics, `volatile double` tricks, `<cmath>` /
  `<math.h>` includes. Tests may use host fp64 as an oracle; `src/`
  never uses it as an implementation.
- Tune `internal.h` rounding primitives, Mesa `s_*` helpers, or SLEEF
  constants to match an observed output. Ports stay faithful to
  upstream; if the output is wrong, the port is wrong.
- Stub `sf64_*` with weak symbols, link-order tricks, `LD_PRELOAD`,
  or test-local overrides. Tests must call the real library.
- Populate an expected-output table from the current implementation.
  Any expected-bits table cites its source (TestFloat row, MPFR
  computation, IEEE-754 derivation) in a comment.
- Update `bench/baseline.json` to erase a regression. Regenerate only
  on hardware-class change or a reviewed deliberate perf change.
- Disable / remove fuzz targets, sanitizer options, or ctest entries
  because they fail. `ASAN_OPTIONS=detect_leaks=0` (MPFR caches leak
  past `main`) is the only exemption.
- Claim "tests pass" via `ctest -E <failing>`, `ctest -R <subset>`,
  `GTEST_FILTER=-<failing>`, or selective re-runs. "Green" means a
  full unfiltered `ctest --test-dir build --output-on-failure`.
- Land a numerical change inside a commit labeled `refactor` /
  `style`. Any line move inside `src/` is a numerical claim; back it
  with a ctest run.
- Fabricate tool output. If you did not run the command in this
  session, you do not have the result.

### Reviewer claims

- **Reproduce before acting.** If a reviewer (human or AI) reports "N
  sites do X" or "the code cheats by Y," reproduce the finding with a
  concrete command before changing code. Counting the token `double`
  across ABI signatures and calling that a "host-FPU dependency" is a
  category error. The guard script is authoritative on what counts as
  a host-FPU use; if the script says clean and a reviewer says dirty,
  the reviewer owes a `file:line` citation that the script misses —
  patch the script to catch it, then act. Do not delete or rewrite
  correct ABI code to pacify an unverified claim.
- **Scope-correct claims must not be dismissed.** If the reviewer's
  grep target is `src/sleef/` and yours was `src/` minus
  `src/sleef/`, the reviewer is right and you are wrong — re-run the
  scan with their scope before publishing a rebuttal.

### Ports are not carve-outs

Files derived from upstream but hand-modified into the repo (e.g.
`src/sleef/`, the Mesa-derived arithmetic in `src/arithmetic.cpp`)
are subject to every rule including the no-host-FPU rule. Vendored
code lives in a `vendor/` subdirectory with a pinned upstream SHA
(see `tests/testfloat/vendor/`, `bench/external/`); anything outside
those directories is first-party. "We followed upstream structure"
is not a defense — if upstream uses raw `double` arithmetic and we
ship a port, the port rewrites every host-FPU op.

### Sweep windows must match claimed sampling

`sweep2_uniform` over `[10⁻ᴺ, 10⁺ᴺ]` for `N ≥ 3` is degenerate:
linear sampling concentrates ~99% of points in the top decade, so
the lower decades of the advertised window are effectively
unexercised. Wide-decade windows must use `sweep2_log` (or
`sweep1_log`). The `record()` function in
`tests/mpfr/test_mpfr_diff.cpp` and `tests/test_coverage_mpfr.cpp`
enforces this with a trivial-match gate (>25% of (NaN,NaN) or
matching-sign-(inf,inf) fails the sweep) — if the gate fires, fix
the sampling, do not raise the threshold.

### Test claims and code agree

Every numeric claim in `README.md`, `CHANGELOG.md`, `TODO.md`, or
header Doxygen must be traceable to a specific test with a matching
bound. Test-corpus sizes named in file headers are contracts. Any
discrepancy between "`kN = 10000`" in the header comment and `for
(int i = 0; i < 4096; ++i)` in the body is a bug, not a shortcut.
The `readme-claim-parity` hook in `.pre-commit-config.yaml` checks
that sweep windows quoted in `README.md` appear verbatim in the test
source.

## Reporting

- No "tests pass" without pasting the final ctest `Passed`/`Failed`
  line.
- No "bench is neutral" without pasting the `compare.py` table.
- End-of-turn summaries must call out edits under `tests/`, `fuzz/`,
  `bench/baseline.json`, `bench/compare.py`, `src/sleef/`, or
  compile/sanitizer/ctest args in `CMakeLists.txt`.
- Ban "should work" / "should be bit-exact" — run it.

When something is broken you can't fix, say so explicitly ("this
failure is real, I don't have a fix, leaving the test red") and stop.
Landing green CI at the cost of a weaker suite is the worst outcome.
