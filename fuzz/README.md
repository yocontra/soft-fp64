# soft-fp64 fuzz targets

libFuzzer targets for arithmetic, sqrt/fma, transcendentals, fcmp, remainder,
and narrow-int conversions, plus an exhaustive `f32 <-> f64` round-trip test.

## What lives here

| File | Kind | What it does |
|---|---|---|
| `fuzz_arithmetic.cpp` | libFuzzer | `sf64_add/sub/mul/div`: commutativity, identities, trap-hunt. |
| `fuzz_sqrt_fma.cpp` | libFuzzer | `sf64_sqrt`, `sf64_fma`: NaN/Inf handling, `sqrt(x*x) ≈ |x|`. |
| `fuzz_transcendental.cpp` | libFuzzer | sin/cos/tan/exp/log/pow: special-case handling, coarse libm agreement. |
| `fuzz_fcmp.cpp` | libFuzzer | `sf64_fcmp(a, b, pred)` across all 16 predicates + NaN pairs. |
| `fuzz_remainder.cpp` | libFuzzer | `sf64_remainder(x, y)` agreement with libm, Sterbenz edges. |
| `fuzz_narrow_int.cpp` | libFuzzer | i8/i16/u8/u16 ↔ f64 conversions, saturation/rounding edges. |
| `exhaustive_f32_f64_roundtrip.cpp` | plain ctest | Every 32-bit f32 pattern through `sf64_from_f32` → `sf64_to_f32`. |

These targets are crash-hunts, not accuracy graders. ULP regressions
are covered by `tests/test_arithmetic_exact.cpp`,
`tests/test_sqrt_fma_exact.cpp`, `tests/test_transcendental_1ulp.cpp`,
and the MPFR differential harness. libFuzzer surfaces traps, sanitizer
findings, and qualitative divergences (NaN↔finite, sign-of-inf, wildly
different output) under coverage-guided input mutation.

## Building

### Default developer build (no fuzz)

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build
```

Fuzz targets are gated behind `-DSOFT_FP64_BUILD_FUZZ=ON`, so this path is
unaffected. The exhaustive round-trip test is **also** gated — the
`add_subdirectory(fuzz)` call only fires when the option is ON. If you want
the exhaustive test to run in a default build, re-run cmake with the option
enabled:

```bash
cmake -B build -DSOFT_FP64_BUILD_FUZZ=ON
cmake --build build --target exhaustive_f32_f64_roundtrip
ctest --test-dir build -R test_exhaustive_f32 --output-on-failure
```

Expected runtime on an M-series Mac: 5–15 minutes (uses up to 16 threads
via `std::thread::hardware_concurrency()`). The ctest timeout is 900s.

### libFuzzer build (requires clang)

libFuzzer is built into clang via `-fsanitize=fuzzer,address,undefined`,
so the compiler must be clang or AppleClang.

```bash
cmake -B build-fuzz \
    -DSOFT_FP64_BUILD_FUZZ=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-fuzz -j
```

This produces:

* `build-fuzz/fuzz/fuzz_arithmetic`
* `build-fuzz/fuzz/fuzz_sqrt_fma`
* `build-fuzz/fuzz/fuzz_transcendental`
* `build-fuzz/fuzz/exhaustive_f32_f64_roundtrip` (also built; ASan+UBSan enabled
  indirectly through inherited flags only if the user sets them — this target
  itself uses only `-O2 -ffp-contract=off`).

### Running a smoke fuzz (30s)

```bash
./build-fuzz/fuzz/fuzz_arithmetic     -max_total_time=30 -print_final_stats=1
./build-fuzz/fuzz/fuzz_sqrt_fma       -max_total_time=30 -print_final_stats=1
./build-fuzz/fuzz/fuzz_transcendental -max_total_time=30 -print_final_stats=1
```

### Running a long fuzz (e.g. overnight)

```bash
mkdir -p corpus/arithmetic
./build-fuzz/fuzz/fuzz_arithmetic corpus/arithmetic/ -max_total_time=28800 \
    -jobs=8 -workers=8
```

libFuzzer persists interesting inputs in `corpus/arithmetic/` across runs.
Crashes are saved as `crash-<sha1>` in the cwd — pass the file to the
binary to reproduce:

```bash
./build-fuzz/fuzz/fuzz_arithmetic crash-0123456789abcdef
```

## NaN policy (exhaustive round-trip)

`sf64_from_f32` and `sf64_to_f32` preserve finite, zero, and infinity f32
bit-patterns *bit-exactly*. For NaN inputs, the implementation may **quiet
on entry** and canonicalize the payload — this matches IEEE-754 §6.2 and
the ARMv8 / x86-SSE hardware behavior. The round-trip test reports:

* `nan_preserved` — NaN patterns whose bits survived the round-trip.
* `nan_canonicalized` — NaN patterns that round-tripped to a different NaN
  bit-pattern (still NaN, just different payload/quiet bit).
* `mismatches` — **BUG**: non-NaN pattern didn't round-trip, or NaN went to
  a finite value. Any mismatch fails the test.

## If the fuzzer finds something

1. libFuzzer writes `crash-<sha1>` to the cwd and prints a stack trace.
2. Minimize: `./fuzz_arithmetic -minimize_crash=1 crash-<sha1>` produces a
   smaller `minimized-from-<sha1>` reproducer.
3. Add the minimized reproducer as a golden input to
   `tests/test_arithmetic_exact.cpp` (or the appropriate test) so regressions
   get caught by the normal `just test` run.
4. Fix the bug in `src/` and re-run the fuzzer to confirm.
