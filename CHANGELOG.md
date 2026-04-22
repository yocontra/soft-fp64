# Changelog

All notable changes to `soft-fp64` land here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every numeric claim in this file traces to a specific CI-gated sweep —
no prose-only bounds. See `README.md` for the full precision table.

## [1.0.0] — unreleased

First stable release. Ships a complete `sf64_*` IEEE-754 binary64 surface
built entirely on 32/64-bit integer ops (no host-FPU dependency) plus the
AdaptiveCpp Metal adapter that lets fp64-gated GPU kernels on Apple
Silicon resolve against soft-fp64 instead of trapping.

### Added

- **`sf64_*` public ABI.** `extern "C"` entry points in
  `include/soft_fp64/soft_f64.h`, hidden-by-default visibility, enforced
  by the `install-smoke` CI job (`nm -g` on the archive rejects any
  non-`sf64_*` export).
- **Arithmetic — BIT_EXACT.** `add`, `sub`, `mul`, `div`, `rem`, `neg`,
  `sqrt`, `fma`, `fmod`, `remainder`. Gated against host FPU + Berkeley
  TestFloat 3e vectors by `tests/test_arithmetic_exact.cpp` and
  `tests/test_sqrt_fma_exact.cpp`; `fmod` / `remainder` additionally
  gated 0-ULP against MPFR in `tests/mpfr/test_mpfr_diff.cpp` at tier
  `BIT_EXACT = 0` across x ∈ [-1e15, 1e15], y ∈ [1, 1e10] (quotient
  bit-count ≳ 2⁵⁰).
- **Conversion — BIT_EXACT.** Full matrix `i{8,16,32,64} ↔ f64`,
  `u{8,16,32,64} ↔ f64`, `f32 ↔ f64`, including exhaustive 2³² `f32 →
  f64 → f32` round-trip. Gated by `tests/test_convert_widths.cpp`.
- **Compare + classify — BIT_EXACT.** All 16 IEEE-754 `fcmp` predicates,
  `fmin` / `fmax` (IEEE and non-IEEE variants including `fmin_precise` /
  `fmax_precise`), `fdim`, `maxmag`, `minmag`, `nextafter`, `hypot`,
  `isnan`, `isinf`, `isfinite`, `isnormal`, `signbit`, `fabs`,
  `copysign`. Gated by `tests/test_compare_all_predicates.cpp` +
  TestFloat vectors.
- **Rounding — BIT_EXACT.** `floor`, `ceil`, `trunc`, `rint`, `round`,
  `fract`, `modf`, `ldexp`, `frexp`, `ilogb`, `logb`. Gated by
  `tests/test_rounding_edges.cpp`.
- **Transcendentals — U10 (≤4 ULP vs MPFR 200-bit).** `sin`, `cos`,
  `sincos`, `asin`, `acos`, `atan`, `exp`, `exp2`, `exp10`, `expm1`,
  `log`, `log2`, `log10`, `log1p`, `cbrt`, `cosh`, `acosh`, `atanh`.
  Gated by `tests/mpfr/test_mpfr_diff.cpp` (independent oracle)
  plus `tests/test_transcendental_1ulp.cpp` (libm consistency).
- **Transcendentals — U35 (≤8 ULP vs MPFR 200-bit).** `tan`, `atan2`,
  `sinh`, `tanh`, `asinh`, `pi`-variants (`sinpi`, `cospi`, `tanpi`,
  `asinpi`, `acospi`, `atanpi`, `atan2pi`), and the `pow` / `rootn`
  family (`pow`, `powr`, `pown`, `rootn`) over three overlapping
  bounded windows.
  Near-unit-base × huge-exponent corner is documented as
  out-of-region, tracked in `TODO.md`.
- **`erf` / `erfc` / `tgamma` / `lgamma` / `lgamma_r` — GAMMA
  (≤1024 ULP vs MPFR 200-bit).** Shippable ranges per the README tiers
  table. `lgamma` zero-crossings `[0.5, 3)` are parked in the
  report-only `tests/experimental/` sweep (not a shipped tier claim)
  pending the `logk_dd` rewrite tracked in `TODO.md`.
- **CI oracle stack.** Host FPU, Berkeley TestFloat 3e
  (`tests/testfloat/`), MPFR 200-bit (`tests/mpfr/`,
  `ORACLE_PREC = 200`), and libFuzzer crash-hunt targets (`fuzz/`).
- **Bench regression gate.** `bench_soft_fp64` + `bench/compare.py`
  enforce 20% threshold against `bench/baseline.json` (macos-14 M-series)
  on every PR.
- **CMake + pkg-config integration.** `find_package(soft_fp64 CONFIG)`
  and `pkg-config --cflags --libs soft_fp64`; `install-smoke` CI job
  exercises both.
- **`scripts/pre_push.sh` local guard.** Runs full `ctest` +
  `clang-format-19 --dry-run --Werror` + the bench regression gate at
  `--threshold=0.10`.
- **AdaptiveCpp Metal SSCP adapter** (`adapters/acpp_metal/`). Opt-in
  (`-DSOFT_FP64_BUILD_ACPP_METAL_ADAPTER=ON`). Stages a flat directory
  of soft-fp64 sources plus one-line forwarders for every required
  `__acpp_sscp_soft_f64_*` primitive (all 23) and the full optional
  `__acpp_sscp_*_f64` math surface (transcendentals, rounding, hypot,
  fmod / remainder, fract / frexp / modf / ldexp / ilogb, pown /
  rootn, classify predicates). Ready to be consumed via
  `-DACPP_METAL_EXTERNAL_FP64_DIR=<staged dir>` by downstream
  consumers such as pg_accel on Apple Silicon. Pinned against
  AdaptiveCpp `fork-safe-metal` at
  `c86d474a3f1fb06705679efa527ea262b5a991cf`; CI smoke job
  `acpp-metal-smoke` rebuilds `libkernel-metal-bitcode` against the
  staged dir on macos-14. Adapter emits no public ABI — forwarders
  live only in the Metal libkernel bitcode, `install-smoke`'s
  `sf64_*`-only archive grep is unaffected.

### Integrity guarantees

- `sf64_*` returns identical bits on every target for every input.
  No host-FPU dependency (SLEEF `src/sleef/*.cpp` raw-`double` helpers
  are the single carve-out, contractually lowered by the SSCP emitter
  on fp64-less targets).
- `-ffp-contract=off` is enforced globally on the library target to
  prevent `a*b+c` from fusing into `llvm.fma.f64` and closing a
  link-time cycle against `sf64_fma`.
- No silent skip paths in oracle sweeps. Every test that declares a
  corpus size in its file header runs that size.

### Not included — tracked for 1.1+ in `TODO.md`

- Non-RNE rounding modes (`sf64_*_r(mode, …)`).
- IEEE exception flags + thread-local fenv.
- sNaN payload preservation.
- `soft-fp128` sibling package.
- `logk_dd` DD-Horner rewrite (moves worst-case `pow` from ~40 ULP to
  ≤4 ULP in the near-unit-base × huge-exponent corner; closes the
  `lgamma (0.5, 3)` zero-crossing sweep).

[1.0.0]: https://github.com/contra/soft-fp64/releases/tag/v1.0.0
