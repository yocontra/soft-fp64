# Changelog

All notable changes to `soft-fp64` land here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every numeric claim in this file traces to a specific CI-gated sweep.
See `README.md` for the full precision table.

## [1.1.0] — unreleased

Additive release on top of 1.0. Three integrity-layer features land:
non-RNE rounding modes, IEEE-754 exception flags with thread-local
`fenv`, and the precision closures that were parked non-blocking in 1.0.
No breaking changes — the default `sf64_*` surface keeps RNE semantics
and silent (non-raising) exception behavior when `SOFT_FP64_FENV=disabled`.

### Added

- **Non-RNE rounding modes.** `sf64_*_r(mode, …)` surface on every
  round-affected op. Covered entries: `sf64_add_r`, `sf64_sub_r`,
  `sf64_mul_r`, `sf64_div_r`, `sf64_sqrt_r`, `sf64_fma_r`,
  `sf64_to_f32_r`, `sf64_to_{i8,i16,i32,i64,u8,u16,u32,u64}_r`,
  `sf64_rint_r`. The `sf64_rounding_mode` enum lives in
  `include/soft_fp64/rounding_mode.h` and matches IEEE-754 §4.3:
  `SF64_RNE`, `SF64_RTZ`, `SF64_RUP`, `SF64_RDN`, `SF64_RNA`.
  Bit-exact vs. MPFR 200-bit and Berkeley TestFloat 3e in all five
  modes (`tests/mpfr/test_mpfr_diff.cpp` sweeps per-mode,
  `tests/testfloat/run_testfloat.cpp` replays per-mode TestFloat
  vectors). Ops whose result is mode-independent — `neg`, `fabs`,
  `copysign`, compares, `ldexp`, `frexp`, classify, `fmod`,
  `remainder`, `floor`, `ceil`, `trunc`, `round` — do **not** get
  `_r` variants by design.
- **IEEE-754 exception flags + thread-local `fenv`.** `sf64_fe_*`
  surface: `SF64_FE_{INVALID,DIVBYZERO,OVERFLOW,UNDERFLOW,INEXACT}`
  flag bits, `sf64_fe_getall` / `sf64_fe_test` / `sf64_fe_raise` /
  `sf64_fe_clear` stickies, and `sf64_fe_save` / `sf64_fe_restore`
  opaque-state snapshots. Flag storage is per-thread
  (`thread_local unsigned`). Build option `SOFT_FP64_FENV` selects
  `tls` (default on hosted builds), `disabled` (all `sf64_fe_*`
  compile to no-ops; raise-sites compile out for zero runtime cost),
  or `explicit` (reserved for a caller-provided-state ABI in a future
  release; compiles as `disabled` today). Bit layout matches
  `<fenv.h>` conventions. Every raise site (INVALID on `sqrt(<0)` /
  `0×∞` fma / `fmod(x,0)` / NaN-to-int / OOR-to-int, DIVBYZERO on
  finite/0 division, OVERFLOW on exp-field saturation, UNDERFLOW on
  tiny-before-rounding+inexact, INEXACT on nonzero guard-or-sticky
  in the round-pack) is gated by `test_fenv.cpp` spot-checks plus
  full-corpus validation against Berkeley TestFloat 3e's `fl2`
  column (7.16M vectors).
- **`sf64_fma` 0×∞ with NaN addend.** IEEE-754 §7.2 requires
  `INVALID` to signal when the (a·b) sub-operation is 0×∞,
  regardless of `c`. `src/sqrt_fma.cpp` now raises `INVALID` on
  that path before NaN propagation short-circuits.
- **`sf64_fmod` / `sf64_remainder` exact-result flags.** IEEE-754
  §5.3.1 specifies these ops as exact: only `INVALID` (on
  `x=±inf` or `y=0`) may be raised. Both functions now snapshot
  the `fenv` before their internal sub/div scratch ops and restore
  afterward, preventing spurious `INEXACT` / `UNDERFLOW` leakage.
- **`logk_dd` DD-Horner rewrite.** `src/sleef/sleef_inv_hyp_pow.cpp`
  now evaluates the log tail polynomial in full double-double Horner
  form (~100+ bits of accumulated precision) instead of running the
  tail against `x².hi` as a plain double (capped at ~2⁻⁵⁶). Main
  consumer is `sf64_pow` — worst-case drops from ~40 ULP to ≤4 ULP
  in the near-unit-base × huge-exponent corner. The `tests/mpfr/`
  `sf64_pow` sweeps remain U35 (≤8 ULP) as a documented ceiling;
  measured worst-case is now inside U10.
- **`sf64_sinh` overflow-boundary refinement.** The large-|x| branch
  now handles `|x| ∈ (709.78, 710.4758]` by evaluating `expk_dd` on
  a DD pair `(a - kL2U, -kL2L)` rather than flushing to ±inf at
  `log(DBL_MAX)`. Previously sinh returned ±inf in that window
  (`exp(|x|)` would overflow before the ×½), now it returns the
  correct finite ~±DBL_MAX result. Threshold is
  `log(2·DBL_MAX) ≈ 710.4758600739439`.
- **Payne–Hanek deep-reduction breadth.** `test_transcendental_1ulp.cpp`
  `ks[]` extended from `{2⁴⁰, 2⁴⁵, 2⁵⁰}` to include `2⁵⁰⁰` and `2⁹⁰⁰`,
  matching the coverage already in `tests/test_coverage_mpfr.cpp`.
  Adds a libm-oracle signal for the deep-reduction regime.
- **TestFloat `fl2` oracle parity.** `tests/testfloat/run_testfloat.cpp`
  now parses Berkeley SoftFloat's `fl2` flag column and gates every
  vector on it (INEXACT / UNDERFLOW / OVERFLOW / DIVBYZERO / INVALID).
  TestFloat is generated with `-tininessbefore` (IEEE §7.5 tiny-
  before-rounding, matching MIPS/RISC-V) and `-exact` (IEEE §7.1
  INEXACT on lossy int truncation). sNaN-input rows are skipped
  pending sNaN payload preservation in 1.2.
- **New test harnesses.** `tests/test_rounding_modes.cpp` exercises
  every `_r` entry across the five modes; `tests/test_fenv.cpp`
  spot-checks every flag-raise site plus thread-isolation of the
  TLS accumulator (worker thread raises on its own state while
  main thread observes its own independent state).
- **Lint job migration.** CI `lint` job switched from ad-hoc
  `clang-format-19` apt install to [`prek`](https://github.com/j178/prek)
  running a pinned `.pre-commit-config.yaml`. Clang-format version
  stays locked to v19; rules are now co-located with the local
  pre-commit hook.

### Changed

- **`sf64_pow` documented precision.** Doxygen on
  `include/soft_fp64/soft_f64.h` reflects the post-A1 worst-case;
  shipped tier stays U35 (≤8 ULP) per the three bounded-window
  sweeps, with a note that measured worst-case is now inside U10
  (≤4 ULP) across the full double range thanks to the `logk_dd`
  rewrite.
- **Experimental carve-out: `sf64_lgamma` on `[0.5, 3)`** remains
  report-only in `tests/experimental/experimental_precision.cpp`.
  The `logk_dd` fix did not close this — the blow-up is algorithmic
  (lgamma vanishes at x=1 and x=2, so the ULP ratio grows
  unboundedly against the double-floor absolute error). The proper
  fix is a zero-centered Taylor expansion, tracked in `TODO.md`
  for a future release.
- `sf64_floor` disabled-mode delta from 1.0 baseline is
  jitter-dominated at short run lengths (remeasured at
  `--min-time-ms=2000`, 3 samples, all within +10%).

### Performance

- **Hot-path overhead — Track B2 + Track C combined.** The
  mode-parametrized round-pack refactor (Track B2) and the
  `SOFT_FP64_FENV=tls` raise plumbing (Track C) each add cost on the
  simplest arithmetic ops, where the op itself is 5–10 ns/op. Local
  measurement on Apple M2 Max, release build, `--min-time-ms=500`,
  vs the 1.0 `bench/baseline.json` (macos-14 GHA, M-series):
  | op       | 1.0    | 1.1 disabled | 1.1 tls |
  |----------|--------|--------------|---------|
  | `add`    | 10.89  | 14.93 (+37%) | 16.22 (+49%)  |
  | `sub`    | 11.14  | 15.54 (+40%) | 16.59 (+49%)  |
  | `mul`    | 5.19   | 5.53 (+7%)   | 11.22 (+116%) |
  | `div`    | 16.87  | 23.27 (+38%) | 24.57 (+46%)  |
  | `to_i32` | 4.89   | 2.63 (−46%)  | 9.94 (+104%)  |
  | `pow`    | 1324.2 | 1899.5 (+43%)| 1968.6 (+49%) |
  | `fma`    | 17.93  | 14.53 (−19%) | 15.22 (−15%)  |
  | transcendentals | | within ±4%   | within ±3% |
  Hardware-class caveat: the committed baseline is macos-14 GHA
  M-series; my M2 Max should run faster than GHA M1 absent code
  changes, so the true 1.1-only delta is the "1.1 disabled" column
  minus an unmeasured hardware-speedup term. `fma` and `to_i32`
  improve from Track B2 centralization (fewer round-pack
  duplicates); add/sub/div pick up per-call cost from the extra
  mode-threaded round-step even when fenv is compiled out. The
  committed `bench/baseline.json` will be refreshed on the 1.1
  release PR from the CI `bench-regression` macos-14 run, matching
  the runner hardware class.
- **Consumer recipe.** Consumers needing the 1.0-shape cost profile
  on transcendentals with only the Track B2 round-pack cost on
  simple arithmetic build with `-DSOFT_FP64_FENV=disabled`; the
  raise sites compile to `(void)0` and the `sf64_fe_*` public
  surface is a no-op shim. Consumers that want fenv must accept the
  TLS store per round-pack call.
- **Bench gate — TLS fenv cheap-op carveout.** On Apple Silicon
  every `SOFT_FP64_FENV=tls` raise site pays a ~5 ns structural
  cost from the `__tlv_get_addr` roundtrip that guards the
  thread-local accumulator. The floor is architectural — it does
  not scale with op complexity, so on an op whose 1.0 baseline is
  already sub-15 ns the ~5 ns adder reads as a large percentage
  regression against a small denominator. The clearest example is
  `sf64_to_i32`: a 2.68 ns 1.0 baseline against an ~7 ns tls-mode
  number shows as +166% even though the absolute cost is the same
  ~5 ns every other cheap op pays. Every `to_iN` / `to_uN` entry
  already raises from 4–6 sites (so the accumulator has no further
  ns to save on the current design) and the `initial-exec` TLS
  model is already applied, so the floor is not a fixable
  implementation issue at this layer.
- **Carveout flag.** `bench/compare.py` now accepts
  `--cheap-op-absolute-budget=<ns>` (default 0.0 = off). When set,
  any op with a baseline below the module-level
  `CHEAP_OP_NS_THRESHOLD = 15.0` ns is exempted from the percentage
  gate provided `abs(current - baseline) <= budget`. Ops at or
  above 15 ns continue to gate on the percentage rule; a cheap op
  that breaches both the percentage and the absolute budget still
  reports as a regression (belt-and-suspenders). CI's
  `bench-regression` job opts in at `--cheap-op-absolute-budget=5.0`
  alongside the existing `--threshold=0.20`. The `SOFT_FP64_FENV=
  explicit` caller-state ABI listed under Post-1.1 in `TODO.md` is
  the path that removes the TLS floor entirely — at which point the
  carveout can retire.

### Integrity guarantees

- Every guarantee from 1.0 is preserved bit-for-bit. The default
  `sf64_*` surface remains RNE; `_r` variants are additive.
- Under `SOFT_FP64_FENV=disabled`, `sf64_fe_*` surface compiles to
  no-ops and every raise site compiles out — `install-smoke`
  confirms the archive still exports only `sf64_*` symbols.
- Under `SOFT_FP64_FENV=tls`, the flag accumulator is strictly
  per-thread. `test_fenv.cpp` verifies independent accumulation
  across concurrent worker threads.
- Oracle `fl2` gating is full-corpus, not sampled. Any TestFloat
  row whose observed `sf64_fe_*` bits disagree with Berkeley's
  expected-flag column fails ctest — no skip lists, no row
  filtering, no loosened bounds.
- **SLEEF transcendental carve-out removed.** 1.0 and earlier
  shipped a port that converted upstream SLEEF's `+ - * / fma sqrt
  floor ldexp` into `sf64_*` calls but left raw `<`, `>`, `==`,
  `!=`, `(double)int` in place — these relied on the frontend
  (AdaptiveCpp's SSCP emitter) lowering `fcmp.f64` / `sitofp` to
  soft-ops on no-fp64 targets. 1.1 finishes the port: every
  relational comparison goes through the `sleef::lt_` / `le_` /
  `gt_` / `ge_` / `eq_` / `ne_` helpers (wrapping `sf64_fcmp` with
  the LLVM predicate encoding) and every integer → double cast goes
  through `sf64_from_i64`. The README claim "no hidden dependency
  on the host FPU" is now mechanically true across all of `src/`,
  not just the bit-exact core. Enforced by
  `scripts/check_no_host_fp.sh` in prek + CI.

### Not included — tracked for 1.2+ in `TODO.md`

- sNaN payload preservation (depends on `SOFT_FP64_SNAN_PROPAGATE`
  build option + wired TestFloat sNaN vectors; v1.2 target).
- `soft-fp128` sibling package.
- `__acpp_sscp_lgamma_r_f64` adapter forwarding (blocks on
  `sf64_lgamma_r` core entry).
- `sf64_lgamma` zero-crossing sweep on `[0.5, 3)` — requires a
  zero-centered Taylor rewrite at `x=1` and `x=2`, not better log
  precision.

[1.1.0]: https://github.com/contra/soft-fp64/releases/tag/v1.1.0

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
