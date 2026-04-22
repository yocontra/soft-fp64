# TODO

Single source of truth for open work. Items closed at 1.0 / 1.1 are
recorded in `CHANGELOG.md`, not here.

## Pre-1.1 — in flight

Currently uncommitted in the working tree on top of commit `c3c1b90`.
These are polish on the 1.1 fenv / adapter surface that was bundled
into commit `580c781`; they land before the 1.1 tag.

- **Fenv raise-site coverage across arithmetic / convert / sqrt-fma.**
  `src/arithmetic.cpp`, `src/convert.cpp`, `src/sqrt_fma.cpp` add
  INEXACT / UNDERFLOW / OVERFLOW / INVALID raise sites at the
  IEEE-754 §7 locations; `src/internal_fenv.h` now carries a
  two-raise-path scheme (a hot inline accumulator macro plus the
  out-of-line `SF64_FE_RAISE`) so the INEXACT fast path does not
  touch TLS storage on every op. Before commit: run the TestFloat
  `fl2` column gate over the full 7.16M-vector corpus under
  `SOFT_FP64_FENV=tls` and confirm every expected flag bit is
  raised — the CI cell exists but needs a green pass after the
  new raise sites land.
- **ACPP Metal adapter staging picks up `SOFT_FP64_FENV_MODE`.**
  `adapters/acpp_metal/CMakeLists.txt` and
  `cmake/rewrite_sleef_include.cmake` need to propagate the
  top-level `SOFT_FP64_FENV` build option into the staged source
  tree. Without it the adapter's Metal bitcode always compiles with
  `SOFT_FP64_FENV_MODE=0` (disabled) regardless of the core
  configuration, so fenv flag raising silently no-ops on Metal.

## Post-1.1

### Numerical

- **`sf64_lgamma` zero-crossings on `(0.5, 3)`.** `lgamma(x)` vanishes
  at `x = 1` and `x = 2`; near those zeros the result is O(1e-5) but
  the absolute error floor of any log-of-Γ path is O(ulp(1)) ≈ 2.2e-16,
  so the ULP ratio blows past GAMMA=1024 even with a perfectly
  computed log ingredient. v1.1's `logk_dd` DD-Horner rewrite
  confirmed the issue is algorithmic, not ingredient-precision. The
  proper fix is a **zero-centered Taylor expansion** around `x=1` and
  `x=2` — a branch inside `sf64_lgamma` that detects the vanishing
  regime and returns `(x-1)·P₁(x)` or `(x-2)·P₂(x)` with
  coefficients computed from the known series for `ψ(x)`. Currently
  parked report-only in `tests/experimental/experimental_precision.cpp`;
  gated promotion to GAMMA tier requires the rewrite to land first.

### Feature surface (not yet implemented)

- **`SOFT_FP64_FENV=explicit` caller-provided state ABI.** v1.1 reserves
  the `explicit` mode in CMake but compiles it identically to `disabled`
  (zero-cost no-op raise sites, `sf64_fe_*` surface present but
  stateless). The target shape is `sf64_fe_*` variants that take an
  `sf64_fe_state_t*` directly, enabling GPU/freestanding kernels that
  can't rely on `thread_local`. Requires a parallel ABI to avoid
  breaking 1.x consumers. TestFloat `run_testfloat.cpp` skips the
  7.16M-vector flag gate under explicit mode today; wire it up once
  the surface lands.
- **sNaN payload preservation.** v1.0–v1.1 quiet sNaN on entry (sNaN →
  qNaN with canonical payload). v1.1 raises `SF64_FE_INVALID` on that
  entry when fenv is enabled. Consumers needing §6.2 full payload
  preservation require a `SOFT_FP64_SNAN_PROPAGATE` build option that
  preserves the signalling payload bits through the quiet-bit force.
  TestFloat has dedicated sNaN vectors; `tests/testfloat/run_testfloat.cpp`
  currently skips them (documented carve-out). Target: v1.2.
- **`soft-fp128` sibling.** Same design playbook (Mesa arithmetic
  port + SLEEF transcendentals + TestFloat + MPFR oracle) extended to
  113-bit significand. Storage wrapper + full conversion matrix
  (`f64 ↔ f128`, `i128 ↔ f128`), u10 transcendentals vs MPFR
  300-bit. Likely ships as a separate package once fp64 stabilizes.

### ACPP Metal adapter follow-ups

- **`__acpp_sscp_lgamma_r_f64` forwarding.** The adapter currently
  leaves the trap-stub in place (see
  `adapters/acpp_metal/README.md` → "Skipped"). Blocks on adding a
  `sf64_lgamma_r` core entry point — computing the Γ sign from
  `sf64_tgamma(x)` round-trips through an overflow-prone path for
  `|x| > 170`. Will land with whichever core release exposes
  `sf64_lgamma_r`.
- **`_r`-variant forwarders.** Once the non-RNE `sf64_*_r` surface
  from v1.1 is stable on the Metal target, the adapter gains
  one-line `__acpp_sscp_soft_f64_*_r` forwarders. No core change
  required — pure forwarding, zero ULP to add.
- **`sf64_fe_*` surface on Metal.** The adapter may optionally
  re-export the fenv surface for kernels that care about
  accumulated-flag reporting. `SOFT_FP64_FENV=disabled` stays the
  default on GPU targets (no `thread_local` support on Metal SSCP);
  an `explicit`-state ABI (v1.2+) lines this up.

## Not on the roadmap

- **fp16 / bfloat16.** Different design space — table-driven
  implementations win at that precision. No fp64 synergy.
- **Decimal FP (IEEE-754 §3.5).** Different rounding philosophy; see
  `libdfp` or Intel's DFP library.
- **Complex-number math** (`csin`, `clog`, etc). Pure wrapper work on
  top of real scalars; belongs in a consumer.
- **Guest-FPU emulation glue.** `fenv` compat, `softfp` ABI lowering,
  calling-convention shims belong in the frontend (compiler runtime,
  emulator), not here. `sf64_*` is the contract; how you call it is
  your problem.
- **Runtime CPU dispatch for fast paths.** The point is
  architecture-independent bit-exactness. If you have fast native
  fp64, you don't need this library.
