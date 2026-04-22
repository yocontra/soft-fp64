# TODO

Single source of truth for open work. Items closed at 1.0 are recorded
in `CHANGELOG.md`, not here.

## 1.0 — remaining

- **Payne–Hanek stress breadth (libm-oracle sweep).** Extend the
  `k ∈ {2⁴⁰, 2⁴⁵, 2⁵⁰}` corpus in
  `tests/test_transcendental_1ulp.cpp:720-724` to also include
  `2⁵⁰⁰` and `2⁹⁰⁰`. The MPFR-oracle sweep in
  `tests/test_coverage_mpfr.cpp` already covers those; the gap is a
  libm-oracle signal for the deep-reduction regime. Non-blocking for
  the 1.0 tag but should land in the 1.0.x window.

- **`sf64_sinh` overflow-boundary refinement.** The large-|x| branch at
  `src/sleef/sleef_inv_hyp_pow.cpp:550-557` flushes to ±inf at
  `|x| > 709.78`, but sinh doesn't actually overflow until
  `|x| > log(2·DBL_MAX) ≈ 710.4758`. For `|x| ∈ (709.78, 710.476]`
  sinh should return a finite value near ±DBL_MAX; the current
  implementation returns ±inf. Fix: evaluate `exp(|x| - log 2)` in
  that narrow band so the intermediate doesn't overflow before the
  ×0.5. Out of the shipped `[1e-4, 20]` sweep range, so not a tier
  violation for 1.0; landing this closes a visible edge on the
  positive-infinity side of the real line. Non-blocking for the 1.0
  tag.

## Post-1.0

### Numerical

- **`logk_dd` DD-Horner rewrite.** `sf64_pow` drifts above U35 in the
  "near-unit base × huge exponent" corner (`x ∈ [0.5, 2], |y| ≳ 200`)
  because `logk_dd` in `src/sleef/sleef_inv_hyp_pow.cpp` evaluates its
  tail polynomial on `x².hi` as a plain double, capping the log DD at
  ~2⁻⁵⁶ relative. Fix: evaluate the minimax polynomial in full DD
  arithmetic (DD Horner) and promote coefficient storage to DD pairs
  for the high-degree terms. Expected to move the worst-case `pow`
  from ~40 ULP to ≤4 ULP across the full double range. Also closes
  the `lgamma` `(0.5, 3)` zero-crossing report-only sweep.

### Feature surface (not yet implemented)

- **Non-RNE rounding modes.** `sf64_*_r(mode, …)` variants taking an
  explicit mode (`SF64_RNE`, `SF64_RTZ`, `SF64_RUP`, `SF64_RDN`,
  `SF64_RNA` — IEEE-754 §4.3). Enables hardware-emulation frontends
  (RISC-V `frm` CSR, ARM FPCR, x86 MXCSR), interval arithmetic, and
  freestanding runtimes. No ABI break — default stays RNE. Internal
  round-pack primitives in `src/internal.h` already abstract the
  rounding step; parametrize on mode. TestFloat emits vectors for all
  five modes; MPFR oracle: swap `MPFR_RNDN`. Target: v1.1.
- **IEEE exception flags + thread-local fenv.** Strict §7 conformance.
  Flag bits: `SF64_FE_{INVALID, DIVBYZERO, OVERFLOW, UNDERFLOW,
  INEXACT}` matching `<fenv.h>`. Entry points: `sf64_fe_{clear, test,
  raise, getall}`; optional `sf64_fe_state_t` opaque `_save` /
  `_restore` for freestanding / GPU contexts. Thread-local default
  (`thread_local` / `__thread`); build option
  `SOFT_FP64_FENV=tls|explicit|disabled`. Measured cost expected
  ≤10% on the hot arithmetic path, zero when disabled. TestFloat
  already emits expected-flag bits. Target: v1.1 / v1.2.
- **sNaN payload preservation.** Currently quiet-on-entry (sNaN →
  qNaN with canonical payload). Consumers needing §6.2 payload
  preservation require a `SOFT_FP64_SNAN_PROPAGATE` build option.
  Depends on the exception-flag work above (preservation raises
  `SF64_FE_INVALID`). TestFloat has dedicated sNaN vectors; wire them.
  Target: v1.2.
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
