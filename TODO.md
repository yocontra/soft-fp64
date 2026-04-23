# TODO

Single source of truth for open work. Items closed at 1.0 / 1.1 are
recorded in `CHANGELOG.md`, not here.

## Blocking v1.1.0

These block tagging. Each is a concrete, bounded change — not a
research task.

### `sf64_pow` cross-TU inlining regression

**What.** `sf64_pow` runs +43% slower in disabled-mode bench than the
1.0 baseline.

**Why it matters.** Fails the 1.0-baseline bench gate, which blocks the
tag. The regression is the 1.1 rounding-mode parameter payment at
every DD helper call site in `src/sleef/sleef_common.h`: the DD
primitives call `sf64_add` / `sub` / `mul` / `div` / `fma` as cross-TU
`extern "C"` entries, so `SF64_ALWAYS_INLINE` doesn't reach them
without library-wide LTO. The mode argument arrives as a runtime
value, the 5-way rounding switch survives in the DD primitive, and
`sf64_pow`'s ~160 arithmetic calls per invocation accumulate the
cost. Arithmetic / sqrt / fma called directly (same TU as their
public entry) already recovered — this is only the transcendentals
that compose through DD primitives.

**What's needed.** Expose hidden-visibility header-inlined RNE
specializations of the hot primitives in a new `src/internal_arith.h`
(`sf64_internal_{add,sub,mul,div,fma,sqrt}_rne`, no `mode` parameter —
RNE is hard-specialized). `sf64_add` etc. public entries become thin
wrappers. Every direct arithmetic call site across
`src/sleef/sleef_common.h`, `sleef_trig.cpp`, `sleef_exp_log.cpp`,
`sleef_inv_hyp_pow.cpp`, `sleef_stubs.cpp` swaps to the `_rne`
helpers; DD primitives thread a stack-local `sf64_internal_fe_acc&
fe` through the call tree and flush once at each SLEEF public
entry's return. Alternative rejected: library-wide LTO, which
breaks static-archive ABI and the `install-smoke` `nm -g` gate.
Verification: full unfiltered ctest green in both fenv modes; disasm
of `sf64_pow` shows no `bl _sf64_add` / `bl _sf64_mul` calls and no
5-way mode switch; `compare.py` pow delta within +10% vs 1.0.

### Bench-gate cheap-op carveout

**What.** `bench/compare.py` gains a `--cheap-op-absolute-budget=5.0`
flag that exempts any op with a <15 ns baseline from the percentage
gate provided the absolute delta is within the budget.

**Why it matters.** `sf64_to_i32` runs +166% in tls-mode vs
disabled-mode. Investigation confirmed this is the structural Apple
Silicon TLS-access cost (~4.45 ns per `__tlv_get_addr`), not a
fixable implementation issue — every `to_iN` / `to_uN` entry raises
from 4–6 sites so the accumulator is already optimal, and the
`initial-exec` TLS model is already applied. On a 2.68 ns baseline
the absolute delta is ~5 ns; the percentage is what looks alarming.
Cheap ops with sub-15 ns baselines pay the same ~5 ns TLS roundtrip
regardless, so the percentage gate is the wrong signal for them.

**What's needed.** `bench/compare.py` flag implementation;
`.github/workflows/ci.yml` bench-regression job updated to pass it;
`CHANGELOG.md` 1.1.0 "Performance — fenv-tls mode" subsection
documenting the ~5 ns per-op absolute cost on Apple Silicon plus the
cheap-op carveout. Cross-reference the `explicit`-state fenv ABI
(Post-1.1) as the path that removes the TLS floor.

### `sf64_floor` disabled-mode noise remeasure

**What.** `sf64_floor` shows +12% disabled-mode regression vs 1.0 at
`--min-time-ms=500`.

**Why it matters.** Fails the 1.0-baseline gate, but source is
byte-identical to 1.0 and every helper it calls was already
`SF64_ALWAYS_INLINE` pre-1.1. The delta is 0.78 ns on a 6.5 ns
baseline — consistent with measurement noise on shared-runner-class
hardware.

**What's needed.** Re-measure at `--min-time-ms=2000` ×3 samples. If
all three samples land within +10% of 1.0, note jitter dominance in
`CHANGELOG.md` and declare resolved. If the regression persists at
the longer run length, ship 1.1 with it documented and carry the
instruction-cache-alignment investigation to v1.1.1.

### TestFloat flag-column gate

**What.** Remove the "flags ignored" carve-out in
`tests/testfloat/run_testfloat.cpp`; parse the `fl2` column, compare
against `sf64_fe_getall()` after each vector, fail on mismatch.

**Why it matters.** The 7.16M TestFloat corpus ships with expected
flag bits; without the gate, the 1.1 fenv raise-site coverage is
untested at scale. The carve-out existed because raise sites didn't
exist; now they do.

**What's needed.** Runs under `SOFT_FP64_FENV=tls`. Any mismatch
blocks the tag — no row-skipping or suppression, no allowlist.

### Thread-safety test for the fenv accumulator

**What.** Add a two-thread test where each thread runs arithmetic
that raises different flag bits and asserts each thread observes only
its own accumulator.

**Why it matters.** The implementation uses thread-local storage; the
thread-independence claim needs a test.

**What's needed.** `tests/test_fenv_threads.cpp` new file, or extend
`tests/test_fenv.cpp`.

### ACPP Metal adapter picks up `SOFT_FP64_FENV_MODE`

**What.** `adapters/acpp_metal/CMakeLists.txt` and
`cmake/rewrite_sleef_include.cmake` propagate the top-level
`SOFT_FP64_FENV` build option into the staged source tree.

**Why it matters.** Today the adapter always compiles Metal bitcode
with `SOFT_FP64_FENV_MODE=0` regardless of core configuration, so
fenv flag raising silently no-ops on Metal — a configuration drift
bug, not a design choice.

**What's needed.** Plumbing only; no numerical change.

---

## Planned for v1.1

Non-blocking but targeted at the 1.1 tag.

### `logk_dd` DD-Horner rewrite

**What.** Replace the plain-double tail polynomial at the end of
`logk_dd` (`src/sleef/sleef_inv_hyp_pow.cpp:207-235`) with a DD
Horner chain against the full `x²` DD pair; promote the top 2–3
coefficients to DD-pair storage.

**Why it matters.** Today the tail caps the log DD at ~2⁻⁵⁶
relative, which is the root cause of `sf64_pow` worst-case ~40 ULP
in the near-unit-base × huge-exponent window. With the fix, pow
drops to ≤4 ULP across the double range and the gate in
`tests/mpfr/test_mpfr_diff.cpp` can demote U35 → U10. Also unblocks
`sf64_lgamma` `(0.5, 3)` graduation from experimental to the shipped
suite at GAMMA tier (see Post-1.1).

**What's needed.** Uses existing DD helpers (`ddmul_dd_dd_d`,
`ddadd2_dd_dd_d`) already in scope. Signature / call site in
`src/sleef/sleef_stubs.cpp` unchanged.

### `sf64_sinh` overflow boundary

**What.** `src/sleef/sleef_inv_hyp_pow.cpp:550-557` flushes to ±∞ at
`|x| > 709.78` (the `exp` overflow threshold); correct threshold for
sinh is `log(2·DBL_MAX) ≈ 710.4758`. Insert an intermediate branch
for `|x| ∈ (709.78, 710.4758]` that evaluates
`sf64_internal_exp_core(|x| − LN2) * 0.5` (the `* 0.5` keeps the
intermediate out of overflow).

**Why it matters.** ±inf return is wrong across ~0.7 units of input
width. Low-traffic regime, but a real bug.

**What's needed.** Extend the sinh sweep in
`tests/mpfr/test_mpfr_diff.cpp` with spot-check rows at
`{709.79, 710.0, 710.4, 710.48, ±}` gated at U35 = 8.

### Payne–Hanek deep-reduction coverage

**What.** Append `std::ldexp(1.0, 500)` and `std::ldexp(1.0, 900)`
to the `ks[]` array in `tests/test_transcendental_1ulp.cpp:720-724`.

**Why it matters.** Matches `tests/test_coverage_mpfr.cpp:248-254`,
which already exercises these k-values. Trivial coverage gap.

**What's needed.** Line-count-trivial test edit.

### Non-RNE rounding mode test parametrization

**What.** The `sf64_*_r(mode, …)` surface landed in 1.1 but the
tests still run RNE-only. Thread `mode` through the MPFR sweep
harness (`record()` / `sweep*_uniform()` in
`tests/mpfr/test_mpfr_diff.cpp`, mapping `sf64_rounding_mode` →
`mpfr_rnd_t`). Add a mode loop to `tests/testfloat/run_testfloat.cpp`
and regenerate `tests/testfloat/vectors/` per mode. Add explicit
per-mode bit-exact rows to `tests/test_arithmetic_exact.cpp`,
`test_sqrt_fma_exact.cpp`, `test_convert_widths.cpp` guarded by host
FPU fenv. Add a new `fuzz/fuzz_rounding_modes.cpp` target.

**Why it matters.** The non-RNE surface claims bit-exactness under
MPFR + TestFloat but no test asserts it across all five modes.

**What's needed.** Per-mode bench tracking in `CHANGELOG.md` once
the pow cross-TU fix above lands — that refactor may re-measure the
`_r` path cost as a side effect.

---

## Post-1.1

### Tighten `erf` / `erfc` / `tgamma` to U10 or U35

**What.** These currently gate at GAMMA (≤1024 ULP); SLEEF upstream
has tighter coefficient sets and extra range-reduction branches for
them.

**Why it matters.** Biggest accuracy gap a scientific consumer would
actually hit. Removes the loudest "is this really libm-grade?"
objection — medium-effort, high-credibility payoff.

**What's needed.** Port the higher-degree minimax polynomials and the
range-reduction branches onto the existing `sf64_*` primitives in
`src/sleef/`. Rewire the MPFR diff harness at the new bound; demote
the gate in `tests/mpfr/test_mpfr_diff.cpp`. Update the README
precision table.

### `sf64_lgamma` zero-crossings on `(0.5, 3)`

**What.** `lgamma(x)` vanishes at `x = 1` and `x = 2`; near those
zeros the result is O(1e-5) while the absolute error floor of any
log-of-Γ path is O(ulp(1)) ≈ 2.2e-16, so the ULP ratio blows past
GAMMA = 1024 regardless of ingredient precision. The 1.1 `logk_dd`
rewrite (if it lands) will confirm the issue is algorithmic, not
ingredient-precision.

**Why it matters.** Currently report-only in
`tests/experimental/experimental_precision.cpp` with a README caveat;
needs to become a hard gate to close the claim.

**What's needed.** Zero-centered Taylor expansion around `x = 1` and
`x = 2` — a branch inside `sf64_lgamma` that detects the vanishing
regime and returns `(x-1)·P₁(x)` / `(x-2)·P₂(x)` with coefficients
from the known series for `ψ(x)`. After the rewrite lands, graduate
the sweep to the shipped suite at GAMMA tier and drop the README
caveat.

### `SOFT_FP64_FENV=explicit` caller-state ABI for real

**What.** The `explicit` mode is reserved in the 1.1 CMake config
but compiles identically to `disabled` (zero-cost no-op raise sites,
surface present but stateless). Target shape: a parallel `sf64_fe_*`
ABI that takes an `sf64_fe_state_t*` directly instead of reading
thread-local storage.

**Why it matters.** TLS is fine on CPU but a non-starter on GPU /
SIMT targets — exactly the consumers this project exists for. Metal
and WebGPU kernels can't use `thread_local` at all, so there is no
path to fenv flag observability on the GPU without this. Also
removes the Apple Silicon TLS floor that forces the cheap-op
carveout in the 1.1 bench gate.

**What's needed.** Mechanical plumbing: thread a pointer parameter
through `sf64_internal_round_pack` and the existing `_r` variants;
add the new ABI as a parallel surface so 1.x consumers don't break;
new CMake configuration cell for the explicit mode; test-matrix row;
wire `tests/testfloat/run_testfloat.cpp`'s 7.16M-vector flag gate
under explicit mode (currently skipped).

### OpenCL C semantics mode

**What.** `sf64_*` is strict IEEE-754 today — denormals preserved,
correctly-rounded basic ops, u10/u35 transcendentals. OpenCL C's
`double` contract overlaps heavily but differs in three places:
denormal handling is implementation-defined for double (most drivers
FTZ to ±0); `native_{sin,cos,tan,exp,exp2,exp10,log,log2,log10,sqrt,
rsqrt,recip,divide,powr}` is specified as "implementation-defined
accuracy" (the fast/loose tier we don't ship); and some ops have
looser ULP bounds (a no-op for us where ours is tighter, but
`half_*` at 8192 ULP has no counterpart here).

**Why it matters.** Frontends like AdaptiveCpp's SSCP emitter (and
any SYCL / CUDA-on-Metal layer forwarding to `sf64_*`) spec-match
against the OpenCL reference. Consumers expecting FTZ'd subnormals
or `native_*` fast paths get different bits from strict IEEE.

**What's needed.** Additive `SOFT_FP64_OCL=off|on` build option
emitting a parallel `sf64_ocl_*` ABI; orthogonal
`SOFT_FP64_FTZ=off|on` that flushes `|x| < DBL_MIN` inputs/outputs to
signed zero at entry/exit. New `sf64_native_*` implementations — not
dials on the u10 SLEEF cores (the point is to skip the DD tails
entirely); propose u1024 as a non-binding ULP bound so consumers
have a number. TestFloat FTZ vectors wired into `run_testfloat.cpp`
under the new mode. New test tier in
`tests/mpfr/test_mpfr_diff.cpp` for the flushed edges. Adapter
forwarders `__acpp_sscp_native_*_f64` → `sf64_native_*`. README +
NOTICE updates; document conformance against OpenCL 3.0 full
profile. Not in scope: directed rounding (already covered by the
`sf64_*_r` surface), `half_*` / single-precision variants (different
library), fenv exception flags (OpenCL has no `<fenv.h>` equivalent;
`disabled` is the OpenCL-matching fenv setting).

### Portable 64×64→128 multiply in `sf64_fma`

**What.** `src/sqrt_fma.cpp` assumes the compiler provides
`__uint128_t`. Replace with a portable 64×64→128 via four 32-bit
partials, gated on `#ifndef __SIZEOF_INT128__`.

**Why it matters.** The hard dependency blocks 32-bit MCUs, MSVC, and
some WASM toolchains — exactly the target matrix soft-fp64 is
supposed to widen. ~40 lines, the SoftFloat reference has the exact
pattern.

**What's needed.** Implementation plus a CI cell that builds under a
toolchain without `__uint128_t` (cheapest option: MSVC on Windows,
or a wasm32 target). Numerical claim stays bit-exact — the two paths
compute identical bits by construction; a ctest run covers both.

### sNaN payload preservation

**What.** Today sNaN → qNaN on entry with a canonical payload; 1.1
raises `SF64_FE_INVALID` on that entry when fenv is enabled. IEEE
754 §6.2 allows preserving the signalling-payload bits through the
quiet-bit force.

**Why it matters.** The last TestFloat vectors currently skipped are
the documented sNaN carve-out in `tests/testfloat/run_testfloat.cpp`.
Un-skipping them lets the README claim 100% of 7.16M vectors pass —
a real credibility line. The preservation itself is completeness
work for most consumers (§6.2 is implementation-defined), but the
change is ~5 lines in `internal.h:144-158` (preserve bits 50:0, set
bit 51).

**What's needed.** New `SOFT_FP64_SNAN_PROPAGATE` build option
gating the payload-preserving path; un-skip the TestFloat sNaN rows
under it; add a handful of bit-exact spot-checks for payload
survival across each arithmetic entry.

### `soft-fp128` sibling

**What.** Same design playbook (Mesa arithmetic port + SLEEF
transcendentals + TestFloat + MPFR oracle) extended to 113-bit
significand. Storage wrapper; full conversion matrix (`f64 ↔ f128`,
`i128 ↔ f128`); u10 transcendentals vs MPFR 300-bit.

**Why it matters.** Same consumers (GPU / MCU targets that lack
hardware fp128) need it. Decoupled from fp64 on the release
timeline.

**What's needed.** Ships as a separate package once fp64 stabilizes.

### ACPP Metal adapter follow-ups

- **`__acpp_sscp_lgamma_r_f64` forwarding.** The adapter currently
  leaves the trap-stub in place. Blocks on adding a `sf64_lgamma_r`
  core entry point — computing the Γ sign from `sf64_tgamma(x)`
  round-trips through an overflow-prone path for `|x| > 170`.
- **`_r`-variant forwarders.** Once the non-RNE `sf64_*_r` surface
  from 1.1 stabilizes on the Metal target, the adapter gains
  one-line `__acpp_sscp_soft_f64_*_r` forwarders. Pure forwarding,
  zero ULP to add.
- **`sf64_fe_*` surface on Metal.** Optional re-export once
  `explicit`-state fenv lands; `SOFT_FP64_FENV=disabled` stays the
  default on GPU targets.

---

## Not on the roadmap

- **fp16 / bfloat16.** Different design space — table-driven
  implementations win at that precision. No fp64 synergy.
- **Decimal FP (IEEE-754 §3.5).** Different rounding philosophy; see
  `libdfp` or Intel's DFP library.
- **Complex-number math** (`csin`, `clog`, etc.). Pure wrapper work
  on top of real scalars; belongs in a consumer.
- **Guest-FPU emulation glue.** `fenv` compat, `softfp` ABI lowering,
  calling-convention shims belong in the frontend (compiler runtime,
  emulator), not here. `sf64_*` is the contract; how you call it is
  your problem.
- **Runtime CPU dispatch for fast paths.** The point is
  architecture-independent bit-exactness. If you have fast native
  fp64, you don't need this library.
