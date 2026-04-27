# soft-fp

Portable, bit-exact IEEE-754 soft-float libraries in pure integer code —
for any target without hardware floating-point of the relevant width.

The repository hosts a suite of integer-only IEEE-754 implementations that
share build infrastructure, oracle setup (Berkeley TestFloat 3e + MPFR),
benchmark harness, and ABI conventions. Currently shipping:

- **`soft-fp64`** — the full binary64 surface (arithmetic, compare, full
  width-matrix conversions, sqrt, fma, rounding, classification,
  transcendentals). Static archive `libsoft_fp64.a`, public `sf64_*`
  C ABI, `find_package(soft_fp64)`. The rest of this README is about
  this library — see precision tables, install paths, and integration
  notes below.

Planned (not yet implemented; see `TODO.md`):

- **`soft-fp128`** — same playbook extended to 113-bit binary128. Sibling
  static archive `libsoft_fp128.a`, public `sf128_*` C ABI,
  `find_package(soft_fp128)`. Decoupled release cadence from `soft-fp64`.

The README, CMakeLists, `find_package` name, `sf64_*` ABI prefix, and
`include/soft_fp64/` header path are stable surface for the fp64 library
and won't shift when fp128 lands alongside.

## What soft-fp64 does

Lots of modern hardware ships without fp64 units: Apple GPUs, most mobile
and tile-based GPUs, WebGPU / OpenGL ES class devices, many embedded DSPs,
FPGAs, and custom accelerators. When scientific, geospatial, cryptographic,
or scientific-computing code hits one of these targets, the compiler either
traps, silently truncates to `float`, or refuses to lower the code at all.

`soft-fp64` is a clean, header-plus-object C++17 implementation of the full
IEEE-754 binary64 surface, built entirely on 32/64-bit integer bit
operations. There is no hidden dependency on the host FPU. Any frontend
that can emit a call to an `extern "C"` symbol can get correct `double`
behavior on a device without a hardware fp64 unit.

## Where this is useful

Targets and toolchains that currently have no good story for `double`
when the hardware lacks an fp64 unit. Integration work for each is
tracked in `TODO.md`.

- **SYCL on Apple GPUs.** AdaptiveCpp's Metal SSCP emitter compiles
  fp64 but traps at runtime. Linking soft-fp64 into libkernel bitcode
  lets `double` kernels run (much slower than fp32, but correct).
- **`torch.float64` on MPS.** PyTorch's MPS backend errors on fp64.
  A dispatch path backed by this library would produce a working
  (slow) tensor.
- **WebGPU.** WGSL has no `f64`. Cross-compiling `sf64_*` to a
  linkable WGSL/SPIR-V module gives Tint / Naga something to lower
  fp64 ops against.
- **Mesa drivers without fp64.** Lima, Panfrost, older Freedreno,
  LLVMpipe-on-WASM reject fp64 OpenCL kernels. Mesa's in-tree
  softfloat covers arithmetic but not transcendentals; this library
  covers the full surface.
- **compiler-rt / libgcc softfloat comparison.** The `__adddf3` /
  `__muldf3` / `__divdf3` builtins used on RISC-V without D, ARMv6-M,
  and WASM-softfp have known subnormal / NaN-payload corner cases.
  soft-fp64 is TestFloat- and MPFR-verified bit-for-bit on the same
  inputs, so it's usable as a differential oracle.
- **Independent reference.** Every op is gated in CI against Berkeley
  TestFloat 3e plus a 200-bit MPFR oracle, with the precision table
  published, so the library can serve as an oracle for other
  softfloat implementations.

## Who this is for

- **Compiler / runtime authors** retargeting language front-ends to hardware
  without fp64 (AdaptiveCpp's MSL emitter is the first consumer; anyone
  doing the same for Vulkan / WebGPU / a custom ISA can link against the
  same ABI).
- **Shader / kernel authors** who need correct `double` in a single hot path
  and don't want to carry a full libm port.
- **Scientific / geospatial / financial code** being cross-compiled to an
  fp64-less target (mobile, embedded, browser) that previously required
  giving up precision or the target.
- **Test oracles** — the implementation is also a clean, self-contained
  reference for anyone validating their own soft-fp64 code.

## Install

Three supported integration paths. Pick whichever matches your build.

### 1. CMake `find_package`

After `cmake --install`, consumers use:

```cmake
find_package(soft_fp64 REQUIRED)
target_link_libraries(my_app PRIVATE soft_fp64::soft_fp64)
```

### 2. Git submodule + `add_subdirectory`

No install step; vendor the source tree:

```cmake
add_subdirectory(extern/soft-fp64)
target_link_libraries(my_app PRIVATE soft_fp64::soft_fp64)
```

### 3. pkg-config

For non-CMake build systems:

```bash
cc $(pkg-config --cflags soft_fp64) my_app.c $(pkg-config --libs soft_fp64)
```

## Usage

Minimal working example:

```c
#include "soft_fp64/soft_f64.h"

int main(void) {
    double r = sf64_add(1.0, 2.0);   // 3.0
    double s = sf64_sqrt(2.0);       // 1.4142135623730951
    double t = sf64_sin(sf64_log(r));
    (void)s; (void)t;
    return 0;
}
```

Every symbol is a pure `extern "C"` function with no global state. See
`include/soft_fp64/soft_f64.h` for the full API (Doxygen-annotated).

### Integrating under a different symbol name

The library exports stable, vendor-neutral `sf64_*` symbols. If your
frontend's code generator emits calls under a different name (for example,
AdaptiveCpp's MSL emitter emits `__acpp_sscp_soft_f64_*`), don't rebuild
this library — add a thin shim in your frontend that forwards to the
`sf64_*` symbols:

```cpp
extern "C" double __acpp_sscp_soft_f64_add(double a, double b) {
    return sf64_add(a, b);
}
// …one line per symbol.
```

That keeps vendor-specific naming where it belongs (in the vendor's
frontend) and lets the library stay generic.

### AdaptiveCpp Metal backend

The AdaptiveCpp Metal SSCP path — used by downstream GPU consumers on
Apple Silicon, where the GPU lacks native fp64 — used to ship as a
bundled adapter under `adapters/acpp_metal/`. As of v1.2.0 that glue
lives in AdaptiveCpp itself (the `__acpp_sscp_*_f64` forwarders are
AdaptiveCpp ABI symbols, not soft-fp64's public surface). See
[`yocontra/AdaptiveCpp`](https://github.com/yocontra/AdaptiveCpp) on
the `fork-safe-metal` branch for the integration; soft-fp64 is
consumed there as a generic source dependency via the public `sf64_*`
ABI plus the `src/` and `src/sleef/` source trees (exposed by
`find_package(soft_fp64)` as `soft_fp64_SOURCE_DIR` /
`soft_fp64_SLEEF_SOURCE_DIR`).

## Precision guarantees

Every bound below is **CI-gated** against a 200-bit MPFR oracle by the
sweep named in the last column. The tier for each row is the tightest
bound ctest enforces; measured worst-case may be inside it, but the
guarantee is the gate.

Tiers: `BIT_EXACT` (0 ULP), `U10 = 4 ULP`, `U35 = 8 ULP`, `GAMMA = 1024 ULP`.

| Category | Examples | Tier | Range / oracle |
|---|---|---|---|
| Arithmetic | `add`, `sub`, `mul`, `div`, `rem`, `fma`, `sqrt`, `remainder`, `fmod` | **BIT_EXACT** | host FPU + TestFloat vectors + `test_arithmetic_exact.cpp`, `test_sqrt_fma_exact.cpp`; `fmod` / `remainder` also 0-ULP vs MPFR in `test_mpfr_diff.cpp` |
| Conversion | `i{8,16,32,64} ↔ f64`, `u{8,16,32,64} ↔ f64`, `f32 ↔ f64` | **BIT_EXACT** | `test_convert_widths.cpp`; exhaustive 2³² `f32 → f64 → f32` round-trip |
| Comparison / classification | `fcmp` (all 16 IEEE-754 predicates), `isnan`, `isinf`, `isfinite`, `isnormal`, `signbit`, `fabs`, `copysign` | **BIT_EXACT** | TestFloat vectors + `test_compare_all_predicates.cpp` |
| Rounding | `floor`, `ceil`, `trunc`, `rint`, `round`, `fract`, `modf`, `ldexp`, `frexp`, `ilogb`, `logb` | **BIT_EXACT** | `test_rounding_edges.cpp` |
| Transcendentals (u10) | `sin`, `cos`, `asin`, `acos`, `atan`, `atan2`, `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`, `cbrt`, `cosh`, `acosh`, `atanh`, `sinpi`, `cospi`, `asinpi`, `acospi`, `atanpi`, `atan2pi` | **U10** ≤ 4 ULP | `test_mpfr_diff.cpp` |
| Transcendentals (u35) | `tan`, `tanpi`, `sinh`, `tanh`, `asinh` | **U35** ≤ 8 ULP | `test_mpfr_diff.cpp` |
| `pow` / `powr` / `pown` / `rootn` | `pow(x, y)`, `powr(x, y)`, `pown(x, n)`, `rootn(x, n)` | **U35** ≤ 8 ULP, bounded region | three overlapping windows, see note |
| `erf` | `erf(x)` | **U10** ≤ 4 ULP | `[-5, 5]`, `test_mpfr_diff.cpp` (SLEEF u1 port; measured worst 1 ULP) |
| `erfc` | `erfc(x)` | **U10** ≤ 4 ULP | `[-5, 27]` (full active range incl. deep tail; measured worst 1 ULP) |
| `tgamma` | `tgamma(x)` | **U10** ≤ 4 ULP | `[0.5, 170]` (through the overflow boundary; measured worst 1 ULP) |
| `lgamma`, `lgamma_r` | `lgamma(x)` | **U10** ≤ 4 ULP zero-free tail; **GAMMA** at zero-crossings | `[3, 1e4]` U10 Lanczos tail + `(0.5, 3)` GAMMA (ULP ratio unbounded as `\|lgamma\| → 0` near `x=1, 2`; absolute error stays inside U10) |

### `sf64_pow` — bounded-region U35

`sf64_pow` is composed as `pow(x, y) = exp(y · log(x))` with a double-double
`y · log(x)` intermediate collapsed to a single `double` before the final
`sf64_exp`. CI gates three overlapping sweeps at **U35 ≤ 8 ULP**:

- main: `x ∈ [1e-6, 1e6]`, `y ∈ [-50, 50]`
- x-wide: `x ∈ [1e-100, 1e100]`, `y ∈ [-5, 5]`
- y-wide: `x ∈ [1e-6, 1e3]`, `y ∈ [-100, 100]`

Across the full double range, 1.1's `logk_dd` DD-Horner rewrite brings
measured worst-case `sf64_pow` inside U10 (≤4 ULP); the shipped U35
tier remains the gated ceiling.

### Report-only (NOT CI-gated)

There are currently no parked precision regimes. The previous
`sf64_lgamma` zero-crossing caveat on `(0.5, 3)` was closed by
zero-centered Taylor branches in `src/sleef/sleef_stubs.cpp` (DD-Horner
on the `log Γ(1+z)` and `log Γ(2+z)` series, DLMF §5.7.3, with windows
`|x-1| ≤ 0.25` and `|x-2| ≤ 0.5`); the sweep graduated to
`tests/mpfr/test_mpfr_diff.cpp` at `GAMMA` tier.

"Bit-exact" means every output matches the host FPU's round-to-nearest-even
result in every bit, for every tested input — including every signed-zero,
subnormal, NaN, and infinity edge case.

## Performance

Self-contained microbench from `bench/bench_soft_fp64.cpp` against
`bench/baseline.json`.

| op | ns/op | Mops/sec |
|----|-------|----------|
| add | 10.9 | 91.8 |
| sub | 11.1 | 89.8 |
| mul | 5.2 | 192.7 |
| div | 16.9 | 59.3 |
| fma | 17.9 | 55.8 |
| sqrt | 123.6 | 8.1 |
| rsqrt | 142.3 | 7.0 |
| from_i32 | 1.2 | 842.8 |
| to_i32 | 4.9 | 204.7 |
| floor | 6.5 | 153.2 |
| ceil | 6.4 | 156.7 |
| trunc | 1.2 | 843.7 |
| rint | 5.6 | 177.4 |
| fcmp_oeq | 1.9 | 528.6 |
| sin | 446.5 | 2.2 |
| cos | 524.7 | 1.9 |
| tan | 626.5 | 1.6 |
| asin | 374.2 | 2.7 |
| acos | 392.6 | 2.5 |
| atan | 644.5 | 1.6 |
| atan2 | 709.5 | 1.4 |
| sinh | 494.6 | 2.0 |
| cosh | 521.2 | 1.9 |
| tanh | 310.1 | 3.2 |
| asinh | 758.0 | 1.3 |
| acosh | 706.8 | 1.4 |
| atanh | 570.3 | 1.8 |
| exp | 256.5 | 3.9 |
| exp2 | 274.6 | 3.6 |
| expm1 | 758.0 | 1.3 |
| log | 472.1 | 2.1 |
| log2 | 465.1 | 2.1 |
| log1p | 333.5 | 3.0 |
| pow | 1324.2 | 0.8 |
| cbrt | 1109.0 | 0.9 |

Hardware: Apple M-class, macOS, Release build, `-ffp-contract=off`.

### Comparative bench

A separate harness `bench/bench_compare.cpp` measures `sf64_*` against
Berkeley SoftFloat 3e (core IEEE ops) and ckormanyos/soft_double (core
ops + transcendentals). Comparison libraries are vendored on demand via
`bench/fetch_external.sh`; the CI job `comparative-bench` runs on
manual workflow dispatch and uploads `compare.json` as an artifact.
See `bench/README.md` for the full comparison table and scope.
Informational — not a regression gate.

## Testing

The oracle stack runs at four depths, each stricter than the last:

1. **Host FPU + random sweeps** — every op is cross-checked bit-for-bit
   against the host's native `double` across edge corpora plus 10⁴ random
   inputs. See `tests/test_arithmetic_exact.cpp`,
   `tests/test_convert_widths.cpp`, `tests/test_compare_all_predicates.cpp`,
   `tests/test_rounding_edges.cpp`, `tests/test_sqrt_fma_exact.cpp`.
2. **Berkeley TestFloat vectors** — the canonical IEEE-754 conformance
   corpus, replayed through `sf64_*`. See `tests/testfloat/`.
3. **MPFR 200-bit differential** — every transcendental is compared against
   a 200-bit MPFR reference for ULP measurement. See `tests/mpfr/` and
   `tests/test_transcendental_1ulp.cpp`.
4. **libFuzzer + exhaustive round-trip** — nightly CI fuzzes each op and
   exhaustively round-trips every `float ↔ double` value. See `fuzz/`.

## Rounding modes

The default `sf64_*` entry points round to nearest, ties-to-even (RNE).
Every round-affected op also has a `_r`-suffixed variant that takes an
explicit mode from `include/soft_fp64/rounding_mode.h` (matching
IEEE-754 §4.3):

```c
#include "soft_fp64/soft_f64.h"

double a = sf64_add_r(SF64_RTZ, 1.0, std::ldexp(1.0, -53));  // round toward zero
int32_t i = sf64_to_i32_r(SF64_RUP, 1.5);                    // 2
double r = sf64_rint_r(SF64_RDN, -0.5);                       // -1.0
```

Mode enum values: `SF64_RNE`, `SF64_RTZ`, `SF64_RUP`, `SF64_RDN`,
`SF64_RNA`. Every `_r` entry is bit-exact against MPFR 200-bit and
Berkeley TestFloat 3e in every mode. Ops whose result does not depend
on the rounding attribute (`neg`, `fabs`, `copysign`, compares,
`ldexp`, `frexp`, classify, `fmod`, `remainder`) have no `_r` form
by design.

## IEEE exception flags

Thread-local sticky flags for `INVALID`, `DIVBYZERO`, `OVERFLOW`,
`UNDERFLOW`, and `INEXACT`, exposed via:

```c
sf64_fe_clear(0x1Fu);                    // clear all flags
double r = sf64_div(1.0, 0.0);            // raises DIVBYZERO
unsigned f = sf64_fe_getall();            // f & SF64_FE_DIVBYZERO != 0
if (sf64_fe_test(SF64_FE_INVALID)) { …}

sf64_fe_state_t saved;
sf64_fe_save(&saved);                     // snapshot
// … scratch work that may raise flags …
sf64_fe_restore(&saved);                  // roll back
```

Flag storage is per-thread (`thread_local`); bit positions match
`<fenv.h>` conventions so consumers can bridge to glibc fenv without a
lookup table. Build option `SOFT_FP64_FENV`:

- `tls` (default on hosted builds) — thread-local accumulator.
- `disabled` — every raise-site compiles out and every `sf64_fe_*` entry
  becomes a no-op; zero runtime cost on the hot path.
- `explicit` — reserved for a caller-provided state ABI; currently
  compiles as `disabled`.

```bash
cmake -S . -B build -DSOFT_FP64_FENV=disabled     # no flag overhead
cmake -S . -B build                                # default: tls
```

Full-corpus flag parity is gated by `tests/testfloat/run_testfloat.cpp`
against Berkeley SoftFloat's `fl2` column (7.16M vectors,
`-tininessbefore`, `-exact`). sNaN-input rows are skipped pending
payload preservation in 1.2.

## Non-goals

- No sNaN payload preservation (sNaN inputs are quieted on entry;
  `INVALID` is still raised when fenv is enabled).
- No complex-number math.
- No fp128 / fp16 sibling in this library.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -V
```

Tests cross-check every soft-fp64 op against the oracle stack described
above.

## License + attribution

MIT — see [`LICENSE`](LICENSE). Third-party code incorporated under their
respective licenses; full attribution in [`NOTICE`](NOTICE).

- Skeleton and public API shape derived from
  **[philipturner/metal-float64](https://github.com/philipturner/metal-float64)**
  (MIT, 2023).
- Arithmetic (add/sub/mul/div, sqrt, fma, compare, convert) ported from
  **Mesa** `src/compiler/glsl/float64.glsl` (BSD-3-Clause) and
  `src/compiler/nir/nir_lower_double_ops.c` (MIT).
- Transcendentals (sin/cos/tan/asin/acos/atan/exp/log/pow/erf/tgamma/…)
  ported from **SLEEF 3.6** `sleefinline_purec_scalar.h` + scalar sources
  (Boost-1.0).
