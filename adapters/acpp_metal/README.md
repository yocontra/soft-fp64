# soft-fp64 → AdaptiveCpp Metal SSCP adapter

This adapter lets AdaptiveCpp's Metal libkernel use soft-fp64 as its
software fp64 backend on Apple Silicon (where the GPU lacks native fp64).
It is the integration path required by consumers like
[pg_accel](https://github.com/) to run fp64-gated Metal kernels instead
of trapping.

## What this produces

A staged flat directory of `.cpp` and `.h` files, written to

```
<soft-fp64 build>/adapters/acpp_metal/staged/
```

that AdaptiveCpp's Metal libkernel CMake can glob via its
`-DACPP_METAL_EXTERNAL_FP64_DIR` cache variable. The staged directory
contains:

- Every `sf64_*` TU (arithmetic, classify, compare, convert, rounding,
  sqrt/fma, and the four SLEEF TUs) — re-rooted into a flat layout.
- `acpp_metal_primitives.cpp` — one-line forwarders for every required
  `__acpp_sscp_soft_f64_*` primitive (arithmetic, min/max, convert,
  fcmp).
- `acpp_metal_math.cpp` — one-line forwarders for the optional
  `__acpp_sscp_*_f64` math surface (unary / binary / ternary / mixed).
- Public headers under `soft_fp64/` and the private `internal.h` /
  sleef headers, positioned so the unmodified core sources' relative
  `#include` directives resolve.

No conventional archive is emitted — the adapter is a staging target
only. `install-smoke` (the `sf64_*`-only ABI grep on `libsoft_fp64.a`)
is unaffected.

## How to integrate

### 1. Build soft-fp64 with the adapter enabled

```bash
cd /path/to/soft-fp64
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOFT_FP64_BUILD_ACPP_METAL_ADAPTER=ON
cmake --build build --target soft_fp64_acpp_metal_stage
```

The stage path is exported as the CMake cache variable
`SOFT_FP64_ACPP_METAL_STAGED_DIR`, defaulting to
`<build>/adapters/acpp_metal/staged/`.

### 2. Configure AdaptiveCpp to consume the staged dir

```bash
cd /path/to/AdaptiveCpp         # pinned: branch fork-safe-metal
                                # SHA    c86d474a3f1fb06705679efa527ea262b5a991cf
cmake -S . -B build \
    -DACPP_COMPILER_FEATURE_PROFILE=full -DWITH_METAL_BACKEND=ON \
    -DACPP_METAL_EXTERNAL_FP64_DIR=/absolute/path/to/soft-fp64/build/adapters/acpp_metal/staged \
    # …other AdaptiveCpp options…
cmake --build build --target libkernel-sscp-metal
```

AdaptiveCpp's Metal libkernel CMake globs `*.cpp` from that directory
(`file(GLOB … CONFIGURE_DEPENDS "${…}/*.cpp")`) and folds them into
`METAL_LIBKERNEL_BITCODE_SOURCES`. When `ACPP_METAL_EXTERNAL_FP64_DIR`
is set, the libkernel CMake also passes `-DACPP_HAS_EXTERNAL_SOFT_FP64`
to every bitcode TU, which elides the `__builtin_trap()` `_f64` block
in `AdaptiveCpp/src/libkernel/sscp/metal/math.cpp`. `llvm-link` then
sees a single definition per `__acpp_sscp_*_f64` symbol — the one
coming from this adapter. (Unlike ELF, `llvm-link` has no weak-symbol
preemption, so a preprocessor elide is the only way to avoid
"symbol multiply defined" errors at bitcode-link.)

## Pinned upstream

This adapter tracks the AdaptiveCpp `fork-safe-metal` branch.

| | |
|---|---|
| Branch | `fork-safe-metal` |
| SHA | `c86d474a3f1fb06705679efa527ea262b5a991cf` |
| Contract doc | `src/libkernel/sscp/metal/float64/README.md` |
| Glob rule | `src/libkernel/sscp/metal/CMakeLists.txt:28-62` |

Bumping the pin requires re-reading the contract doc at the new SHA and
updating this README + the two forwarder source headers.

## Symbol coverage

### Required primitives — all 23 forwarded

`__acpp_sscp_soft_f64_{add,sub,mul,div,rem,neg,fmin_precise,fmax_precise,
from_f32,to_f32,from_i32,from_i64,to_i32,to_i64,to_i16,to_i8,from_u32,
from_u64,to_u32,to_u64,to_u16,to_u8,fcmp}`.

### Optional math surface — forwarded 1:1

All SLEEF-backed transcendentals (trig / inverse trig / hyperbolic / exp
/ log / pow family / erf / gamma / cbrt, including `lgamma_r`), plus
rounding, `hypot`, `fmod` / `remainder`, `fract` / `frexp` / `modf` /
`ldexp` / `ilogb`, `pown` / `rootn`, and the classification predicates
(`isnan` / `isinf` / `isfinite` / `isnormal` / `signbit`). No trap
stubs retained from the pinned fork-safe-metal contract.

## Verification

After staging + building AdaptiveCpp with the external dir set, the
libkernel bitcode target must compile the staged sources:

```bash
cmake --build /path/to/acpp-build --target libkernel-metal-bitcode 2>&1 | grep -i fp64
```

Expected: lines citing the staged `.cpp` files being compiled, and a
clean bitcode-link (no unresolved `sf64_*` symbols).

Optional end-to-end — only valid when pg_accel is checked out alongside:

```bash
cd /path/to/pg_accel
just gpu-build && just gpu-test
```

With the adapter in place, pg_accel's fp64-gated kernels
(`reduce.cpp` / `spatial_predicates.cpp` / `h3_ops.cpp`) stop tripping
the `caps.has_fp64 == false` path and execute through soft-fp64.

## Layering contract

- This adapter does **not** add new public ABI. The only symbols it
  exports are the `__acpp_sscp_*` forwarders, and those live exclusively
  in the Metal libkernel bitcode (not in `libsoft_fp64.a`).
- Bug fixes in soft-fp64 propagate through the forwarders without
  adapter changes.
- Precision, rounding-mode, and IEEE-flag contracts match soft-fp64
  exactly.
- Subject to the same integrity rules as the core library (see
  `CLAUDE.md`): no host-FPU dependency, no tolerance widening, and no
  out-of-tree fork divergence from upstream AdaptiveCpp without
  re-pinning.
