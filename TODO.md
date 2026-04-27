# TODO

Single source of truth for open work. Items closed at 1.0 / 1.1 / 1.2
are recorded in `CHANGELOG.md`, not here.

## Post-1.2

### sNaN payload preservation

**What.** Today sNaN → qNaN on entry with a canonical payload (sign +
quieted high-bit). 1.2 raises `SF64_FE_INVALID` on every sNaN-input
arithmetic / sqrt / fma / convert / fmod / remainder operation, but the
payload bits 50:0 of the original sNaN are still discarded by
`propagate_nan` in `src/internal.h`. IEEE 754 §6.2 lets implementations
preserve the signalling-payload bits through the quiet-bit force, so a
strict consumer can chain the original payload across an arithmetic
chain.

**Why it matters.** A handful of TestFloat vectors test payload
preservation specifically (separate from the sNaN-input INVALID raise,
which is now gated). Some scientific consumers encode debug
information in the payload and expect it to survive arithmetic.

**What's needed.** New `SOFT_FP64_SNAN_PROPAGATE` build option (default
OFF — preserves current behaviour) gating the payload-preserving path
in `propagate_nan` and the four sites that quiet a NaN without going
through it (`sf64_internal_sqrt_rne`, `sf64_internal_fma_rne`,
`to_f32_impl`, `from_f32_impl`). `sf64_sub` / `sf64_sub_r`'s
b-operand sign XOR-flip needs a sNaN-aware fast-path so the propagated
sign survives. `sf64_fmod` / `sf64_remainder` need a payload-bearing
`propagate_nan_xy(x, y)` helper. New
`tests/test_snan_payload.cpp` for bit-exact spot-checks across
`add` / `sub` / `mul` / `div` / `sqrt` / `fma` / `from_f32` under
both option states; un-skip any remaining payload-specific TestFloat
rows under the option.

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
profile. Adding `_ex` variants for `sf64_remainder` / `sf64_fmod`
in `src/sleef/` lives under this work too — the SLEEF surface
deliberately stays out of the 1.2 explicit-state lift to keep the
diff bounded.

Not in scope: directed rounding (already covered by the `sf64_*_r`
surface), `half_*` / single-precision variants (different library),
fenv exception flags (OpenCL has no `<fenv.h>` equivalent;
`disabled` is the OpenCL-matching fenv setting).

### `soft-fp128` sibling library

**What.** Same design playbook (Mesa arithmetic port + SLEEF / DD-style
transcendentals + TestFloat + MPFR oracle) extended to 113-bit
significand. Static archive `libsoft_fp128.a` alongside `libsoft_fp64.a`
in this same `yocontra/soft-fp` repo. Public C ABI prefix `sf128_*`
mirroring `sf64_*` (arithmetic, compare, full conversion matrix
including `f64 ↔ f128` and `i128 ↔ f128`, sqrt, fma, rounding,
classify, transcendentals). u10 transcendentals gated against MPFR
300-bit; bit-exact arithmetic gated against TestFloat fp128 vectors.

**Why it matters.** Same consumers (GPU / MCU / wasm targets without
hardware fp128) need it. Decoupled from fp64 on the release timeline,
but co-located so the build infrastructure, oracle setup, benchmark
harness, and ABI conventions are shared rather than re-invented in a
sibling repo.

**What's needed.** When work starts, introduce a top-level layout
split — likely `fp64/` and `fp128/` subtrees with a top-level
`CMakeLists.txt` orchestrating both — and grow the existing
`adapters/` and `tests/testfloat/` / `tests/mpfr/` infrastructure to
exercise both precisions. Until then, the repo stays flat with `src/`
+ `include/soft_fp64/` at root; restructuring before fp128 has files
to put somewhere is premature. SLEEF doesn't ship fp128 polynomials,
so the transcendental story will draw from a different reference (Sun
fdlibm-q, Boost.Multiprecision-derived coefficient sets, or DD/QD
arithmetic on top of `sf64_*` — TBD when the work starts).

---

## Integrations to land upstream

This repo owns the `sf64_*` core ABI. Adapter code that wires
`sf64_*` into a consumer project (CMake glue, builtin forwarders,
bitcode-archive packaging, dispatch-table edits) belongs in that
consumer's tree, not here. The previous `adapters/acpp_metal/`
prototype was removed in v1.2.0; the AdaptiveCpp glue lives in
`yocontra/AdaptiveCpp` on the `fork-safe-metal` branch, and
soft-fp64 is consumed there as a generic source dependency.

Each item below is a concrete wedge into a downstream project —
what they're stuck on, where `sf64_*` fits, the PR shape, and the
real hurdles.

What this library adds over Berkeley SoftFloat (the established
softfloat oracle that Mesa `float64.glsl`, compiler-rt, and libgcc
`soft-fp` all descend from): (a) **transcendentals** — `sin`,
`cos`, `log`, `exp`, `pow`, `erf`, etc., ported from SLEEF and
gated against MPFR-200-bit; SoftFloat has no transcendentals —
and (b) a packaged library with a stable C ABI. For arithmetic /
convert / compare / classify, Berkeley SoftFloat is already the
canonical oracle and this library doesn't clear a higher bar;
integration pitches that depend on "better arithmetic oracle" are
not viable and aren't listed.

### AdaptiveCpp (SYCL implementation)

**The gap.** AdaptiveCpp's SSCP emitter hard-rejects `double` at
IR-to-MSL translation
(`src/compiler/llvm-to-backend/metal/Emitter.cpp:1432`, error:
`Double type is not supported on Metal GPU`). The runtime
reports `device_support_aspect::fp64 → false`
(`src/runtime/metal/metal_hardware_manager.cpp:348`). The Metal
libkernel file `src/libkernel/sscp/metal/math.cpp` ships 87
`_f32` entries and zero `_f64` entries — an empty slot, not
missing bodies. Upstream is receptive: issue #864 lists "float64
emulation" under optional features, PR #1961 ("Metal Backend",
merged Feb 2026) scoped fp64 out as a follow-up, and PR #1980
(merged Feb 2026) added the `device_support_aspect::fp64`
plumbing. No in-flight fp64 PR.

**The fit.** Two parts:

1. **libkernel bodies.** One-line forwarders from each
   `__acpp_sscp_*_f64` builtin (~70 entries declared in
   `include/hipSYCL/sycl/libkernel/sscp/builtins/math.hpp` et
   al.) to the corresponding `sf64_*` symbol. The forwarder set
   prototyped historically under `adapters/acpp_metal/` now lives
   in AdaptiveCpp's tree (`yocontra/AdaptiveCpp`,
   `fork-safe-metal` branch).
2. **IR-level fp64 legalization pass.** MSL has no `double` at
   the language level — `Emitter.cpp` rejects `llvm::DoubleTy`
   as a source-language error, not a missing-builtin link
   error. Before the emitter runs, every `DoubleTy` value
   (loads, stores, GEPs, function args, phis) must be rewritten
   to `i64` (or `uint2`) and every fp64 IR op (`fadd`, `fmul`,
   `fcmp`, `fptosi`, `sitofp`, `fpext`, `fptrunc`) outlined into
   a call to the `sf64_*` ABI. This is a whole LLVM pass
   (similar to LLVM's `SoftFloat` lowering but retargeted to
   `sf64_*`). AdaptiveCpp does not do this rewrite today, and
   it is the actual scope of the integration — the forwarders
   alone are not enough.

**PR shape.** (1) Pre-Emitter LLVM pass lowering all `DoubleTy`
values + fp64 IR ops to `i64` + `sf64_*` calls. Register
forwarders in the Metal `remapped_llvm_math_builtins` table
(`src/compiler/llvm-to-backend/metal/LLVMToMetal.cpp:60-112`).
(2) Forwarder bodies live in AdaptiveCpp's SSCP extension tree
(roughly `src/runtime/sscp/extensions/soft_fp64/`); soft-fp64 is
consumed as a source dependency via `find_package(soft_fp64)` and
`soft_fp64_SOURCE_DIR` / `soft_fp64_SLEEF_SOURCE_DIR`. (3) Opt-in
CMake option on the AdaptiveCpp side. The previous in-repo
`adapters/acpp_metal/` prototype was removed in soft-fp64 v1.2.0.

**Hurdles.** License (MIT → AdaptiveCpp BSD-3, compatible).
Release-cadence coupling — consumers pinned to an older
soft-fp64 need a `soft_fp64_VERSION_MIN_REQUIRED` compile guard.
The IR-legalization pass is multi-week work, not a weekend port.

### PyTorch MPS backend

**The gap — and what this library does NOT fix.** PyTorch's MPS
backend rejects `torch.float64` at tensor-construction time via a
~10-line check in `aten/src/ATen/mps/EmptyTensor.cpp` (and a mirror
in `aten/src/ATen/native/mps/OperationUtils.mm`). The actual user
pain — library code (sklearn, scipy, HF Diffusers, PyTorch's own
`WeightedRandomSampler`, GitHub issues #125844 / #148670 / the
#77764 op tracker) incidentally allocating `torch.double` and
crashing — is **not a missing-kernels problem**. It's a hard reject
before dispatch ever runs. Almost nobody is asking to *train* in
fp64 on MPS; they're asking not to crash when an upstream library
defaults to it.

The 99% fix is to delete the rejects and route fp64 ops to
PyTorch's existing `cpu_fallback()` (`aten/src/ATen/mps/
MPSFallback.mm`), gated on `PYTORCH_ENABLE_MPS_FP64=1`. On Apple
Silicon's unified memory, MPS↔CPU "copies" are near-metadata-only,
so CPU fp64 runs at ~1–3× hardware-fp32 speed. Device-side
soft-fp64 in Metal runs ~300× slower than fp32. **For the actual
user population, CPU fallback is both simpler and faster than this
library.** Soft-fp64 is the wrong tool for that PR — it doesn't
belong in it.

**Where soft-fp64 does fit.** The narrow case where a hot Metal
kernel wants a handful of fp64 ops in its inner loop and cannot
afford the CPU-roundtrip scheduling cost — e.g. a fused reduction
inside a larger Metal graph where the rest of the work stays on
GPU. That's a real but small user base, well downstream of the
CPU-fallback PR.

**PR shape (Tier 2, soft-fp64 on device).**  Only pursue after the
CPU-fallback PR lands and a concrete consumer asks for it. Rewrite
the raw-Metal kernels under `aten/src/ATen/native/mps/kernels/` to
carry fp64 as `uint2` (MSL has no `double` syntax — you cannot
just link soft-fp64 bitcode into an MSL kernel that declares a
`double` variable; every affected kernel needs its types rewritten)
and call `sf64_*`. Opt-in `PYTORCH_MPS_FP64_DEVICE=1`. Consumes the
same Metal bitcode archive as AdaptiveCpp.

**Hurdles (Tier 2).** Of the ~50 op files under
`aten/src/ATen/native/mps/`, 34 go through MPSGraph (Apple's
closed-source graph compiler) and 16 use raw Metal kernels.
MPSGraph has no fp64 and cannot be patched from the outside — so
even Tier 2 only covers the raw-Metal subset. Ops reachable only
via MPSGraph permanently route through CPU fallback regardless.
This caps the device-side soft-fp64 surface to roughly 1/3 of MPS
op coverage. Factor that into any pitch.

### libclc fp64 transcendentals (for Mesa rusticl softfp64)

**The gap.** Mesa's runtime fp64-in-shader path
(`src/compiler/glsl/float64.glsl` → `nir_lower_doubles` in
`src/compiler/nir/nir_lower_double_ops.c`) already covers
bit-correct fp64 arithmetic on shader-side — it's SoftFloat-
derived and mature (MR !4142, MR !38088 merged). But Mesa has no
softfloat transcendentals for OpenCL fp64 kernels; those are
expected to come from libclc, and the libclc fp64-on-softfp path
is incomplete. Rusticl's softfp64 has open correctness bugs
rooted in this gap: GitLab #10822 (wrong values on Asahi),
#15195 (denorm handling in cl_khr_fp64 emulation), #14192 (iris
subgroup perf). Panfrost is the one driver with an unmet
softfp64 request (`pan_screen.c` doesn't set `caps->doubles`);
Freedreno/ir3 already ships softfp64
(`freedreno_screen.c:506`), LLVMpipe runs on CPU with hardware
fp64, Lima has no compute path.

**The fit.** Port the SLEEF-derived transcendental bodies from
`src/sleef/` into libclc as the `cl_khr_fp64` backend for
softfp64 devices. Additive where Mesa has a real gap.

**PR shape.** Libclc patches under `generic/lib/math/` as C
source, parameterized on fp64 representation. No soft-fp64
in-tree dependency in Mesa required — port as first-party libclc
code (license-compatible: libclc is Apache-2.0-with-LLVM-
exception, our SLEEF port is Boost-1.0).

**Hurdles.** Libclc's fp64 story is historically minimal;
reviewers may push back on a large transcendental port from an
external project. Landing incrementally (one transcendental
family per MR, with the rusticl bug it fixes cited) is the
practical path.

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
