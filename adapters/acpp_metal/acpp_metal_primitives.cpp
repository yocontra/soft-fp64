// Forwarder shims: AdaptiveCpp Metal SSCP primitives → soft-fp64 sf64_*.
//
// This file is staged into the AdaptiveCpp Metal libkernel bitcode build
// via -DACPP_METAL_EXTERNAL_FP64_DIR=<path>. Every symbol below is one of
// the "Required primitives" listed in
// src/libkernel/sscp/metal/float64/README.md of the pinned
// AdaptiveCpp fork-safe-metal branch. Each body is a one-line forward
// to the matching sf64_* entry point — no local arithmetic, no host FPU
// dependency, no SYCL-isms.
//
// Visibility: the contract requires default visibility with non-static
// C linkage (HIPSYCL_SSCP_BUILTIN). We define a local macro rather than
// pulling in AdaptiveCpp's builtin_config.hpp to keep this adapter
// decoupled from upstream header layout.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <cstdint>

#define ACPP_METAL_FP64_EXPORT extern "C" __attribute__((__visibility__("default")))

// ---- Arithmetic ------------------------------------------------------------

ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_add(double a, double b) {
    return sf64_add(a, b);
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_sub(double a, double b) {
    return sf64_sub(a, b);
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_mul(double a, double b) {
    return sf64_mul(a, b);
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_div(double a, double b) {
    return sf64_div(a, b);
}
// LLVM `frem` (fmod semantics: sign of result = sign of dividend); maps to
// sf64_rem. sf64_fmod is documented as identical semantics — pick one.
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_rem(double a, double b) {
    return sf64_rem(a, b);
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_neg(double a) {
    return sf64_neg(a);
}

// ---- Min / max (IEEE 754-2008 precise: NaN-propagating) --------------------

ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_fmin_precise(double a, double b) {
    return sf64_fmin_precise(a, b);
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_fmax_precise(double a, double b) {
    return sf64_fmax_precise(a, b);
}

// ---- Conversions: f64 ↔ f32 ------------------------------------------------

ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_from_f32(float a) {
    return sf64_from_f32(a);
}
ACPP_METAL_FP64_EXPORT float __acpp_sscp_soft_f64_to_f32(double a) {
    return sf64_to_f32(a);
}

// ---- Conversions: f64 ↔ signed integer ------------------------------------

ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_from_i32(int a) {
    return sf64_from_i32(static_cast<int32_t>(a));
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_from_i64(long a) {
    return sf64_from_i64(static_cast<int64_t>(a));
}
ACPP_METAL_FP64_EXPORT int __acpp_sscp_soft_f64_to_i32(double a) {
    return static_cast<int>(sf64_to_i32(a));
}
ACPP_METAL_FP64_EXPORT long __acpp_sscp_soft_f64_to_i64(double a) {
    return static_cast<long>(sf64_to_i64(a));
}

// ---- Conversions: f64 ↔ unsigned integer ----------------------------------

ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_from_u32(unsigned int a) {
    return sf64_from_u32(static_cast<uint32_t>(a));
}
ACPP_METAL_FP64_EXPORT double __acpp_sscp_soft_f64_from_u64(unsigned long a) {
    return sf64_from_u64(static_cast<uint64_t>(a));
}
ACPP_METAL_FP64_EXPORT unsigned int __acpp_sscp_soft_f64_to_u32(double a) {
    return static_cast<unsigned int>(sf64_to_u32(a));
}
ACPP_METAL_FP64_EXPORT unsigned long __acpp_sscp_soft_f64_to_u64(double a) {
    return static_cast<unsigned long>(sf64_to_u64(a));
}

// ---- Narrow integer conversions --------------------------------------------
// Metal SSCP emitter Emitter.cpp:988-1007 emits _to_i16/_to_i8/_to_u16/_to_u8
// for narrow integer types. Signatures return signed/unsigned N-bit ints.

ACPP_METAL_FP64_EXPORT signed char __acpp_sscp_soft_f64_to_i8(double a) {
    return static_cast<signed char>(sf64_to_i8(a));
}
ACPP_METAL_FP64_EXPORT short __acpp_sscp_soft_f64_to_i16(double a) {
    return static_cast<short>(sf64_to_i16(a));
}
ACPP_METAL_FP64_EXPORT unsigned char __acpp_sscp_soft_f64_to_u8(double a) {
    return static_cast<unsigned char>(sf64_to_u8(a));
}
ACPP_METAL_FP64_EXPORT unsigned short __acpp_sscp_soft_f64_to_u16(double a) {
    return static_cast<unsigned short>(sf64_to_u16(a));
}

// ---- Compare ---------------------------------------------------------------
// sf64_fcmp takes the same LLVM FCmpInst::Predicate encoding (0..15) that
// the Metal SSCP emitter uses — see Emitter.cpp:1321. Pass-through forward.

ACPP_METAL_FP64_EXPORT int __acpp_sscp_soft_f64_fcmp(double lhs, double rhs, int pred) {
    return sf64_fcmp(lhs, rhs, pred);
}
