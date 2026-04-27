#pragma once

// Hidden-visibility, header-inlined classify / sign-magnitude / exponent-
// manipulation helpers (fcmp / trunc / ldexp / frexp / fabs / neg).
//
// Rationale. Post-1.1 disasm of `sf64_pow` showed ~15 `bl _sf64_fcmp`,
// ~3 `bl _sf64_trunc`, ~3 `bl _sf64_fabs`, plus `bl _sf64_ldexp` /
// `bl _sf64_frexp` / `bl _sf64_neg` per invocation. On Apple Silicon
// each is a full call frame plus (in tls fenv mode) a `__tlv_get_addr`
// roundtrip; the public-ABI cost is the dominant remaining contributor
// to the residual `sf64_pow` regression vs the 1.0 baseline.
//
// This header exposes a hidden-visibility inline surface for the same
// bodies so SLEEF (and any other internal caller) can fully inline
// every classify / manipulation op into the caller's instruction
// stream. The implementations are literal extracts of the public-ABI
// bodies in src/classify.cpp / src/compare.cpp / src/rounding.cpp —
// no algorithmic change, no tolerance widening, no signed-zero / NaN-
// payload reshuffle. Bits-in-bits-out identical for every input.
//
// Visibility. Every function below is `static inline
// __attribute__((always_inline)) __attribute__((visibility("hidden")))`.
// The `sf64_internal_` prefix matches the cross-TU-helper convention
// documented in CLAUDE.md § Hard constraints — these never emit
// external symbols and therefore never appear on the `nm -g` ABI
// surface.
//
// SF64_NO_OPT carve-out. The public `sf64_fabs` / `sf64_neg` /
// `sf64_copysign` / `sf64_fcmp` entries carry `SF64_NO_OPT` as
// defense-in-depth against an AGX recursion bug at the AdaptiveCpp
// intrinsic-rewrite path. The internal helpers below intentionally do
// NOT carry that attribute — they inline directly into their callers
// and never become emitted intrinsics that ReplaceIntrinsics would
// target.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "soft_fp64/soft_f64.h"

#include <climits>
#include <cstdint>

namespace soft_fp64::internal {

#define SF64_INTERNAL_INLINE                                                                       \
    static inline __attribute__((always_inline)) __attribute__((visibility("hidden")))

// Larger bodies (ldexp / frexp) keep `inline` only — let the compiler
// decide whether to inline. Forcing `always_inline` inflates callers
// like `xexp` / `xatanh` that ldexp at multiple sites with the full
// subnormal-handling body, which costs more in icache pressure than the
// saved call frame. The hidden-visibility attribute alone is enough to
// keep these off the public ABI surface.
#define SF64_INTERNAL_INLINE_LARGE static inline __attribute__((visibility("hidden")))

// ---------------------------------------------------------------------------
// fabs / neg — pure bit ops
// ---------------------------------------------------------------------------
// Bodies lifted from src/classify.cpp:75 (sf64_fabs) — the public
// `sf64_neg` body is the same XOR with kSignMask. internal_arith.h already
// exposes `sf64_internal_neg` for use by sleef_common.h; we re-expose it
// here too so callers that include only this header have it available.

SF64_INTERNAL_INLINE double sf64_internal_fabs(double x) noexcept {
    // SAFETY: bit_cast then clear sign bit. No host FP arithmetic.
    return from_bits(bits_of(x) & ~kSignMask);
}

// `sf64_internal_neg` is already defined in internal_arith.h as a member of
// the same namespace; we don't redefine it here. Callers needing it should
// include either header. The sleef_fe_macros.h `sf64_neg` macro routes to
// that pre-existing definition.

// ---------------------------------------------------------------------------
// fcmp — full 16-predicate switch (LLVM FCmpInst::Predicate)
// ---------------------------------------------------------------------------
// Body lifted from src/compare.cpp::fcmp_impl. Inputs are by-value `double`
// so callers don't need to bit_cast at the call site; ordered_lt / ordered_eq
// are inlined here (file-static in compare.cpp namespace; we replicate them).

SF64_INTERNAL_INLINE bool sf64_internal_fcmp_ordered_lt(uint64_t a, uint64_t b) noexcept {
    if (is_zero_bits(a) && is_zero_bits(b)) {
        return false;
    }
    const int64_t sa = static_cast<int64_t>(a) ^ ((static_cast<int64_t>(a) >> 63) &
                                                  static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
    const int64_t sb = static_cast<int64_t>(b) ^ ((static_cast<int64_t>(b) >> 63) &
                                                  static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
    return sa < sb;
}

SF64_INTERNAL_INLINE bool sf64_internal_fcmp_ordered_eq(uint64_t a, uint64_t b) noexcept {
    if (a == b)
        return true;
    return is_zero_bits(a) && is_zero_bits(b);
}

SF64_INTERNAL_INLINE int sf64_internal_fcmp(double a_, double b_, int pred) noexcept {
    const uint64_t a = bits_of(a_);
    const uint64_t b = bits_of(b_);
    const bool nan = is_nan_bits(a) || is_nan_bits(b);
    switch (pred) {
    case 0:
        return 0;
    case 1:
        return !nan && sf64_internal_fcmp_ordered_eq(a, b) ? 1 : 0;
    case 2:
        return !nan && sf64_internal_fcmp_ordered_lt(b, a) ? 1 : 0;
    case 3:
        return !nan && !sf64_internal_fcmp_ordered_lt(a, b) ? 1 : 0;
    case 4:
        return !nan && sf64_internal_fcmp_ordered_lt(a, b) ? 1 : 0;
    case 5:
        return !nan && !sf64_internal_fcmp_ordered_lt(b, a) ? 1 : 0;
    case 6:
        return !nan && !sf64_internal_fcmp_ordered_eq(a, b) ? 1 : 0;
    case 7:
        return !nan ? 1 : 0;
    case 8:
        return nan ? 1 : 0;
    case 9:
        return (nan || sf64_internal_fcmp_ordered_eq(a, b)) ? 1 : 0;
    case 10:
        return (nan || sf64_internal_fcmp_ordered_lt(b, a)) ? 1 : 0;
    case 11:
        return (nan || !sf64_internal_fcmp_ordered_lt(a, b)) ? 1 : 0;
    case 12:
        return (nan || sf64_internal_fcmp_ordered_lt(a, b)) ? 1 : 0;
    case 13:
        return (nan || !sf64_internal_fcmp_ordered_lt(b, a)) ? 1 : 0;
    case 14:
        return (nan || !sf64_internal_fcmp_ordered_eq(a, b)) ? 1 : 0;
    case 15:
        return 1;
    default:
        return 0;
    }
}

// ---------------------------------------------------------------------------
// trunc — bit-level truncation
// ---------------------------------------------------------------------------
// Body lifted from src/rounding.cpp::trunc_bits (the file-static helper
// that the public `sf64_trunc` delegates to).

SF64_INTERNAL_INLINE double sf64_internal_trunc(double x) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t e = extract_exp(b);

    if (e == kExpMax)
        return x;

    const int unbiased = static_cast<int>(e) - kExpBias;

    if (unbiased < 0) {
        return make_signed_zero(extract_sign(b));
    }
    if (unbiased >= kFracBits) {
        return x;
    }

    const uint64_t frac_mask_below = kFracMask >> unbiased;
    return from_bits(b & ~frac_mask_below);
}

// ---------------------------------------------------------------------------
// ldexp — scale x by 2^n (full subnormal handling, RNE round)
// ---------------------------------------------------------------------------
// Body lifted from src/rounding.cpp::sf64_ldexp. The non-trivial subnormal
// branch (right-shift with round-to-nearest-even on the dropped bits) lifts
// cleanly because the public body is already a single static-shaped
// function with no host-FP ops and no fenv plumbing.

SF64_INTERNAL_INLINE_LARGE double sf64_internal_ldexp(double x, int n) noexcept {
    const uint64_t b = bits_of(x);

    if (is_nan_bits(b)) {
        return from_bits(b | kQuietNaNBit);
    }
    if (is_inf_bits(b) || is_zero_bits(b))
        return x;

    const uint32_t sign = extract_sign(b);

    if (n > 2100)
        n = 2100;
    if (n < -2100)
        n = -2100;

    uint64_t frac = extract_frac(b);
    int exp_unbiased;
    if (extract_exp(b) == 0) {
        const int lz = clz64(frac);
        const int shift = lz - (63 - kFracBits);
        frac = (frac << shift) & kFracMask;
        exp_unbiased = 1 - kExpBias - shift;
    } else {
        exp_unbiased = static_cast<int>(extract_exp(b)) - kExpBias;
    }

    const int new_exp_unbiased = exp_unbiased + n;

    if (new_exp_unbiased > 1023) {
        return make_signed_inf(sign);
    }

    if (new_exp_unbiased >= -1022) {
        const uint32_t new_exp_biased = static_cast<uint32_t>(new_exp_unbiased + kExpBias);
        return from_bits(pack(sign, new_exp_biased, frac));
    }

    const int rshift = -1022 - new_exp_unbiased;
    if (rshift >= 64) {
        return make_signed_zero(sign);
    }
    const uint64_t full_mant = kImplicitBit | frac;

    uint64_t retained;
    uint64_t shifted_out;
    if (rshift == 0) {
        retained = full_mant;
        shifted_out = 0;
    } else {
        retained = full_mant >> rshift;
        const uint64_t drop_mask = (uint64_t{1} << rshift) - 1;
        shifted_out = full_mant & drop_mask;
    }

    uint64_t round_bit = 0;
    uint64_t sticky_bit = 0;
    if (rshift > 0) {
        round_bit = (shifted_out >> (rshift - 1)) & uint64_t{1};
        const uint64_t sticky_mask =
            (rshift >= 2) ? ((uint64_t{1} << (rshift - 1)) - 1) : uint64_t{0};
        sticky_bit = (shifted_out & sticky_mask) ? uint64_t{1} : uint64_t{0};
    }

    if (round_bit != 0 && (sticky_bit != 0 || (retained & 1u) != 0)) {
        retained = retained + 1;
    }

    if ((retained & kImplicitBit) != 0) {
        return from_bits(pack(sign, 1u, retained & kFracMask));
    }
    return from_bits(pack(sign, 0u, retained & kFracMask));
}

// ---------------------------------------------------------------------------
// frexp — split into (fraction in [0.5, 1), *exp)
// ---------------------------------------------------------------------------
// Body lifted from src/rounding.cpp::sf64_frexp.

SF64_INTERNAL_INLINE_LARGE double sf64_internal_frexp(double x, int* exp_out) noexcept {
    const uint64_t b = bits_of(x);

    if (is_nan_bits(b)) {
        if (exp_out)
            *exp_out = 0;
        return from_bits(b | kQuietNaNBit);
    }
    if (is_inf_bits(b)) {
        if (exp_out)
            *exp_out = 0;
        return x;
    }
    if (is_zero_bits(b)) {
        if (exp_out)
            *exp_out = 0;
        return x;
    }

    const uint32_t sign = extract_sign(b);
    uint64_t frac = extract_frac(b);
    int exp_unbiased;

    if (extract_exp(b) == 0) {
        const int lz = clz64(frac);
        const int shift = lz - (63 - kFracBits);
        frac = (frac << shift) & kFracMask;
        exp_unbiased = 1 - kExpBias - shift;
    } else {
        exp_unbiased = static_cast<int>(extract_exp(b)) - kExpBias;
    }

    if (exp_out)
        *exp_out = exp_unbiased + 1;
    return from_bits(pack(sign, static_cast<uint32_t>(kExpBias - 1), frac));
}

#undef SF64_INTERNAL_INLINE
#undef SF64_INTERNAL_INLINE_LARGE

} // namespace soft_fp64::internal
