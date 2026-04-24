#pragma once

// Hidden-visibility, header-inlined RNE specializations of the hot
// arithmetic primitives (add / sub / mul / div / fma / sqrt).
//
// Rationale. The 1.1 rounding-mode parameter adds a 5-way switch to the
// public sf64_add / sub / mul / div / fma / sqrt entries. Cross-TU callers
// (notably the SLEEF DD primitives in src/sleef/) pay that switch on every
// invocation because SF64_ALWAYS_INLINE does not cross the archive-TU
// boundary without library-wide LTO. sf64_pow alone makes ~160 arithmetic
// calls per invocation and the cost accumulates (+43-58% vs 1.0 baseline).
//
// This header exposes an RNE-specialized inline surface so SLEEF (and
// anything else that always needs round-to-nearest-even semantics) can
// inline the full body and skip the switch. The implementations are
// literal extracts of the RNE path from src/arithmetic.cpp /
// src/sqrt_fma.cpp — no algorithmic change, no tolerance widening, no
// signed-zero / NaN-payload reshuffle. Bits-in-bits-out identical for
// every input.
//
// Visibility. Every function below is static-inline SF64_ALWAYS_INLINE.
// The `sf64_internal_` prefix matches the cross-TU-helper convention
// documented in CLAUDE.md § Hard constraints — these never emit external
// symbols and therefore never appear on the `nm -g` ABI surface.
//
// Flag plumbing. Each helper takes a `sf64_internal_fe_acc&` as the
// final parameter. Callers declare one accumulator per public op, thread
// it through every internal arithmetic call, and flush once to TLS at
// end-of-op. Under SOFT_FP64_FENV_MODE==0 (disabled) the accumulator is
// an empty class whose methods are no-ops — the parameter DCEs.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

#include <cstdint>

namespace soft_fp64::internal {

// ===========================================================================
// Shared layout notes
// ===========================================================================
//
// During add/sub we keep the significand in a 64-bit word shifted left by 9
// bits:
//
//   bit 62       : overflow bit (set when an add/mul carries out)
//   bit 61       : implicit bit when normalised
//   bit 61..9    : 53 mantissa bits (implicit + 52 fraction)
//   bit 8        : round (guard) bit
//   bit 7..0     : sticky bits

// Extract (sign, raw_exp, sig_shifted_left_by_9). Does NOT handle NaN / inf
// / zero — caller pre-filters. Identical to arithmetic.cpp::unpack_finite_nonzero.
struct UnpackedArith {
    uint32_t sign;
    int32_t exp;  // working (biased) exponent
    uint64_t sig; // significand shifted left by 9 (implicit bit at bit 62)
};

SF64_ALWAYS_INLINE UnpackedArith arith_unpack_finite_nonzero(uint64_t bits) noexcept {
    UnpackedArith u;
    u.sign = extract_sign(bits);
    const uint32_t raw_exp = extract_exp(bits);
    const uint64_t frac = extract_frac(bits);
    if (raw_exp == 0) {
        // Subnormal. Caller guarantees frac != 0, so clz is defined.
        const int lz = __builtin_clzll(frac);
        const int shift = lz - 11;
        u.sig = (frac << shift) << 9;
        u.exp = 1 - shift;
    } else {
        u.exp = static_cast<int32_t>(raw_exp);
        u.sig = (frac | kImplicitBit) << 9;
    }
    return u;
}

// ---------------------------------------------------------------------------
// Round-and-pack, RNE specialization
// ---------------------------------------------------------------------------
// `exp` is the biased target exponent corresponding to the implicit bit
// sitting at bit 61 of `sig`. RNE-only: the 5-way mode switch is folded
// into the two overflow and decision constants.
SF64_ALWAYS_INLINE double arith_round_and_pack_rne(uint32_t sign, int32_t exp, uint64_t sig,
                                                   sf64_internal_fe_acc& fe) noexcept {
    if (exp >= 0x7FF) {
        fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
        // RNE overflow -> signed infinity.
        return from_bits(pack(sign, 0x7FFu, 0));
    }

    const bool was_tiny_before_rounding = (exp <= 0);
    if (was_tiny_before_rounding) {
        const int shift = 1 - exp;
        sig = shift_right_jamming(sig, shift);
        exp = 0;
    }

    const bool round_bit = ((sig >> 8) & 1ULL) != 0;
    const bool sticky = (sig & 0xFFULL) != 0;
    const bool lsb = ((sig >> 9) & 1ULL) != 0;
    const bool inexact = round_bit || sticky;

    uint64_t rounded = sig;
    // RNE: round up iff round_bit && (sticky || lsb).
    if (round_bit && (sticky || lsb)) {
        rounded += (1ULL << 9);
    }

    if ((rounded >> 62) & 1ULL) {
        rounded = shift_right_jamming(rounded, 1);
        ++exp;
        if (exp >= 0x7FF) {
            fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
            return from_bits(pack(sign, 0x7FFu, 0));
        }
    }

    const uint64_t mantissa = (rounded >> 9) & kFracMask;
    const bool implicit_set = ((rounded >> 61) & 1ULL) != 0;

    uint32_t final_exp;
    if (exp == 0) {
        final_exp = implicit_set ? 1u : 0u;
    } else {
        if (!implicit_set && mantissa == 0) {
            final_exp = 0u;
        } else {
            final_exp = static_cast<uint32_t>(exp);
        }
    }

    if (inexact) {
        if (was_tiny_before_rounding) {
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        } else {
            fe.raise(SF64_FE_INEXACT);
        }
    }

    return from_bits(pack(sign, final_exp, mantissa));
}

// Add two significands that share the same sign. RNE only.
SF64_ALWAYS_INLINE double arith_add_magnitudes_rne(uint32_t sign, uint64_t a_bits, uint64_t b_bits,
                                                   sf64_internal_fe_acc& fe) noexcept {
    const UnpackedArith a = arith_unpack_finite_nonzero(a_bits);
    const UnpackedArith b = arith_unpack_finite_nonzero(b_bits);

    int32_t exp_diff = a.exp - b.exp;
    int32_t exp;
    uint64_t sig_a, sig_b;
    if (exp_diff > 0) {
        sig_a = a.sig;
        sig_b = shift_right_jamming(b.sig, exp_diff);
        exp = a.exp;
    } else if (exp_diff < 0) {
        sig_a = shift_right_jamming(a.sig, -exp_diff);
        sig_b = b.sig;
        exp = b.exp;
    } else {
        sig_a = a.sig;
        sig_b = b.sig;
        exp = a.exp;
    }

    uint64_t sum = sig_a + sig_b;
    if ((sum >> 62) & 1ULL) {
        sum = shift_right_jamming(sum, 1);
        ++exp;
    }

    return arith_round_and_pack_rne(sign, exp, sum, fe);
}

// Subtract magnitudes. RNE-only; zero_sign for exact cancellation is always +0.
SF64_ALWAYS_INLINE double arith_sub_magnitudes_rne(uint32_t a_sign, uint64_t a_bits,
                                                   uint64_t b_bits,
                                                   sf64_internal_fe_acc& fe) noexcept {
    const UnpackedArith a = arith_unpack_finite_nonzero(a_bits);
    const UnpackedArith b = arith_unpack_finite_nonzero(b_bits);

    int32_t exp_diff = a.exp - b.exp;
    int32_t exp;
    uint64_t sig_a, sig_b;
    uint32_t sign = a_sign;

    if (exp_diff > 0) {
        sig_a = a.sig;
        sig_b = shift_right_jamming(b.sig, exp_diff);
        exp = a.exp;
    } else if (exp_diff < 0) {
        sig_a = b.sig;
        sig_b = shift_right_jamming(a.sig, -exp_diff);
        exp = b.exp;
        sign ^= 1u;
    } else {
        if (a.sig > b.sig) {
            sig_a = a.sig;
            sig_b = b.sig;
            exp = a.exp;
        } else if (a.sig < b.sig) {
            sig_a = b.sig;
            sig_b = a.sig;
            exp = a.exp;
            sign ^= 1u;
        } else {
            return make_signed_zero(0u); // RNE: exact cancellation -> +0.
        }
    }

    uint64_t diff = sig_a - sig_b;
    if (diff == 0) {
        return make_signed_zero(0u);
    }

    const int lz = __builtin_clzll(diff);
    const int norm_shift = lz - 2;
    diff <<= norm_shift;
    exp -= norm_shift;

    return arith_round_and_pack_rne(sign, exp, diff, fe);
}

SF64_ALWAYS_INLINE double arith_mul_core_rne(uint64_t ab, uint64_t bb, uint32_t r_sign,
                                             sf64_internal_fe_acc& fe) noexcept {
    int32_t exp_a, exp_b;
    uint64_t sig_a, sig_b;
    {
        const UnpackedArith ua = arith_unpack_finite_nonzero(ab);
        const UnpackedArith ub = arith_unpack_finite_nonzero(bb);
        sig_a = ua.sig >> 9;
        sig_b = ub.sig >> 9;
        exp_a = ua.exp;
        exp_b = ub.exp;
    }

    const __uint128_t product = static_cast<__uint128_t>(sig_a) * static_cast<__uint128_t>(sig_b);
    int32_t exp_r = exp_a + exp_b - kExpBias;

    uint64_t sig;
    if ((product >> 105) & 1u) {
        const int shift = 44;
        const __uint128_t lost_mask = ((static_cast<__uint128_t>(1) << shift) - 1);
        const bool sticky = (product & lost_mask) != 0;
        sig = static_cast<uint64_t>(product >> shift);
        if (sticky)
            sig |= 1ULL;
        ++exp_r;
    } else {
        const int shift = 43;
        const __uint128_t lost_mask = ((static_cast<__uint128_t>(1) << shift) - 1);
        const bool sticky = (product & lost_mask) != 0;
        sig = static_cast<uint64_t>(product >> shift);
        if (sticky)
            sig |= 1ULL;
    }

    return arith_round_and_pack_rne(r_sign, exp_r, sig, fe);
}

SF64_ALWAYS_INLINE double arith_div_core_rne(uint64_t ab, uint64_t bb, uint32_t r_sign,
                                             sf64_internal_fe_acc& fe) noexcept {
    const UnpackedArith ua = arith_unpack_finite_nonzero(ab);
    const UnpackedArith ub = arith_unpack_finite_nonzero(bb);
    uint64_t sig_a = ua.sig >> 9;
    const uint64_t sig_b = ub.sig >> 9;

    int32_t exp_r = ua.exp - ub.exp + kExpBias;

    if (sig_a < sig_b) {
        sig_a <<= 1;
        --exp_r;
    }

    const __uint128_t num = static_cast<__uint128_t>(sig_a) << 55;
    const __uint128_t den = static_cast<__uint128_t>(sig_b);
    const uint64_t quotient = static_cast<uint64_t>(num / den);
    const __uint128_t remainder = num % den;

    uint64_t sig = quotient << 6;
    if (remainder != 0) {
        sig |= 1ULL;
    }

    if ((sig >> 62) & 1ULL) {
        sig = shift_right_jamming(sig, 1);
        ++exp_r;
    }

    return arith_round_and_pack_rne(r_sign, exp_r, sig, fe);
}

// ===========================================================================
// Public RNE-specialized helpers
// ===========================================================================

// The top-level `sf64_internal_*_rne` helpers are SF64_ALWAYS_INLINE so
// they cross the TU boundary at every call site and the RNE body (with
// its NaN/zero/inf dispatch) fuses into the caller's instruction stream.
// This is the point of the header — the 1.1 regression came from the
// DD primitives paying the cross-TU call boundary plus the 5-way mode
// switch on every sf64_add/sub/mul/div/fma/sqrt, and fully inlining the
// RNE-only body removes both costs. The inner `arith_*_core_rne` /
// `arith_*_magnitudes_rne` / `arith_round_and_pack_rne` bodies are kept
// plain `inline` so the compiler can decide to emit them out-of-line
// when a large poly (e.g. sleef_exp_log's 10-term Horner) would bloat
// the caller — the entry wrapper stays inlined, the magnitude body may
// not.
SF64_ALWAYS_INLINE double sf64_internal_add_rne(double a, double b,
                                                sf64_internal_fe_acc& fe) noexcept {
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint32_t a_exp = extract_exp(ab);
    const uint32_t b_exp = extract_exp(bb);
    const uint32_t a_sign = extract_sign(ab);
    const uint32_t b_sign = extract_sign(bb);

    if (is_nan_bits(ab) || is_nan_bits(bb)) {
        return propagate_nan(ab, bb);
    }

    if (a_exp == kExpMax || b_exp == kExpMax) {
        const bool a_inf = (a_exp == kExpMax);
        const bool b_inf = (b_exp == kExpMax);
        if (a_inf && b_inf) {
            if (a_sign == b_sign)
                return make_signed_inf(a_sign);
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return a_inf ? make_signed_inf(a_sign) : make_signed_inf(b_sign);
    }

    const bool a_zero = is_zero_bits(ab);
    const bool b_zero = is_zero_bits(bb);
    if (a_zero && b_zero) {
        // RNE: +0 + -0 -> +0. Same-sign zeros preserve sign.
        if (a_sign == b_sign)
            return make_signed_zero(a_sign);
        return make_signed_zero(0u);
    }
    if (a_zero)
        return b;
    if (b_zero)
        return a;

    if (a_sign == b_sign) {
        return arith_add_magnitudes_rne(a_sign, ab, bb, fe);
    }
    return arith_sub_magnitudes_rne(a_sign, ab, bb, fe);
}

SF64_ALWAYS_INLINE double sf64_internal_sub_rne(double a, double b,
                                                sf64_internal_fe_acc& fe) noexcept {
    // a - b = a + (-b); flip sign bit of b via integer XOR.
    const uint64_t bb = bits_of(b);
    return sf64_internal_add_rne(a, from_bits(bb ^ kSignMask), fe);
}

SF64_ALWAYS_INLINE double sf64_internal_mul_rne(double a, double b,
                                                sf64_internal_fe_acc& fe) noexcept {
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint32_t a_sign = extract_sign(ab);
    const uint32_t b_sign = extract_sign(bb);
    const uint32_t r_sign = a_sign ^ b_sign;
    const uint32_t a_exp = extract_exp(ab);
    const uint32_t b_exp = extract_exp(bb);

    if (is_nan_bits(ab) || is_nan_bits(bb))
        return propagate_nan(ab, bb);

    if (a_exp == kExpMax || b_exp == kExpMax) {
        if (is_zero_bits(ab) || is_zero_bits(bb)) {
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return make_signed_inf(r_sign);
    }

    if (is_zero_bits(ab) || is_zero_bits(bb))
        return make_signed_zero(r_sign);

    return arith_mul_core_rne(ab, bb, r_sign, fe);
}

SF64_ALWAYS_INLINE double sf64_internal_div_rne(double a, double b,
                                                sf64_internal_fe_acc& fe) noexcept {
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint32_t a_sign = extract_sign(ab);
    const uint32_t b_sign = extract_sign(bb);
    const uint32_t r_sign = a_sign ^ b_sign;
    const uint32_t a_exp = extract_exp(ab);
    const uint32_t b_exp = extract_exp(bb);

    if (is_nan_bits(ab) || is_nan_bits(bb))
        return propagate_nan(ab, bb);

    const bool a_inf = (a_exp == kExpMax);
    const bool b_inf = (b_exp == kExpMax);
    if (a_inf && b_inf) {
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }
    if (a_inf)
        return make_signed_inf(r_sign);
    if (b_inf)
        return make_signed_zero(r_sign);

    const bool a_zero = is_zero_bits(ab);
    const bool b_zero = is_zero_bits(bb);
    if (a_zero && b_zero) {
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }
    if (b_zero) {
        fe.raise(SF64_FE_DIVBYZERO);
        return make_signed_inf(r_sign);
    }
    if (a_zero)
        return make_signed_zero(r_sign);

    return arith_div_core_rne(ab, bb, r_sign, fe);
}

// ===========================================================================
// sqrt (RNE-only specialization)
// ===========================================================================
// Digit-by-digit integer sqrt on the 53-bit mantissa. Identical algorithm
// to src/sqrt_fma.cpp's sqrt_r_impl with the mode switch folded away.

struct IsqrtResultInternal {
    uint64_t root;
    uint64_t remainder;
    bool remainder_nonzero;
};

SF64_ALWAYS_INLINE IsqrtResultInternal arith_isqrt_bits(uint64_t radicand_hi, uint64_t radicand_lo,
                                                        int num_result_bits) noexcept {
    __uint128_t rem = 0;
    __uint128_t root = 0;
    __uint128_t rad = ((__uint128_t)radicand_hi << 64) | (__uint128_t)radicand_lo;
    const int total_radicand_bits = num_result_bits * 2;
    if (total_radicand_bits < 128) {
        rad = rad << (128 - total_radicand_bits);
    }

    for (int i = 0; i < num_result_bits; ++i) {
        const __uint128_t next_pair = (rad >> 126) & 0x3U;
        rad = rad << 2;
        rem = (rem << 2) | next_pair;
        const __uint128_t test = (root << 2) | 0x1U;
        if (rem >= test) {
            rem = rem - test;
            root = (root << 1) | 0x1U;
        } else {
            root = root << 1;
        }
    }

    IsqrtResultInternal r;
    r.root = static_cast<uint64_t>(root);
    r.remainder = static_cast<uint64_t>(rem);
    r.remainder_nonzero = rem != 0;
    return r;
}

SF64_ALWAYS_INLINE double sf64_internal_sqrt_rne(double x, sf64_internal_fe_acc& fe) noexcept {
    const uint64_t bx = bits_of(x);
    const uint32_t sign = extract_sign(bx);
    const uint32_t exp_biased = extract_exp(bx);
    const uint64_t frac = extract_frac(bx);

    if (exp_biased == kExpMax) {
        if (frac != 0) {
            return from_bits(bx | kQuietNaNBit);
        }
        if (sign != 0) {
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return x;
    }
    if (is_zero_bits(bx)) {
        return x;
    }
    if (sign != 0) {
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }

    int64_t unbiased_exp;
    uint64_t mantissa;
    if (exp_biased == 0) {
        const int shift = clz64(frac) - (63 - kFracBits);
        mantissa = frac << shift;
        unbiased_exp = static_cast<int64_t>(1 - kExpBias) - static_cast<int64_t>(shift);
    } else {
        mantissa = frac | kImplicitBit;
        unbiased_exp = static_cast<int64_t>(exp_biased) - static_cast<int64_t>(kExpBias);
    }

    const bool exp_odd = (unbiased_exp & 1) != 0;
    int64_t new_unbiased_exp;
    if (exp_odd) {
        new_unbiased_exp = (unbiased_exp - 1) / 2;
    } else {
        new_unbiased_exp = unbiased_exp / 2;
    }

    const int shift = exp_odd ? 55 : 54;
    constexpr int kRootBits = 54;

    uint64_t radicand_hi;
    uint64_t radicand_lo;
    if (shift >= 64) {
        radicand_hi = mantissa << (shift - 64);
        radicand_lo = 0;
    } else {
        radicand_hi = mantissa >> (64 - shift);
        radicand_lo = mantissa << shift;
    }

    IsqrtResultInternal isr = arith_isqrt_bits(radicand_hi, radicand_lo, kRootBits);
    const uint64_t root = isr.root;
    const bool remainder_nonzero = isr.remainder_nonzero;

    const uint64_t guard_bit = root & 1u;
    const uint64_t root53 = root >> 1;
    const bool sticky = remainder_nonzero;
    const bool lsb = (root53 & 1u) != 0;

    // RNE: round up iff guard && (sticky || lsb).
    uint64_t rounded = root53;
    if (guard_bit && (sticky || lsb)) {
        rounded = root53 + 1u;
    }

    if (guard_bit != 0 || sticky) {
        fe.raise(SF64_FE_INEXACT);
    }

    int64_t biased_new_exp = new_unbiased_exp + static_cast<int64_t>(kExpBias);
    if ((rounded & (uint64_t{1} << 53)) != 0) {
        rounded >>= 1;
        biased_new_exp += 1;
    }

    if (biased_new_exp >= static_cast<int64_t>(kExpMax)) {
        return from_bits(kPositiveInf);
    }
    if (biased_new_exp <= 0) {
        return from_bits(0);
    }

    const uint64_t out_frac = rounded & kFracMask;
    return from_bits(pack(0u, static_cast<uint32_t>(biased_new_exp), out_frac));
}

// ===========================================================================
// fma (RNE-only specialization)
// ===========================================================================

SF64_ALWAYS_INLINE int arith_highest_bit_u128(__uint128_t v) noexcept {
    if (v == 0)
        return -1;
    const uint64_t hi = static_cast<uint64_t>(v >> 64);
    if (hi != 0) {
        return 127 - __builtin_clzll(hi);
    }
    const uint64_t lo = static_cast<uint64_t>(v);
    return 63 - __builtin_clzll(lo);
}

// RNE round-and-pack on a (mag, frame_exp) representation.
// Value = mag * 2^frame_exp.
SF64_ALWAYS_INLINE double arith_fma_round_and_pack_rne(uint32_t sign, __uint128_t mag,
                                                       int64_t frame_exp,
                                                       sf64_internal_fe_acc& fe) noexcept {
    if (mag == 0) {
        return make_signed_zero(sign);
    }
    const int msb = arith_highest_bit_u128(mag);
    const int64_t unbiased = frame_exp + static_cast<int64_t>(msb);
    int64_t biased = unbiased + static_cast<int64_t>(kExpBias);

    if (biased >= static_cast<int64_t>(kExpMax)) {
        fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
        return make_signed_inf(sign);
    }

    int target_lsb;
    bool output_subnormal;
    if (biased >= 1) {
        target_lsb = msb - 52;
        output_subnormal = false;
    } else {
        target_lsb = static_cast<int>(-1074 - frame_exp);
        output_subnormal = true;
    }

    bool round_inexact = false;

    uint64_t rounded_mant;
    if (target_lsb <= 0) {
        rounded_mant = static_cast<uint64_t>(mag);
        if (target_lsb < 0) {
            rounded_mant = rounded_mant << (-target_lsb);
        }
    } else if (target_lsb >= 128) {
        if (mag != 0) {
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        }
        return make_signed_zero(sign);
    } else {
        const __uint128_t round_bit_val = ((__uint128_t)1) << (target_lsb - 1);
        const bool round_bit = (mag & round_bit_val) != 0;
        bool sticky;
        if (target_lsb >= 2) {
            const __uint128_t sticky_mask = round_bit_val - 1;
            sticky = (mag & sticky_mask) != 0;
        } else {
            sticky = false;
        }
        round_inexact = round_bit || sticky;
        const __uint128_t trunc = mag >> target_lsb;
        rounded_mant = static_cast<uint64_t>(trunc);
        const bool lsb = (rounded_mant & 1u) != 0;
        // RNE: round up iff round_bit && (sticky || lsb).
        if (round_bit && (sticky || lsb)) {
            rounded_mant += 1u;
        }
    }

    if (output_subnormal) {
        if (round_inexact) {
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        }
        if ((rounded_mant & kImplicitBit) != 0) {
            return from_bits(pack(sign, 1u, rounded_mant & kFracMask));
        }
        return from_bits(pack(sign, 0u, rounded_mant));
    }

    if (round_inexact) {
        fe.raise(SF64_FE_INEXACT);
    }

    if ((rounded_mant & (uint64_t{1} << 53)) != 0) {
        rounded_mant >>= 1;
        biased += 1;
        if (biased >= static_cast<int64_t>(kExpMax)) {
            fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
            return make_signed_inf(sign);
        }
    }
    return from_bits(pack(sign, static_cast<uint32_t>(biased), rounded_mant & kFracMask));
}

SF64_ALWAYS_INLINE double sf64_internal_fma_rne(double a, double b, double c,
                                                sf64_internal_fe_acc& fe) noexcept {
    const uint64_t ba = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint64_t bc = bits_of(c);

    const uint32_t sa = extract_sign(ba);
    const uint32_t sb = extract_sign(bb);
    const uint32_t sc = extract_sign(bc);
    const uint32_t ea = extract_exp(ba);
    const uint32_t eb = extract_exp(bb);
    const uint32_t ec = extract_exp(bc);
    const uint64_t fa = extract_frac(ba);
    const uint64_t fb = extract_frac(bb);
    const uint64_t fc = extract_frac(bc);

    const bool a_nan = (ea == kExpMax) && (fa != 0);
    const bool b_nan = (eb == kExpMax) && (fb != 0);
    const bool c_nan = (ec == kExpMax) && (fc != 0);
    const bool a_inf = (ea == kExpMax) && (fa == 0);
    const bool b_inf = (eb == kExpMax) && (fb == 0);
    const bool c_inf = (ec == kExpMax) && (fc == 0);
    const bool a_zero = is_zero_bits(ba);
    const bool b_zero = is_zero_bits(bb);

    const uint32_t sign_ab = sa ^ sb;

    if ((a_inf && b_zero) || (a_zero && b_inf)) {
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }

    if (a_nan || b_nan || c_nan) {
        if (a_nan)
            return from_bits(ba | kQuietNaNBit);
        if (b_nan)
            return from_bits(bb | kQuietNaNBit);
        return from_bits(bc | kQuietNaNBit);
    }

    if (a_inf || b_inf) {
        if (c_inf) {
            if (sign_ab == sc)
                return make_signed_inf(sign_ab);
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return make_signed_inf(sign_ab);
    }

    if (c_inf) {
        return make_signed_inf(sc);
    }

    if (a_zero || b_zero) {
        if (is_zero_bits(bc)) {
            if (sign_ab == sc) {
                return make_signed_zero(sign_ab);
            }
            return make_signed_zero(0u);
        }
        return c;
    }

    auto normalise = [](uint32_t exp_biased, uint64_t frac, uint64_t& mant_out, int64_t& exp_out) {
        if (exp_biased == 0) {
            const int shift = clz64(frac) - (63 - kFracBits);
            mant_out = frac << shift;
            exp_out = static_cast<int64_t>(1 - kExpBias) - static_cast<int64_t>(shift);
        } else {
            mant_out = frac | kImplicitBit;
            exp_out = static_cast<int64_t>(exp_biased) - static_cast<int64_t>(kExpBias);
        }
    };

    uint64_t ma;
    int64_t expa;
    normalise(ea, fa, ma, expa);
    uint64_t mb;
    int64_t expb;
    normalise(eb, fb, mb, expb);

    const __uint128_t prod = (__uint128_t)ma * (__uint128_t)mb;
    int64_t prod_exp = expa + expb - 104;

    if (is_zero_bits(bc)) {
        return arith_fma_round_and_pack_rne(sign_ab, prod, prod_exp, fe);
    }

    uint64_t mc;
    int64_t expc;
    normalise(ec, fc, mc, expc);

    const int prod_msb_local = arith_highest_bit_u128(prod);
    const int64_t prod_true_msb = prod_exp + static_cast<int64_t>(prod_msb_local);
    const int64_t c_true_msb = expc;
    const int64_t frame_top = (prod_true_msb > c_true_msb ? prod_true_msb : c_true_msb) + 2;
    const int64_t E = frame_top - 127;

    const int64_t prod_shift = prod_exp - E;
    __uint128_t prod_frame;
    if (prod_shift >= 0) {
        prod_frame = prod << prod_shift;
    } else {
        const int rshift = static_cast<int>(-prod_shift);
        if (rshift >= 128) {
            prod_frame = (prod != 0) ? (__uint128_t)1U : (__uint128_t)0U;
        } else {
            const __uint128_t mask = (((__uint128_t)1) << rshift) - 1;
            const bool lost = (prod & mask) != 0;
            prod_frame = prod >> rshift;
            if (lost)
                prod_frame |= (__uint128_t)1U;
        }
    }

    const int64_t c_shift = (expc - 52) - E;
    __uint128_t c_frame;
    if (c_shift >= 0) {
        c_frame = ((__uint128_t)mc) << c_shift;
    } else {
        const int rshift = static_cast<int>(-c_shift);
        if (rshift >= 128) {
            c_frame = (mc != 0) ? (__uint128_t)1U : (__uint128_t)0U;
        } else {
            const __uint128_t mask = (((__uint128_t)1) << rshift) - 1;
            const bool lost = ((__uint128_t)mc & mask) != 0;
            c_frame = ((__uint128_t)mc) >> rshift;
            if (lost)
                c_frame |= (__uint128_t)1U;
        }
    }

    __uint128_t mag;
    uint32_t result_sign;
    if (sign_ab == sc) {
        mag = prod_frame + c_frame;
        result_sign = sign_ab;
    } else {
        if (prod_frame >= c_frame) {
            mag = prod_frame - c_frame;
            result_sign = sign_ab;
        } else {
            mag = c_frame - prod_frame;
            result_sign = sc;
        }
        if (mag == 0) {
            return make_signed_zero(0u); // RNE: exact cancellation -> +0.
        }
    }

    return arith_fma_round_and_pack_rne(result_sign, mag, E, fe);
}

// Negation (sign-bit flip). Mirrors sf64_neg but inlines across TUs.
SF64_ALWAYS_INLINE double sf64_internal_neg(double a) noexcept {
    return from_bits(bits_of(a) ^ kSignMask);
}

} // namespace soft_fp64::internal
