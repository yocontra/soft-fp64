// Soft-fp64 arithmetic: add / sub / mul / div / rem / neg.
//
// Reference: Mesa `src/compiler/glsl/float64.glsl` — __fadd64, __fsub64,
// __fmul64, __fdiv64, __fmod64. Adapted to use native uint64_t bit ops
// (MSL supports ulong, so Mesa's u32-pair GLSL workarounds are skipped).
//
// Build requirement: compile with -ffp-contract=off (enforced in top-level
// CMakeLists.txt). Without it clang fuses `a*b+c` patterns into
// llvm.fma.f64 and creates a link-time cycle through sqrt_fma.cpp's fma.
//
// Invariant: these bodies never use host-FPU `+`, `-`, `*`, `/` on a
// `double` lvalue — every arithmetic step is an integer bit op on the
// sign/exp/mantissa fields, or a recursion through another
// `sf64_*` symbol.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "internal_arith.h"
#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

#include <cstdint>

using namespace soft_fp64::internal;

namespace {

// ---- internal mantissa layout --------------------------------------------
//
// During add/sub we keep the significand in a 64-bit word shifted left by 9
// bits:
//
//   bit 62       : overflow bit (set when an add/mul carries out)
//   bit 61       : implicit bit when normalised
//   bit 61..9    : 53 mantissa bits (implicit + 52 fraction)
//   bit 8        : round (guard) bit
//   bit 7..0     : sticky bits
//
// Extraction of a normal value: sig = ((frac | kImplicitBit) << 9)  →
//   kImplicitBit is 1<<52, so shift-left-by-9 lands it at bit 61.
//
// Subnormals are normalised to the same canonical layout by left-shifting
// until the implicit bit re-appears, decrementing the working exponent.

// Mode-aware round-and-pack. `exp` is the biased target exponent corresponding
// to the implicit bit sitting at bit 61 of `sig`; `mode` selects the rounding
// attribute (SF64_RNE reproduces the original pre-1.1 behaviour bit-exactly).
// `fe` is the caller's stack-local flag accumulator — we raise into it and the
// caller flushes once to TLS at end of the public op (see internal_fenv.h).
SF64_ALWAYS_INLINE double round_and_pack(uint32_t sign, int32_t exp, uint64_t sig,
                                         sf64_rounding_mode mode,
                                         sf64_internal_fe_acc& fe) noexcept {
    auto overflow_result = [](uint32_t s, sf64_rounding_mode m) -> double {
        // Directed modes: RTZ always returns max-finite; RUP returns +inf
        // for positive / max-finite for negative; RDN is the mirror.
        // RNE / RNA always return signed infinity.
        switch (m) {
        case SF64_RTZ:
            return from_bits(pack(s, 0x7FEu, kFracMask));
        case SF64_RUP:
            return s == 0u ? from_bits(pack(0u, 0x7FFu, 0))
                           : from_bits(pack(1u, 0x7FEu, kFracMask));
        case SF64_RDN:
            return s != 0u ? from_bits(pack(1u, 0x7FFu, 0))
                           : from_bits(pack(0u, 0x7FEu, kFracMask));
        case SF64_RNE:
        case SF64_RNA:
        default:
            return from_bits(pack(s, 0x7FFu, 0));
        }
    };

    if (exp >= 0x7FF) {
        fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
        return overflow_result(sign, mode);
    }

    const bool was_tiny_before_rounding = (exp <= 0);
    if (was_tiny_before_rounding) {
        // Subnormal / below-subnormal: right-shift-with-jamming to align.
        const int shift = 1 - exp;
        sig = shift_right_jamming(sig, shift);
        exp = 0;
    }

    // Round decision: bit 8 is the guard bit, bits 7..0 are sticky, bit 9 is
    // the target LSB (used for RNE's tiebreak-to-even).
    const bool round_bit = ((sig >> 8) & 1ULL) != 0;
    const bool sticky = (sig & 0xFFULL) != 0;
    const bool lsb = ((sig >> 9) & 1ULL) != 0;
    const bool inexact = round_bit || sticky;

    uint64_t rounded = sig;
    if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
        rounded += (1ULL << 9);
    }

    // If rounding carried the implicit bit from bit 61 into bit 62, shift
    // right once and bump exponent.
    if ((rounded >> 62) & 1ULL) {
        // Shift with jamming to preserve the just-created sticky.
        rounded = shift_right_jamming(rounded, 1);
        ++exp;
        if (exp >= 0x7FF) {
            fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
            return overflow_result(sign, mode);
        }
    }

    // Extract final 52 mantissa bits (bits 60..9 of the rounded significand).
    const uint64_t mantissa = (rounded >> 9) & kFracMask;
    const bool implicit_set = ((rounded >> 61) & 1ULL) != 0;

    uint32_t final_exp;
    if (exp == 0) {
        // Subnormal branch — or exact zero, or promoted-to-smallest-normal
        // when rounding set the implicit bit.
        if (implicit_set) {
            final_exp = 1u;
        } else {
            final_exp = 0u;
        }
    } else {
        // Normal path.
        if (!implicit_set && mantissa == 0) {
            final_exp = 0u;
        } else {
            final_exp = static_cast<uint32_t>(exp);
        }
    }

    // IEEE 754 §7.5: underflow = tiny-before-rounding AND inexact (the
    // "before rounding" convention — MIPS / RISC-V default). Tiny-before-
    // rounding is captured by was_tiny_before_rounding; the inexact signal
    // still comes from the guard/sticky of the rounded tail.
    if (inexact) {
        if (was_tiny_before_rounding) {
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        } else {
            fe.raise(SF64_FE_INEXACT);
        }
    }

    return from_bits(pack(sign, final_exp, mantissa));
}

// Normalise a subnormal significand: shift left until the implicit bit
// appears at bit 52. Returns the shift count and updates the significand
// in-place. The caller decrements the working exponent by `shift`.
//
// `frac` has no implicit bit (0 < frac < 2^52). Returns (new_sig, shift)
// with new_sig having the implicit bit at bit 52. Caller then applies the
// ((new_sig) << 9) layout if needed.
struct NormalisedSubnormal {
    uint64_t sig;  // mantissa with implicit bit at bit 52
    int32_t shift; // count of positions shifted left (>= 1)
};

SF64_ALWAYS_INLINE NormalisedSubnormal normalise_subnormal(uint64_t frac) noexcept {
    // SAFETY: caller ensures frac != 0.
    const int lz = __builtin_clzll(frac);
    // We want bit 52 to be the MSB, i.e. bit index 52. Currently MSB is at
    // bit (63 - lz). Shift left by `(63 - lz) - 52 = 11 - lz`? Let's derive
    // properly: shift so that frac << shift has bit 52 set. shift = 52 -
    // (63 - lz) = lz - 11. Wait: frac has MSB at bit (63 - lz). To move MSB
    // to bit 52, shift LEFT by ... no, 63-lz > 52 means we shift RIGHT. But
    // a subnormal has frac < 2^52, so MSB index < 52, hence lz > 11, hence
    // shift = 52 - (63 - lz) = lz - 11 is POSITIVE.
    const int shift = lz - 11;
    return {frac << shift, static_cast<int32_t>(shift)};
}

// Extract (sign, raw_exp, sig_shifted_left_by_9) for either a normal or
// subnormal value. Does NOT handle NaN / inf / zero — caller pre-filters.
//
// For normals: working_exp = raw_exp, sig = (frac | implicit) << 9.
// For subnormals: normalise and adjust working_exp accordingly; final
//   working_exp is (1 - normalise_shift). Mesa keeps the math simple by
//   treating the subnormal as having biased exp 1 and a shifted fraction.
struct Unpacked {
    uint32_t sign;
    int32_t exp;  // working (biased) exponent
    uint64_t sig; // significand shifted left by 9 (implicit bit at bit 62)
};

SF64_ALWAYS_INLINE Unpacked unpack_finite_nonzero(uint64_t bits) noexcept {
    Unpacked u;
    u.sign = extract_sign(bits);
    const uint32_t raw_exp = extract_exp(bits);
    const uint64_t frac = extract_frac(bits);
    if (raw_exp == 0) {
        // Subnormal (frac != 0 guaranteed by caller).
        const auto n = normalise_subnormal(frac);
        // After shifting, n.sig has implicit bit at bit 52. Working exp:
        // biased 1, then decremented by n.shift.
        u.exp = 1 - n.shift;
        u.sig = n.sig << 9;
    } else {
        u.exp = static_cast<int32_t>(raw_exp);
        u.sig = (frac | kImplicitBit) << 9;
    }
    return u;
}

// Add two significands that share the same sign.
SF64_ALWAYS_INLINE double add_magnitudes(uint32_t sign, uint64_t a_bits, uint64_t b_bits,
                                         sf64_rounding_mode mode,
                                         sf64_internal_fe_acc& fe) noexcept {
    const Unpacked a = unpack_finite_nonzero(a_bits);
    const Unpacked b = unpack_finite_nonzero(b_bits);

    // Align exponents by right-shifting-with-jamming the smaller operand.
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
    // Overflow of bit 61 (implicit) into bit 62: shift right with jamming
    // so we don't drop sticky info, and bump exponent.
    if ((sum >> 62) & 1ULL) {
        sum = shift_right_jamming(sum, 1);
        ++exp;
    }

    return round_and_pack(sign, exp, sum, mode, fe);
}

// Subtract magnitudes. `a_bits` and `b_bits` are finite, non-zero. The sign
// follows the operand with the larger magnitude; if magnitudes are exactly
// equal, the result is +0 under RNE / RTZ / RUP / RNA and -0 under RDN
// (IEEE 754-2008 §6.3: exact cancellation produces -0 only for RDN);
// callers pass the "preferred zero sign" for RDN via `zero_sign`.
SF64_ALWAYS_INLINE double sub_magnitudes(uint32_t a_sign, uint64_t a_bits, uint64_t b_bits,
                                         uint32_t zero_sign, sf64_rounding_mode mode,
                                         sf64_internal_fe_acc& fe) noexcept {
    const Unpacked a = unpack_finite_nonzero(a_bits);
    const Unpacked b = unpack_finite_nonzero(b_bits);

    int32_t exp_diff = a.exp - b.exp;
    int32_t exp;
    uint64_t sig_a, sig_b;
    uint32_t sign = a_sign;

    // Invariant after this block: sig_a is the larger-magnitude operand,
    // sig_b has been right-shifted-with-jamming to align. Result sign is
    // the sign of the larger magnitude.
    if (exp_diff > 0) {
        // a has larger exponent → a is larger.
        sig_a = a.sig;
        sig_b = shift_right_jamming(b.sig, exp_diff);
        exp = a.exp;
    } else if (exp_diff < 0) {
        // b has larger exponent → b is larger; flip sign.
        sig_a = b.sig;
        sig_b = shift_right_jamming(a.sig, -exp_diff);
        exp = b.exp;
        sign ^= 1u;
    } else {
        // Equal exponents → compare raw significands.
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
            // Exactly equal magnitudes → zero result.
            return make_signed_zero(zero_sign);
        }
    }

    uint64_t diff = sig_a - sig_b;
    if (diff == 0) {
        return make_signed_zero(zero_sign);
    }

    // Normalise: implicit bit must end up at bit 61.
    // MSB is at bit (63 - clz). Shift left by (61 - (63 - clz)) = clz - 2.
    // diff < 2^62 by construction (both operands had bit 61 set and their
    // difference is strictly smaller), so clz >= 2 → shift >= 0.
    const int lz = __builtin_clzll(diff);
    const int norm_shift = lz - 2;
    diff <<= norm_shift;
    exp -= norm_shift;

    return round_and_pack(sign, exp, diff, mode, fe);
}

// Core multiplication body — finite, non-zero operands. Mode-parametrized.
SF64_ALWAYS_INLINE double mul_impl(uint64_t ab, uint64_t bb, uint32_t r_sign,
                                   sf64_rounding_mode mode, sf64_internal_fe_acc& fe) noexcept {
    // Unpack magnitudes. Significand sits at bit 52 with implicit bit.
    int32_t exp_a, exp_b;
    uint64_t sig_a, sig_b;
    {
        const Unpacked ua = unpack_finite_nonzero(ab);
        const Unpacked ub = unpack_finite_nonzero(bb);
        // unpack_finite_nonzero returns sig << 9; we want sig at bit 52 for
        // the product, so shift back right by 9. (This is exact.)
        sig_a = ua.sig >> 9;
        sig_b = ub.sig >> 9;
        exp_a = ua.exp;
        exp_b = ub.exp;
    }

    // 53 × 53 → up to 106-bit product.
    const __uint128_t product = static_cast<__uint128_t>(sig_a) * static_cast<__uint128_t>(sig_b);
    int32_t exp_r = exp_a + exp_b - kExpBias;

    // Shift into canonical "<<9" layout.
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

    return round_and_pack(r_sign, exp_r, sig, mode, fe);
}

// Core division body — finite, non-zero operands. Mode-parametrized.
SF64_ALWAYS_INLINE double div_impl(uint64_t ab, uint64_t bb, uint32_t r_sign,
                                   sf64_rounding_mode mode, sf64_internal_fe_acc& fe) noexcept {
    const Unpacked ua = unpack_finite_nonzero(ab);
    const Unpacked ub = unpack_finite_nonzero(bb);
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

    return round_and_pack(r_sign, exp_r, sig, mode, fe);
}

} // unnamed namespace

// ---- ABI ------------------------------------------------------------------

extern "C" double sf64_neg(double a) {
    // Sign-bit flip, preserving NaN payload. Never touch the host FPU.
    return from_bits(bits_of(a) ^ kSignMask);
}

// Mode-parametrized add. Public SF64_RNE path wraps this. `fe` is the
// caller's stack-local flag accumulator — inf-inf INVALID and every raise
// from round_and_pack feeds into it; caller flushes once to TLS at return.
static SF64_ALWAYS_INLINE double add_r_impl(double a, double b, sf64_rounding_mode mode,
                                            sf64_internal_fe_acc& fe) noexcept {
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint32_t a_exp = extract_exp(ab);
    const uint32_t b_exp = extract_exp(bb);
    const uint32_t a_sign = extract_sign(ab);
    const uint32_t b_sign = extract_sign(bb);

    // NaN propagation. (sNaN→INVALID wiring is deferred to 1.2 alongside
    // SOFT_FP64_SNAN_PROPAGATE — the plan parks payload preservation there.)
    if (is_nan_bits(ab) || is_nan_bits(bb)) {
        return propagate_nan(ab, bb);
    }

    // Inf handling.
    if (a_exp == kExpMax || b_exp == kExpMax) {
        const bool a_inf = (a_exp == kExpMax);
        const bool b_inf = (b_exp == kExpMax);
        if (a_inf && b_inf) {
            if (a_sign == b_sign)
                return make_signed_inf(a_sign);
            // inf + (-inf) or (-inf) + inf: invalid.
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return a_inf ? make_signed_inf(a_sign) : make_signed_inf(b_sign);
    }

    // Zero handling.
    const bool a_zero = is_zero_bits(ab);
    const bool b_zero = is_zero_bits(bb);
    if (a_zero && b_zero) {
        // +0 + +0 = +0;  -0 + -0 = -0;  +0 + -0 tie under RDN → -0 per IEEE.
        if (a_sign == b_sign)
            return make_signed_zero(a_sign);
        return make_signed_zero(mode == SF64_RDN ? 1u : 0u);
    }
    if (a_zero)
        return b;
    if (b_zero)
        return a;

    // Both finite, non-zero.
    if (a_sign == b_sign) {
        return add_magnitudes(a_sign, ab, bb, mode, fe);
    }
    // Differing signs → subtraction. Preferred zero sign for exact
    // cancellation is +0 under RNE/RTZ/RUP/RNA and -0 under RDN.
    const uint32_t zero_sign = (mode == SF64_RDN) ? 1u : 0u;
    return sub_magnitudes(a_sign, ab, bb, zero_sign, mode, fe);
}

extern "C" double sf64_add(double a, double b) {
    // RNE specialization — inline body via src/internal_arith.h. The
    // mode-parametrized add_r_impl stays available for sf64_add_r below.
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_add_rne(a, b, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_add_r(sf64_rounding_mode mode, double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = add_r_impl(a, b, mode, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_sub(double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_sub_rne(a, b, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_sub_r(sf64_rounding_mode mode, double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = add_r_impl(a, sf64_neg(b), mode, fe);
    fe.flush();
    return r;
}

static SF64_ALWAYS_INLINE double mul_r_impl(double a, double b, sf64_rounding_mode mode,
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
            // 0 * inf (either ordering): invalid.
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return make_signed_inf(r_sign);
    }

    if (is_zero_bits(ab) || is_zero_bits(bb))
        return make_signed_zero(r_sign);

    return mul_impl(ab, bb, r_sign, mode, fe);
}

extern "C" double sf64_mul(double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_mul_rne(a, b, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_mul_r(sf64_rounding_mode mode, double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = mul_r_impl(a, b, mode, fe);
    fe.flush();
    return r;
}

static SF64_ALWAYS_INLINE double div_r_impl(double a, double b, sf64_rounding_mode mode,
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
        // inf / inf: invalid.
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
        // 0 / 0: invalid.
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }
    if (b_zero) {
        // finite non-zero / 0: division by zero.
        fe.raise(SF64_FE_DIVBYZERO);
        return make_signed_inf(r_sign);
    }
    if (a_zero)
        return make_signed_zero(r_sign);

    return div_impl(ab, bb, r_sign, mode, fe);
}

extern "C" double sf64_div(double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_div_rne(a, b, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_div_r(sf64_rounding_mode mode, double a, double b) {
    sf64_internal_fe_acc fe;
    const double r = div_r_impl(a, b, mode, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_rem(double a, double b) {
    // IEEE fmod(a, b) = a - b * trunc(a / b).
    //
    // We inline the truncation via bit manipulation: extract exponent of
    // (a/b), if >= 52 the value is already an integer (trunc is a no-op);
    // otherwise mask off fractional bits.
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);

    // NaN.
    if (is_nan_bits(ab) || is_nan_bits(bb)) {
        return propagate_nan(ab, bb);
    }

    const uint32_t a_exp = extract_exp(ab);
    const uint32_t b_exp = extract_exp(bb);

    // Stack-local accumulator; flushed to TLS at each return that might raise.
    // rem is bit-exact on its output (no rounding step), so the only raises
    // are the two INVALID paths below — the cost of the accumulator here is
    // mainly so the mandatory round_and_pack call at the end has something
    // to thread into (even though it's always exact).
    sf64_internal_fe_acc fe;

    // fmod(±inf, y) = NaN;  fmod(x, 0) = NaN;  fmod(x, ±inf) = x when x finite.
    if (a_exp == kExpMax) {
        fe.raise(SF64_FE_INVALID);
        fe.flush();
        return canonical_nan();
    }
    if (is_zero_bits(bb)) {
        fe.raise(SF64_FE_INVALID);
        fe.flush();
        return canonical_nan();
    }
    if (b_exp == kExpMax)
        return a; // finite / inf → a itself
    if (is_zero_bits(ab))
        return a; // preserve signed zero

    // Use iterative subtraction in the spirit of Mesa/SoftFloat:
    //
    //   |a| and |b| as raw mantissa+exp. Scale b up to same magnitude as a
    //   (without exceeding it), subtract, iterate.
    //
    // This is numerically exact and avoids the need to trunc a double.

    const uint32_t a_sign = extract_sign(ab);

    // Work with absolute values as a "significand + exponent" where the
    // significand has the implicit bit at bit 52 (no further shift).
    int32_t exp_a, exp_b;
    uint64_t sig_a, sig_b;
    {
        const Unpacked ua = unpack_finite_nonzero(ab);
        const Unpacked ub = unpack_finite_nonzero(bb);
        sig_a = ua.sig >> 9; // implicit at bit 52
        sig_b = ub.sig >> 9;
        exp_a = ua.exp;
        exp_b = ub.exp;
    }

    // If |a| < |b|, remainder is a itself.
    if (exp_a < exp_b || (exp_a == exp_b && sig_a < sig_b)) {
        return a;
    }

    // Shift b up to match a's exponent: we'll iterate from exp_diff down
    // to zero, each step subtracting b<<k from a when a >= b<<k.
    int32_t k = exp_a - exp_b;

    // Use 64-bit space: sig_a has MSB at bit 52. We can freely shift sig_b
    // left by up to (63 - 52) = 11 without overflow, but k may be larger.
    // So we emulate: at each step compare sig_a with sig_b << (min(k, 0)),
    // actually easier: keep aligning b up and subtracting.
    //
    // Iterative long mod:
    while (k >= 0) {
        if (sig_a >= sig_b) {
            sig_a -= sig_b;
        }
        if (sig_a == 0) {
            // Exact multiple → result is signed zero (sign of a).
            return make_signed_zero(a_sign);
        }
        // Shift remainder left by 1, decrement k.
        if (k != 0) {
            sig_a <<= 1;
            --k;
        } else {
            break;
        }
    }

    // `sig_a` is now the remainder with exponent `exp_b` (we've brought
    // things down to b's exponent). But the remainder may have leading
    // zeros — normalise.
    if (sig_a == 0) {
        return make_signed_zero(a_sign);
    }

    // Normalise: shift left so MSB is at bit 52 (implicit position).
    const int lz = __builtin_clzll(sig_a);
    // MSB currently at bit (63 - lz). Want at bit 52.
    const int shift = (63 - lz) - 52;
    int32_t exp_r;
    uint64_t sig_r;
    if (shift > 0) {
        // MSB too high (shouldn't happen since we kept sig_a <= 2*sig_b <= 2^54,
        // but guard anyway).
        sig_r = sig_a >> shift;
        exp_r = exp_b + shift;
    } else {
        // Shift left to place MSB at bit 52.
        const int lshift = -shift;
        sig_r = sig_a << lshift;
        exp_r = exp_b - lshift;
    }

    // Convert to "shifted-left-by-9" layout for round_and_pack. Round bits
    // are zero since the remainder is exact, so `mode` is irrelevant here.
    const uint64_t sig_packed = sig_r << 9;
    const double r = round_and_pack(a_sign, exp_r, sig_packed, SF64_RNE, fe);
    fe.flush();
    return r;
}
