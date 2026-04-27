// Soft-fp64 conversion: f32/iN/uN <-> f64, full width matrix.
//
// Reference: Mesa `src/compiler/glsl/float64.glsl` — __fp32_to_fp64,
// __fp64_to_fp32, __int_to_fp64, __uint_to_fp64, __fp64_to_int, __fp64_to_uint.
// Width variants (i8/i16/i64/u8/u16/u64) extend trivially from the i32/u32
// cases by sign-extending or zero-extending to 64-bit first.
//
// CRITICAL: Apple6+ fp32 is FTZ (MSL §6.20). `soft_f64_from_f32` must read
// the operand via __builtin_bit_cast(uint32_t, x), NOT through an
// intermediate `float` variable — the latter loses subnormal payload when
// the compiler rematerializes the float value. See tests/test_convert_subnormal.cpp.
//
// f64 -> iN/uN must saturate at the integer type's bounds, not wrap. For NaN
// inputs, return 0 (matches LLVM fptosi.sat/fptoui.sat semantics).
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

#include <climits>
#include <cstdint>

namespace {

using soft_fp64::internal::bits_of;
using soft_fp64::internal::clz32;
using soft_fp64::internal::clz64;
using soft_fp64::internal::extract_exp;
using soft_fp64::internal::extract_frac;
using soft_fp64::internal::extract_sign;
using soft_fp64::internal::from_bits;
using soft_fp64::internal::is_snan_bits;
using soft_fp64::internal::kExpBias;
using soft_fp64::internal::kExpMax;
using soft_fp64::internal::kFracBits;
using soft_fp64::internal::kFracMask;
using soft_fp64::internal::kImplicitBit;
using soft_fp64::internal::kQuietNaNBit;
using soft_fp64::internal::kSignMask;
using soft_fp64::internal::pack;
using soft_fp64::internal::sf64_internal_fe_acc;
using soft_fp64::internal::sf64_internal_should_round_up;

// ---- u64 -> f64 (core integer widening path) ----------------------------
//
// Takes a non-zero 64-bit magnitude, returns the correctly-rounded fp64 bit
// pattern with the caller-supplied sign (directed-mode rounding needs the
// sign at the decision point). `fe` collects INEXACT on the round step; the
// caller flushes once at return.
SF64_ALWAYS_INLINE uint64_t u64_magnitude_to_fp64_bits(uint64_t mag, uint32_t sign,
                                                       sf64_rounding_mode mode,
                                                       sf64_internal_fe_acc& fe) noexcept {
    // Pre: mag != 0.
    //
    // Find the position of the leading 1. After a left-shift by `lz`, the
    // leading 1 sits at bit 63. We want it at bit 52 (the implicit bit),
    // so we shift right by (63 - 52) = 11. The bits discarded by the
    // right-shift form the round/sticky payload for round-to-nearest-even.
    const int lz = __builtin_clzll(mag); // SAFETY: mag is non-zero (caller checks).
    // Exponent of the leading 1 in the magnitude (position within a 64-bit word).
    const int msb_pos = 63 - lz; // in [0, 63]
    const uint32_t exp = static_cast<uint32_t>(msb_pos + kExpBias);

    uint64_t frac; // 52 bits, implicit bit excluded
    if (msb_pos <= kFracBits) {
        // Fits exactly — no rounding needed. Shift left so the leading 1
        // lands at bit 52; strip the implicit bit afterwards.
        const int shift = kFracBits - msb_pos;
        frac = (mag << shift) & kFracMask;
        return pack(sign, exp, frac);
    }

    // msb_pos in [53, 63] — we need to round.
    const int shift = msb_pos - kFracBits;  // in [1, 11]
    const uint64_t mantissa = mag >> shift; // 53 bits including implicit
    const uint64_t round_pos = uint64_t{1} << (shift - 1);
    const bool round_bit = (mag & round_pos) != 0;
    const bool sticky = shift >= 2 && (mag & (round_pos - 1u)) != 0;
    const bool lsb = (mantissa & 1u) != 0;

    if (round_bit || sticky) {
        fe.raise(SF64_FE_INEXACT);
    }

    uint64_t rounded = mantissa;
    if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
        rounded += 1u;
    }

    // Rounding can overflow the mantissa (e.g., 0x1FFFFFFFFFFFFFFF rounds up
    // to 0x2000000000000000). If the 54th bit is set, exp += 1, frac = 0.
    if (rounded & (uint64_t{1} << (kFracBits + 1))) {
        // Overflow to next binade — leading implicit bit shifts to bit 53.
        const uint32_t new_exp = exp + 1u;
        return pack(sign, new_exp, 0u);
    }

    const uint64_t frac_out = rounded & kFracMask;
    return pack(sign, exp, frac_out);
}

// ---- signed 64-bit integer -> f64 ---------------------------------------
SF64_ALWAYS_INLINE double i64_to_fp64(int64_t x, sf64_rounding_mode mode,
                                      sf64_internal_fe_acc& fe) noexcept {
    if (x == 0)
        return from_bits(0u);

    const uint64_t raw = static_cast<uint64_t>(x);
    const uint32_t sign = static_cast<uint32_t>(raw >> 63);

    // Compute magnitude. INT64_MIN is its own magnitude (0x8000...0000) when
    // interpreted as unsigned — the two's-complement negation is the same
    // bit pattern, so this falls out naturally.
    uint64_t mag;
    if (sign) {
        // Negation: -x = ~x + 1 (wraps for INT64_MIN, giving 0x80000...0 which
        // is the correct magnitude — the fp64 rounding path handles it).
        mag = static_cast<uint64_t>(0) - raw;
    } else {
        mag = raw;
    }

    return from_bits(u64_magnitude_to_fp64_bits(mag, sign, mode, fe));
}

// ---- unsigned 64-bit integer -> f64 -------------------------------------
SF64_ALWAYS_INLINE double u64_to_fp64(uint64_t x, sf64_rounding_mode mode,
                                      sf64_internal_fe_acc& fe) noexcept {
    if (x == 0)
        return from_bits(0u);
    return from_bits(u64_magnitude_to_fp64_bits(x, /*sign=*/0u, mode, fe));
}

// ---- f64 -> u64 magnitude, mode-rounding & saturating -------------------
//
// Returns the magnitude of `x` rounded to integer per `mode`, saturated to
// [0, 2^64 - 1]. Sign is used only for the rounding-mode decision at the
// fractional boundary (directed modes depend on it). NaN sets `is_nan`.
// +inf / finite overflow sets `too_large` and returns UINT64_MAX.
SF64_ALWAYS_INLINE uint64_t fp64_to_u64_magnitude(double x, uint32_t sign, sf64_rounding_mode mode,
                                                  bool* too_large, bool* is_nan,
                                                  sf64_internal_fe_acc& fe) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t exp = extract_exp(b);
    const uint64_t frac = extract_frac(b);

    *too_large = false;
    *is_nan = false;

    if (exp == kExpMax) {
        if (frac != 0) {
            // NaN → int is invalid.
            fe.raise(SF64_FE_INVALID);
            *is_nan = true;
            return 0;
        }
        // ±inf → out-of-range saturation is invalid (IEEE 754 §7.2).
        fe.raise(SF64_FE_INVALID);
        *too_large = true;
        return ~uint64_t{0};
    }

    const int e = static_cast<int>(exp) - kExpBias;
    if (exp == 0 || e < -1) {
        // |x| < 0.5 (including zero, subnormals). Truncates to 0 under
        // RNE/RTZ/RNA; directed modes bump magnitude to 1 only for the sign
        // that points away from zero.
        if (exp == 0 && frac == 0) {
            return 0; // exact zero, no rounding decision
        }
        // Any non-zero input < 0.5 rounds to 0 or ±1 → inexact.
        fe.raise(SF64_FE_INEXACT);
        const bool nonzero = true;
        if ((mode == SF64_RUP && sign == 0u && nonzero) ||
            (mode == SF64_RDN && sign != 0u && nonzero)) {
            return 1u;
        }
        return 0;
    }

    const uint64_t mant = frac | kImplicitBit;

    if (e == -1) {
        // |x| in [0.5, 1). Truncated magnitude is 0; dropped bits are the
        // entire mantissa. round_bit = implicit bit (true), sticky = frac != 0.
        fe.raise(SF64_FE_INEXACT);
        const bool round_bit = true;
        const bool sticky = frac != 0;
        if (sf64_internal_should_round_up(sign, round_bit, sticky, /*lsb=*/false, mode)) {
            return 1u;
        }
        return 0;
    }

    if (e > 63) {
        // Finite value larger than any representable integer in the target
        // range: invalid (per IEEE 754 §7.2 the saturating-convert path
        // signals INVALID; the callers replace the returned magnitude with
        // the type's saturation value).
        fe.raise(SF64_FE_INVALID);
        *too_large = true;
        return ~uint64_t{0};
    }
    if (e >= kFracBits) {
        // Exact (no fraction bits dropped).
        const int shift = e - kFracBits; // in [0, 11]
        return mant << shift;
    }

    // e in [0, 51] — some fraction bits are dropped.
    const int shift = kFracBits - e; // in [1, 52]
    const uint64_t trunc = mant >> shift;
    const uint64_t round_pos = uint64_t{1} << (shift - 1);
    const bool round_bit = (mant & round_pos) != 0;
    const bool sticky = shift >= 2 && (mant & (round_pos - 1u)) != 0;
    const bool lsb = (trunc & 1u) != 0;
    if (round_bit || sticky) {
        fe.raise(SF64_FE_INEXACT);
    }
    uint64_t rounded = trunc;
    if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
        rounded += 1u;
        if (rounded == 0) {
            // Carry past the 64-bit boundary → invalid.
            fe.raise(SF64_FE_INVALID);
            *too_large = true;
            return ~uint64_t{0};
        }
    }
    return rounded;
}

// ---- f64 -> signed integer, saturating to [type_min, type_max] ----------
//
// `imin`/`imax` are the destination type bounds as signed 64-bit values.
SF64_ALWAYS_INLINE int64_t fp64_to_signed(double x, int64_t imin, int64_t imax,
                                          sf64_rounding_mode mode,
                                          sf64_internal_fe_acc& fe) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);

    bool too_large = false;
    bool is_nan = false;
    const uint64_t mag = fp64_to_u64_magnitude(x, sign, mode, &too_large, &is_nan, fe);

    if (is_nan)
        return 0;

    if (too_large) {
        return sign ? imin : imax;
    }

    // mag fits in 64 bits. Compare against the signed destination bounds.
    if (sign) {
        // Negative: value is -mag. Need -mag >= imin  <=>  mag <= -imin.
        // For imin = INT64_MIN, -imin overflows — the magnitude (uint64)
        // cap is 2^63 which is representable as uint64. Special-case that.
        uint64_t neg_cap;
        if (imin == INT64_MIN) {
            neg_cap = (uint64_t{1} << 63);
        } else {
            neg_cap = static_cast<uint64_t>(-imin);
        }
        if (mag > neg_cap) {
            // Saturation past the signed destination floor → invalid.
            fe.raise(SF64_FE_INVALID);
            return imin;
        }
        // Now -mag is representable as int64_t (or equals INT64_MIN for mag == 2^63).
        if (mag == (uint64_t{1} << 63)) {
            // -(2^63) = INT64_MIN; wider types also accept this.
            return INT64_MIN;
        }
        return -static_cast<int64_t>(mag);
    }

    // Positive: value is +mag. Saturate against imax (>= 0).
    const uint64_t pos_cap = static_cast<uint64_t>(imax);
    if (mag > pos_cap) {
        fe.raise(SF64_FE_INVALID);
        return imax;
    }
    return static_cast<int64_t>(mag);
}

// ---- f64 -> unsigned integer, saturating to [0, type_max] ---------------
SF64_ALWAYS_INLINE uint64_t fp64_to_unsigned(double x, uint64_t umax, sf64_rounding_mode mode,
                                             sf64_internal_fe_acc& fe) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);

    bool too_large = false;
    bool is_nan = false;
    const uint64_t mag = fp64_to_u64_magnitude(x, sign, mode, &too_large, &is_nan, fe);

    if (is_nan)
        return 0;

    if (sign) {
        // Any negative (including -inf) saturates to 0 (the minimum of an
        // unsigned destination). A negative value that rounded to a
        // non-zero magnitude is out-of-range for the unsigned type → INVALID.
        if (mag != 0) {
            fe.raise(SF64_FE_INVALID);
        }
        return 0;
    }

    if (too_large)
        return umax;
    if (mag > umax) {
        fe.raise(SF64_FE_INVALID);
        return umax;
    }
    return mag;
}

} // namespace

// -------------------------------------------------------------------------
// f32 <-> f64
// -------------------------------------------------------------------------

// Shared body for sf64_from_f32 / sf64_from_f32_ex. Routes the sNaN→INVALID
// raise through the caller's accumulator so explicit-mode (state-backed)
// callers observe the flag, not just the TLS-backed default.
SF64_ALWAYS_INLINE double from_f32_impl(float x, sf64_internal_fe_acc& fe) noexcept {
    // SAFETY: read the fp32 operand's raw bit pattern without ever
    // materializing a `float` lvalue — on Apple6+ (MSL §6.20) fp32 is
    // flush-to-zero, so a rematerialized float would collapse subnormals
    // before we get to inspect them. __builtin_bit_cast is a pure type pun
    // (no arithmetic), so the integer payload survives.
    const uint32_t b = __builtin_bit_cast(uint32_t, x);

    const uint32_t sign = (b >> 31) & 1u;
    const uint32_t exp32 = (b >> 23) & 0xFFu;
    const uint32_t frac32 = b & 0x7FFFFFu;

    if (exp32 == 0) {
        if (frac32 == 0) {
            // +/-0
            return from_bits(static_cast<uint64_t>(sign) << 63);
        }
        // Subnormal: renormalize. `lz` is the count of leading zeros in the
        // 32-bit word holding the 23-bit fraction (9 of those zeros are
        // always present — they're the sign + exp field). The MSB of the
        // fraction sits at position `p = 31 - lz`, so the unbiased exponent
        // is `p - 149` and the f64 biased exponent is `p - 149 + 1023`
        // = 874 + p = 905 - lz.
        const int lz = clz32(frac32); // SAFETY: frac32 != 0 here.
        const int p = 31 - lz;        // position of leading 1 (0..22)
        const uint32_t f64_exp = static_cast<uint32_t>(905 - lz);
        // Left-shift to put the leading 1 at bit 52.
        const int left = 52 - p; // in [30, 52]
        const uint64_t f64_frac = (static_cast<uint64_t>(frac32) << left) & kFracMask;
        return from_bits(pack(sign, f64_exp, f64_frac));
    }

    if (exp32 == 0xFFu) {
        if (frac32 == 0) {
            // +/-inf
            return from_bits(pack(sign, kExpMax, 0));
        }
        // NaN — preserve payload and force quiet bit.
        // IEEE 754 §6.2 / §7.2: format-conversion of a sNaN raises INVALID.
        // f32 sNaN: bit 22 of the f32 mantissa clear (and frac != 0).
        if ((frac32 & (1u << 22)) == 0u) {
            fe.raise(SF64_FE_INVALID);
        }
        const uint64_t payload = static_cast<uint64_t>(frac32) << 29;
        return from_bits(pack(sign, kExpMax, payload | kQuietNaNBit));
    }

    // Normal: rebias exponent (f32 bias 127 -> f64 bias 1023).
    const uint32_t f64_exp = exp32 + (kExpBias - 127);
    const uint64_t f64_frac = static_cast<uint64_t>(frac32) << 29;
    return from_bits(pack(sign, f64_exp, f64_frac));
}

extern "C" double sf64_from_f32(float x) {
    sf64_internal_fe_acc fe;
    const double r = from_f32_impl(x, fe);
    fe.flush();
    return r;
}

namespace {

// Directed-mode overflow in f32: RTZ → max-finite; RUP → +inf / max-finite
// (neg); RDN → -inf / max-finite (pos); RNE/RNA → ±inf.
SF64_ALWAYS_INLINE uint32_t f32_overflow_bits(uint32_t sign, sf64_rounding_mode mode) noexcept {
    constexpr uint32_t kF32MaxFinite = 0x7F7FFFFFu;
    const uint32_t inf_bits = (sign << 31) | (0xFFu << 23);
    switch (mode) {
    case SF64_RTZ:
        return (sign << 31) | kF32MaxFinite;
    case SF64_RUP:
        return sign == 0u ? inf_bits : (0x80000000u | kF32MaxFinite);
    case SF64_RDN:
        return sign != 0u ? inf_bits : kF32MaxFinite;
    case SF64_RNE:
    case SF64_RNA:
    default:
        return inf_bits;
    }
}

SF64_ALWAYS_INLINE float to_f32_impl(double x, sf64_rounding_mode mode,
                                     sf64_internal_fe_acc& fe) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);
    const uint32_t exp64 = extract_exp(b);
    const uint64_t frac64 = extract_frac(b);

    if (exp64 == kExpMax) {
        if (frac64 == 0) {
            const uint32_t out = (sign << 31) | (0xFFu << 23);
            return __builtin_bit_cast(float, out);
        }
        // IEEE 754 §6.2 / §7.2: format-conversion of a sNaN raises INVALID.
        if (is_snan_bits(b)) {
            fe.raise(SF64_FE_INVALID);
        }
        const uint32_t payload = static_cast<uint32_t>(frac64 >> 29) & 0x7FFFFFu;
        const uint32_t out = (sign << 31) | (0xFFu << 23) | payload | 0x400000u;
        return __builtin_bit_cast(float, out);
    }

    if (exp64 == 0) {
        // Zero or f64 subnormal — all f64 subnormals are well below f32's
        // smallest subnormal, so they flush to +/-0 under RNE/RTZ/RNA. RUP
        // of a tiny positive rounds up to denorm_min; RDN of a tiny negative
        // rounds down to -denorm_min.
        if (frac64 == 0) {
            const uint32_t out = sign << 31;
            return __builtin_bit_cast(float, out);
        }
        // Non-zero f64 subnormal collapsing to an f32 zero or denorm_min
        // is tiny + inexact → UNDERFLOW + INEXACT.
        fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        if ((mode == SF64_RUP && sign == 0u) || (mode == SF64_RDN && sign != 0u)) {
            const uint32_t out = (sign << 31) | 1u;
            return __builtin_bit_cast(float, out);
        }
        const uint32_t out = sign << 31;
        return __builtin_bit_cast(float, out);
    }

    const int unbiased = static_cast<int>(exp64) - kExpBias;
    const int new_exp = unbiased + 127;

    if (new_exp >= 0xFF) {
        fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
        return __builtin_bit_cast(float, f32_overflow_bits(sign, mode));
    }

    const uint64_t mant = frac64 | kImplicitBit;

    if (new_exp > 0) {
        const uint64_t round_pos = uint64_t{1} << 28;
        const uint64_t top24 = mant >> 29;
        const bool round_bit = (mant & round_pos) != 0;
        const bool sticky = (mant & (round_pos - 1u)) != 0;
        const bool lsb = (top24 & 1u) != 0;

        uint64_t rounded = top24;
        if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
            rounded += 1u;
        }

        uint32_t exp_out = static_cast<uint32_t>(new_exp);
        uint32_t frac_out;
        if (rounded & (uint64_t{1} << 24)) {
            exp_out += 1u;
            if (exp_out >= 0xFFu) {
                fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
                return __builtin_bit_cast(float, f32_overflow_bits(sign, mode));
            }
            frac_out = 0;
        } else {
            frac_out = static_cast<uint32_t>(rounded) & 0x7FFFFFu;
        }
        if (round_bit || sticky) {
            fe.raise(SF64_FE_INEXACT);
        }
        const uint32_t out = (sign << 31) | (exp_out << 23) | frac_out;
        return __builtin_bit_cast(float, out);
    }

    // new_exp <= 0 — subnormal or underflow.
    const int shift = 30 - new_exp;

    if (shift > 53) {
        // Entire 53-bit mantissa sits strictly below the guard position;
        // value magnitude is < denorm_min/2. RNE/RTZ/RNA → ±0; RUP/RDN of
        // the appropriate sign bump to ±denorm_min.
        if (mant == 0) {
            const uint32_t out = sign << 31;
            return __builtin_bit_cast(float, out);
        }
        // Tiny input lost entirely → UNDERFLOW + INEXACT.
        fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        if ((mode == SF64_RUP && sign == 0u) || (mode == SF64_RDN && sign != 0u)) {
            const uint32_t out = (sign << 31) | 1u;
            return __builtin_bit_cast(float, out);
        }
        const uint32_t out = sign << 31;
        return __builtin_bit_cast(float, out);
    }

    const uint64_t round_pos = uint64_t{1} << (shift - 1);
    const uint64_t top = mant >> shift;
    const bool round_bit = (mant & round_pos) != 0;
    const bool sticky = shift >= 2 && (mant & (round_pos - 1u)) != 0;
    const bool lsb = (top & 1u) != 0;

    uint64_t rounded = top;
    if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
        rounded += 1u;
    }

    // f32 output path is always tiny-before-rounding (we're in the
    // new_exp<=0 branch), so any rounding loss is UNDERFLOW + INEXACT. The
    // "rounded into smallest normal" branch still counts — IEEE 754 §7.5
    // treats it as an underflow signalling event.
    if (round_bit || sticky) {
        fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
    }

    if (rounded & (uint64_t{1} << 23)) {
        // Rounded up past subnormal range -> smallest normal.
        const uint32_t out = (sign << 31) | (1u << 23);
        return __builtin_bit_cast(float, out);
    }

    const uint32_t frac_out = static_cast<uint32_t>(rounded) & 0x7FFFFFu;
    const uint32_t out = (sign << 31) | frac_out;
    return __builtin_bit_cast(float, out);
}

} // namespace

extern "C" float sf64_to_f32(double x) {
    sf64_internal_fe_acc fe;
    const float r = to_f32_impl(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

extern "C" float sf64_to_f32_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const float r = to_f32_impl(x, mode, fe);
    fe.flush();
    return r;
}

// -------------------------------------------------------------------------
// intN/uintN -> f64 (widen then one common path)
// -------------------------------------------------------------------------

extern "C" double sf64_from_i8(int8_t x) {
    sf64_internal_fe_acc fe;
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i16(int16_t x) {
    sf64_internal_fe_acc fe;
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i32(int32_t x) {
    sf64_internal_fe_acc fe;
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i64(int64_t x) {
    sf64_internal_fe_acc fe;
    const double r = i64_to_fp64(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_from_u8(uint8_t x) {
    sf64_internal_fe_acc fe;
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u16(uint16_t x) {
    sf64_internal_fe_acc fe;
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u32(uint32_t x) {
    sf64_internal_fe_acc fe;
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u64(uint64_t x) {
    sf64_internal_fe_acc fe;
    const double r = u64_to_fp64(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

// -------------------------------------------------------------------------
// f64 -> intN/uintN (saturating, truncating, NaN -> 0)
// -------------------------------------------------------------------------

extern "C" int8_t sf64_to_i8(double x) {
    sf64_internal_fe_acc fe;
    const int8_t r = static_cast<int8_t>(fp64_to_signed(x, INT8_MIN, INT8_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int16_t sf64_to_i16(double x) {
    sf64_internal_fe_acc fe;
    const int16_t r = static_cast<int16_t>(fp64_to_signed(x, INT16_MIN, INT16_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int32_t sf64_to_i32(double x) {
    sf64_internal_fe_acc fe;
    const int32_t r = static_cast<int32_t>(fp64_to_signed(x, INT32_MIN, INT32_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int64_t sf64_to_i64(double x) {
    sf64_internal_fe_acc fe;
    const int64_t r = fp64_to_signed(x, INT64_MIN, INT64_MAX, SF64_RTZ, fe);
    fe.flush();
    return r;
}

extern "C" uint8_t sf64_to_u8(double x) {
    sf64_internal_fe_acc fe;
    const uint8_t r = static_cast<uint8_t>(fp64_to_unsigned(x, UINT8_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint16_t sf64_to_u16(double x) {
    sf64_internal_fe_acc fe;
    const uint16_t r = static_cast<uint16_t>(fp64_to_unsigned(x, UINT16_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint32_t sf64_to_u32(double x) {
    sf64_internal_fe_acc fe;
    const uint32_t r = static_cast<uint32_t>(fp64_to_unsigned(x, UINT32_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint64_t sf64_to_u64(double x) {
    sf64_internal_fe_acc fe;
    const uint64_t r = fp64_to_unsigned(x, UINT64_MAX, SF64_RTZ, fe);
    fe.flush();
    return r;
}

extern "C" int8_t sf64_to_i8_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const int8_t r = static_cast<int8_t>(fp64_to_signed(x, INT8_MIN, INT8_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int16_t sf64_to_i16_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const int16_t r = static_cast<int16_t>(fp64_to_signed(x, INT16_MIN, INT16_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int32_t sf64_to_i32_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const int32_t r = static_cast<int32_t>(fp64_to_signed(x, INT32_MIN, INT32_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int64_t sf64_to_i64_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const int64_t r = fp64_to_signed(x, INT64_MIN, INT64_MAX, mode, fe);
    fe.flush();
    return r;
}

extern "C" uint8_t sf64_to_u8_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const uint8_t r = static_cast<uint8_t>(fp64_to_unsigned(x, UINT8_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint16_t sf64_to_u16_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const uint16_t r = static_cast<uint16_t>(fp64_to_unsigned(x, UINT16_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint32_t sf64_to_u32_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const uint32_t r = static_cast<uint32_t>(fp64_to_unsigned(x, UINT32_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint64_t sf64_to_u64_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const uint64_t r = fp64_to_unsigned(x, UINT64_MAX, mode, fe);
    fe.flush();
    return r;
}

// ---------------------------------------------------------------------------
// Caller-state (`_ex`) convert entries. Bodies are bit-identical to the
// TLS-backed surface above. See src/arithmetic.cpp for design rationale.
// Compiled out under SOFT_FP64_FENV_MODE == 0 (disabled).
// ---------------------------------------------------------------------------

#if SOFT_FP64_FENV_MODE == 1 || SOFT_FP64_FENV_MODE == 2

extern "C" double sf64_from_f32_ex(float x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = from_f32_impl(x, fe);
    fe.flush();
    return r;
}

extern "C" float sf64_to_f32_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const float r = to_f32_impl(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

extern "C" float sf64_to_f32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const float r = to_f32_impl(x, mode, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_from_i8_ex(int8_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i16_ex(int16_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i32_ex(int32_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = i64_to_fp64(static_cast<int64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_i64_ex(int64_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = i64_to_fp64(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_from_u8_ex(uint8_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u16_ex(uint16_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u32_ex(uint32_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = u64_to_fp64(static_cast<uint64_t>(x), SF64_RNE, fe);
    fe.flush();
    return r;
}
extern "C" double sf64_from_u64_ex(uint64_t x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const double r = u64_to_fp64(x, SF64_RNE, fe);
    fe.flush();
    return r;
}

extern "C" int8_t sf64_to_i8_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int8_t r = static_cast<int8_t>(fp64_to_signed(x, INT8_MIN, INT8_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int16_t sf64_to_i16_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int16_t r = static_cast<int16_t>(fp64_to_signed(x, INT16_MIN, INT16_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int32_t sf64_to_i32_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int32_t r = static_cast<int32_t>(fp64_to_signed(x, INT32_MIN, INT32_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" int64_t sf64_to_i64_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int64_t r = fp64_to_signed(x, INT64_MIN, INT64_MAX, SF64_RTZ, fe);
    fe.flush();
    return r;
}

extern "C" uint8_t sf64_to_u8_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint8_t r = static_cast<uint8_t>(fp64_to_unsigned(x, UINT8_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint16_t sf64_to_u16_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint16_t r = static_cast<uint16_t>(fp64_to_unsigned(x, UINT16_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint32_t sf64_to_u32_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint32_t r = static_cast<uint32_t>(fp64_to_unsigned(x, UINT32_MAX, SF64_RTZ, fe));
    fe.flush();
    return r;
}
extern "C" uint64_t sf64_to_u64_ex(double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint64_t r = fp64_to_unsigned(x, UINT64_MAX, SF64_RTZ, fe);
    fe.flush();
    return r;
}

extern "C" int8_t sf64_to_i8_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int8_t r = static_cast<int8_t>(fp64_to_signed(x, INT8_MIN, INT8_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int16_t sf64_to_i16_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int16_t r = static_cast<int16_t>(fp64_to_signed(x, INT16_MIN, INT16_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int32_t sf64_to_i32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int32_t r = static_cast<int32_t>(fp64_to_signed(x, INT32_MIN, INT32_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" int64_t sf64_to_i64_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const int64_t r = fp64_to_signed(x, INT64_MIN, INT64_MAX, mode, fe);
    fe.flush();
    return r;
}

extern "C" uint8_t sf64_to_u8_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint8_t r = static_cast<uint8_t>(fp64_to_unsigned(x, UINT8_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint16_t sf64_to_u16_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint16_t r = static_cast<uint16_t>(fp64_to_unsigned(x, UINT16_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint32_t sf64_to_u32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint32_t r = static_cast<uint32_t>(fp64_to_unsigned(x, UINT32_MAX, mode, fe));
    fe.flush();
    return r;
}
extern "C" uint64_t sf64_to_u64_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state) {
    sf64_internal_fe_acc fe{state};
    const uint64_t r = fp64_to_unsigned(x, UINT64_MAX, mode, fe);
    fe.flush();
    return r;
}

#endif // SOFT_FP64_FENV_MODE == 1 || SOFT_FP64_FENV_MODE == 2
