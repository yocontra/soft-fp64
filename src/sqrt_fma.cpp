// Soft-fp64 sqrt / rsqrt / fma.
//
// Reference: Mesa `src/compiler/glsl/float64.glsl` — __fsqrt64, __ffma64.
//
// Implementation choices (deviating from the Newton-Raphson sketch in the
// spec in favour of bit-exactness by construction):
//
// * sqrt — integer digit-by-digit square-root extraction on the 53-bit
//   mantissa. 53 compare-subtract-shift iterations produce a 53-bit result
//   plus a remainder, which we use to round to nearest even. No Newton-
//   Raphson refinement, no reliance on soft-fp64 add/mul at runtime — sqrt
//   is self-contained on top of the bit helpers.
//
// * fma — exact 53×53 → 106-bit product through __uint128_t, align and
//   add/subtract the 53-bit `c` mantissa, then normalise + round to 53
//   bits. Again no host-FPU arithmetic and no runtime dependency on the
//   arithmetic TU.
//
// * rsqrt — 1/sqrt(x) via sf64_div. This composes a correctly-rounded div
//   with a bit-exact sqrt; the 1-ULP test tolerance covers the single
//   rounding step.
//
// Constraint: no `+`, `-`, `*`, `/` on a host `double` lvalue anywhere in
// these bodies. Integer math only.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "internal_arith.h"
#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

#include <cstdint>

using namespace soft_fp64::internal;

namespace {

// ---- digit-by-digit integer sqrt on a 64-bit input -----------------------
//
// Given `radicand` with its MSB in a known bit position, computes the
// largest integer `root` such that root*root <= radicand, returning both
// `root` and the final remainder `radicand - root*root`.
//
// This is the classical "paper-and-pencil" binary sqrt algorithm:
// for each bit position i from hi to 0, try setting that bit in the root
// candidate; if candidate*candidate <= radicand, keep the bit.
//
// Iterating `num_iters` times produces `num_iters` bits of the root.
//
// All arithmetic is on unsigned 64/128-bit integers.
struct IntSqrtResult {
    uint64_t root;
    uint64_t remainder;
    bool remainder_nonzero; // sticky across iterations we don't track
};

SF64_ALWAYS_INLINE IntSqrtResult isqrt_bits(uint64_t radicand_hi, uint64_t radicand_lo,
                                            int num_result_bits) noexcept {
    // radicand is viewed as a 128-bit number (hi:lo). We extract two bits
    // at a time from the MSB end, accumulate into `rem`, and compare
    // `(root << 2) | 1` == 2*root bit pattern extended for next-bit guess.
    //
    // Algorithm (schoolbook binary sqrt):
    //   rem = 0; root = 0;
    //   for i from num_result_bits-1 downto 0:
    //     rem = (rem << 2) | next_two_bits_of_radicand
    //     test = (root << 2) | 1
    //     if rem >= test: rem -= test; root = (root << 1) | 1
    //     else:           root = root << 1
    //
    // We represent `rem` in up to 128 bits because at each step it can grow
    // to ~ (2*root+1) <= 2*(2^num_result_bits).
    __uint128_t rem = 0;
    __uint128_t root = 0;
    // The radicand is (radicand_hi, radicand_lo); we consume from the top
    // two bits each iteration. Number of radicand bit-pairs we traverse is
    // `num_result_bits`. The top pair must align to bit positions
    // (2*num_result_bits - 1, 2*num_result_bits - 2) of the radicand.

    // Build a 128-bit view. We shift it left so the topmost meaningful pair
    // ends up at bits (127, 126), then consume with (rad >> 126) & 3 each
    // iteration and shift rad left by 2.
    __uint128_t rad = ((__uint128_t)radicand_hi << 64) | (__uint128_t)radicand_lo;
    const int total_radicand_bits = num_result_bits * 2;
    // If total_radicand_bits < 128, shift rad left to align its MSB pair to
    // bit 127..126. If total_radicand_bits == 128 no shift needed.
    if (total_radicand_bits < 128) {
        rad = rad << (128 - total_radicand_bits);
    }

    for (int i = 0; i < num_result_bits; ++i) {
        // Pull the next two bits off the top of rad.
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

    IntSqrtResult r;
    r.root = static_cast<uint64_t>(root);
    r.remainder = static_cast<uint64_t>(rem);
    r.remainder_nonzero = rem != 0;
    return r;
}

} // namespace

// ---- sqrt ----------------------------------------------------------------

namespace {

SF64_ALWAYS_INLINE double sqrt_r_impl(double x, sf64_rounding_mode mode,
                                      sf64_internal_fe_acc& fe) noexcept {
    // SAFETY: __builtin_bit_cast punning double <-> uint64 is the documented
    // IEEE-754 field-access technique; no UB because both types have the
    // same size and no trap representations.
    const uint64_t bx = bits_of(x);
    const uint32_t sign = extract_sign(bx);
    const uint32_t exp_biased = extract_exp(bx);
    const uint64_t frac = extract_frac(bx);

    // Edge cases.
    if (exp_biased == kExpMax) {
        // Inf or NaN.
        if (frac != 0) {
            // NaN -> quieted NaN (propagate payload, set quiet bit).
            // SAFETY: bitwise OR on uint64, bit-cast back to double — no FPU.
            return from_bits(bx | kQuietNaNBit);
        }
        // Infinity.
        if (sign != 0) {
            // sqrt(-inf) = NaN (invalid).
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        // sqrt(+inf) = +inf.
        return x;
    }
    if (is_zero_bits(bx)) {
        // sqrt(+0) = +0, sqrt(-0) = -0. Preserve sign.
        return x;
    }
    if (sign != 0) {
        // Negative finite -> NaN (sqrt of negative is invalid).
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }

    // Positive finite, non-zero. Normalise subnormals.
    int64_t unbiased_exp; // actual (unbiased) exponent of `x`
    uint64_t mantissa;    // 53-bit significand, MSB at bit 52
    if (exp_biased == 0) {
        // Subnormal: frac != 0 (zero case handled above). Shift so the top
        // bit of frac moves to bit 52.
        // SAFETY: clz64 is __builtin_clzll, which is UB only on zero —
        // we've already excluded zero via is_zero_bits.
        const int shift = clz64(frac) - (63 - kFracBits);
        mantissa = frac << shift;
        // Effective exponent for a subnormal is 1 - bias - shift.
        unbiased_exp = static_cast<int64_t>(1 - kExpBias) - static_cast<int64_t>(shift);
    } else {
        mantissa = frac | kImplicitBit; // bit 52 set
        unbiased_exp = static_cast<int64_t>(exp_biased) - static_cast<int64_t>(kExpBias);
    }

    // We want to compute sqrt(m * 2^unbiased_exp).
    //
    // If unbiased_exp is even:   sqrt = sqrt(m)       * 2^(unbiased_exp/2)
    // If unbiased_exp is odd:    sqrt = sqrt(m*2)     * 2^((unbiased_exp-1)/2)
    //
    // We scale `mantissa` so the integer sqrt produces a 54-bit result (top
    // bit at position 53), then we can round to 53 bits.
    //
    // Starting mantissa has MSB at bit 52 (value in [2^52, 2^53)).
    // To get a 54-bit sqrt, we need radicand with 106 or 108 bits of
    // precision (2 bits per output bit). We shift mantissa left into a
    // 128-bit value. The low bits are zero.
    //
    // Choose shift so that the radicand has:
    //   - MSB aligned to bit 107 or 106 depending on parity
    //   - enough low zeroes that the sqrt result has 54 bits
    //
    // Concretely: start with radicand = mantissa << K, where K is chosen so
    // the effective radicand bit-length is 2 * 54 = 108 (for even exp) or
    // 2 * 54 - 1 + 1 = 108 bits but the high bit pattern is shifted one
    // position (for odd exp). Let's define:
    //
    //   if exp_adj is even:    radicand = mantissa << (108 - 53) = mantissa << 55
    //     => radicand MSB at bit 52+55 = 107 (high bit of 108-bit number)
    //   if exp_adj is odd:     radicand = mantissa << (108 - 53 + 1) = mantissa << 56
    //     => radicand MSB at bit 108 (one extra factor of 2 consumed by sqrt)
    //
    // Wait — for odd unbiased_exp we want sqrt(m * 2) which is equivalent
    // to sqrting a radicand that's mantissa * 2. So we shift by one more.
    //
    // Let's re-derive cleanly. After the sqrt we want a 54-bit root. The
    // integer sqrt of a 2N-bit radicand is N bits. We need N = 54, so
    // radicand must be exactly 108 bits with its MSB ideally at bit 107.
    //
    // For even unbiased_exp:
    //   new_exp = unbiased_exp / 2
    //   radicand = mantissa * 2^55           (mantissa has MSB at 52; result MSB at 107)
    //   sqrt_mantissa_54b = isqrt(radicand)  (54-bit root; MSB at bit 53)
    //
    // For odd unbiased_exp:
    //   new_exp = (unbiased_exp - 1) / 2
    //   radicand = mantissa * 2^56           (mantissa has MSB at 52; result MSB at 108)
    //   sqrt_mantissa_54b = isqrt(radicand)  (still ~54 bits)
    //
    // In both cases the resulting root has its top bit at either position
    // 53 (even) or position 53 (odd, because one of the input bits went to
    // position 108 means the sqrt has MSB at 54). Let's verify:
    //   Even case: radicand in [2^107, 2^108) so sqrt in [2^53.5, 2^54) — 54 bits, MSB at 53.
    //   Odd case:  radicand in [2^108, 2^109) so sqrt in [2^54, 2^54.5) — 55 bits, MSB at 54.
    //
    // So for odd we get an extra bit. We handle that by asking isqrt_bits
    // for 55 bits in the odd case and normalising.

    const bool exp_odd = (unbiased_exp & 1) != 0;
    // Output unbiased exponent: floor(unbiased_exp / 2). Integer-divide
    // rounds toward zero, not down, so handle negative odd values manually.
    int64_t new_unbiased_exp;
    if (exp_odd) {
        // floor(n/2) for odd n: (n - 1) / 2 works for both signs because
        // (odd_n - 1) is even and even/2 is exact.
        new_unbiased_exp = (unbiased_exp - 1) / 2;
    } else {
        new_unbiased_exp = unbiased_exp / 2;
    }

    // Derive the integer sqrt radicand. Let m be the extracted 53-bit
    // mantissa (MSB at bit 52, so m in [2^52, 2^53)), and p = unbiased_exp
    // - 52 the true exponent of the value m*2^p.
    //
    //   sqrt(m * 2^p) = sqrt(m * 2^(p - p%2)) * 2^(p%2 / 2)   -- conceptually
    //
    // Pick a shift K such that K + (p%2 term) leaves an integer exponent
    // and the radicand has ~108 bits. Then isqrt produces a 54-bit root
    // (MSB at bit 53).
    //
    //   even p: R = m << 54, MSB at bit 106, R in [2^106, 2^107), sqrt in [2^53, 2^53.5)
    //   odd  p: R = m << 55, MSB at bit 107, R in [2^107, 2^108), sqrt in [2^53.5, 2^54)
    //
    // In both cases root has MSB at bit 53, so shift-right-by-1 to get the
    // 53-bit output mantissa with MSB at bit 52. The shifted-out bit is the
    // round bit; the remainder of the integer sqrt supplies the sticky.
    const int shift = exp_odd ? 55 : 54;
    constexpr int kRootBits = 54;

    // radicand = mantissa << shift, held in (radicand_hi, radicand_lo).
    uint64_t radicand_hi;
    uint64_t radicand_lo;
    if (shift >= 64) {
        radicand_hi = mantissa << (shift - 64);
        radicand_lo = 0;
    } else {
        // `shift` in [54, 55] here; 64 - shift in [9, 10] so no UB.
        radicand_hi = mantissa >> (64 - shift);
        radicand_lo = mantissa << shift;
    }

    IntSqrtResult isr = isqrt_bits(radicand_hi, radicand_lo, kRootBits);
    const uint64_t root = isr.root;
    const bool remainder_nonzero = isr.remainder_nonzero;

    // Root has MSB at bit 53 (54 bits). Shift right by 1 to get the 53-bit
    // mantissa. The dropped bit is the round bit; the integer sqrt
    // remainder is the sticky.
    const uint64_t guard_bit = root & 1u;
    const uint64_t root53 = root >> 1;
    const bool sticky = remainder_nonzero;

    // sqrt is always positive (sign = 0). Route through the shared mode-
    // parametrized decision so RTZ/RUP/RDN/RNA propagate.
    uint64_t rounded = root53;
    if (sf64_internal_should_round_up(/*sign=*/0u, guard_bit != 0, sticky, (root53 & 1u) != 0,
                                      mode)) {
        rounded = root53 + 1u;
    }

    // INEXACT if the integer-sqrt dropped a nonzero guard or left a nonzero
    // remainder. sqrt of a finite positive never underflows to subnormal
    // (smallest input is 2^-1074 → sqrt ~ 2^-537, well inside normal range)
    // and never overflows (largest input ~ 2^1024 → sqrt ~ 2^512). The
    // clamps below stay as defence-in-depth but don't need fenv raises.
    if (guard_bit != 0 || sticky) {
        fe.raise(SF64_FE_INEXACT);
    }

    // `rounded` now has 53 bits, MSB at bit 52 — except if rounding caused
    // a carry into bit 53, in which case we need to shift right by 1 and
    // bump the exponent.
    int64_t biased_new_exp = new_unbiased_exp + static_cast<int64_t>(kExpBias);
    if ((rounded & (uint64_t{1} << 53)) != 0) {
        rounded >>= 1;
        biased_new_exp += 1;
    }

    // Clamp (shouldn't happen for valid inputs but be defensive).
    if (biased_new_exp >= static_cast<int64_t>(kExpMax)) {
        return from_bits(kPositiveInf);
    }
    if (biased_new_exp <= 0) {
        // Should not occur for sqrt of any finite positive double; the
        // smallest subnormal sqrt(2^-1074) ~ 2^-537 is well within normal
        // range.
        return from_bits(0);
    }

    const uint64_t out_frac = rounded & kFracMask;
    return from_bits(pack(0u, static_cast<uint32_t>(biased_new_exp), out_frac));
}

} // namespace

extern "C" double sf64_sqrt(double x) {
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_sqrt_rne(x, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_sqrt_r(sf64_rounding_mode mode, double x) {
    sf64_internal_fe_acc fe;
    const double r = sqrt_r_impl(x, mode, fe);
    fe.flush();
    return r;
}

// ---- rsqrt ---------------------------------------------------------------

extern "C" double sf64_rsqrt(double x) {
    // rsqrt(x) = 1 / sqrt(x). Composition of bit-exact sqrt and correctly-
    // rounded div yields <= 1 ULP error vs host 1/sqrt(x), which is the
    // tolerance mandated by the spec.
    const double sq = sf64_sqrt(x);
    // SAFETY: `bits_of` bit-cast of the literal 1.0 just yields the IEEE
    // encoding of one (0x3FF0000000000000). No host-FPU arithmetic here.
    const double one = from_bits(0x3FF0000000000000ULL);
    return sf64_div(one, sq);
}

// ---- fma -----------------------------------------------------------------

namespace {

// Position of the highest set bit in a 128-bit value (0-indexed). Returns
// -1 for zero.
SF64_ALWAYS_INLINE int highest_bit_u128(__uint128_t v) noexcept {
    if (v == 0)
        return -1;
    const uint64_t hi = static_cast<uint64_t>(v >> 64);
    if (hi != 0) {
        // SAFETY: __builtin_clzll is UB on zero — guarded above.
        return 127 - __builtin_clzll(hi);
    }
    const uint64_t lo = static_cast<uint64_t>(v);
    // SAFETY: lo != 0 here since v != 0 and hi == 0.
    return 63 - __builtin_clzll(lo);
}

// Mode-parametrized round-and-pack for a (mag, frame_exp) representation.
// Value = mag * 2^frame_exp. Handles normals, subnormals, and overflow in a
// single rounding step (no double rounding). `fe` accumulates INEXACT /
// OVERFLOW / UNDERFLOW; the caller flushes once to TLS at end-of-op.
SF64_ALWAYS_INLINE double round_and_pack(uint32_t sign, __uint128_t mag, int64_t frame_exp,
                                         sf64_rounding_mode mode,
                                         sf64_internal_fe_acc& fe) noexcept {
    if (mag == 0) {
        return make_signed_zero(sign);
    }
    const int msb = highest_bit_u128(mag); // position of top set bit
    // Unbiased exponent if this were a normal with MSB at bit 52 of the
    // mantissa: unbiased = frame_exp + msb. Biased = unbiased + kExpBias.
    const int64_t unbiased = frame_exp + static_cast<int64_t>(msb);
    int64_t biased = unbiased + static_cast<int64_t>(kExpBias);

    // Overflow to infinity. Directed modes round to max-finite in the
    // direction that truncates away from infinity.
    if (biased >= static_cast<int64_t>(kExpMax)) {
        fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
        switch (mode) {
        case SF64_RTZ:
            return from_bits(pack(sign, kExpMax - 1, kFracMask));
        case SF64_RUP:
            return sign == 0u ? make_signed_inf(0u) : from_bits(pack(1u, kExpMax - 1, kFracMask));
        case SF64_RDN:
            return sign != 0u ? make_signed_inf(1u) : from_bits(pack(0u, kExpMax - 1, kFracMask));
        case SF64_RNE:
        case SF64_RNA:
        default:
            return make_signed_inf(sign);
        }
    }

    // Determine the target LSB position within `mag` for the rounded result.
    // For a normal (biased >= 1), the mantissa is 53 bits with MSB at
    // position msb in `mag`; LSB is at position msb - 52. We round at
    // position (msb - 53): that's the round bit; sticky is everything below.
    //
    // For subnormal (biased <= 0): the true LSB lives at a higher position
    // inside `mag`. The value is `mag * 2^frame_exp`; we want to encode as
    // frac * 2^(-1074), so frac = mag * 2^(frame_exp + 1074). LSB of frac
    // sits at bit position -(frame_exp + 1074) in `mag`... but only when
    // that position is >= 0. If it's negative, the value is smaller than a
    // subnormal LSB and rounds to zero (with sign).
    //
    // Easier formulation: define `target_lsb_pos` = bit position in `mag`
    // of the result's LSB. Round at target_lsb_pos - 1 (round bit),
    // sticky = all bits below round bit.
    //
    // For normal (biased >= 1):      target_lsb_pos = msb - 52
    // For subnormal (biased <= 0):  we want the output to represent a value
    //   with LSB at true exponent -1074. The LSB of `mag` is at true
    //   exponent frame_exp. So position of output LSB in `mag` is
    //   (-1074 - frame_exp).

    int target_lsb;
    bool output_subnormal;
    if (biased >= 1) {
        target_lsb = msb - 52;
        output_subnormal = false;
    } else {
        target_lsb = static_cast<int>(-1074 - frame_exp);
        output_subnormal = true;
    }

    // Track whether the round step dropped any nonzero guard/sticky bits
    // so we can raise INEXACT / UNDERFLOW correctly at the end.
    bool round_inexact = false;

    uint64_t rounded_mant; // for normal: 53-bit with MSB at 52 (implicit)
                           // for subnormal: up to 53-bit frac field
    if (target_lsb <= 0) {
        // All bits of mag are kept (exact representation).
        rounded_mant = static_cast<uint64_t>(mag);
        if (target_lsb < 0) {
            // Shouldn't happen for normals/subnormals — frame_exp set too
            // low. Left-shift to align.
            rounded_mant = rounded_mant << (-target_lsb);
        }
    } else if (target_lsb >= 128) {
        // Rounding position is at or below the entire 128-bit frame — the
        // true value is smaller than half an ULP of the smallest subnormal.
        // Under RNE/RTZ/RNA this rounds to ±0. Under RUP a positive mag
        // bumps to the smallest positive subnormal; under RDN a negative
        // mag bumps to the smallest negative subnormal.
        if (mag != 0) {
            // Tiny-before-rounding and inexact by construction.
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
            if ((mode == SF64_RUP && sign == 0u) || (mode == SF64_RDN && sign != 0u)) {
                return from_bits(pack(sign, 0u, 1u));
            }
        }
        return make_signed_zero(sign);
    } else {
        // Need to round. target_lsb in [1, 127].
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
        if (sf64_internal_should_round_up(sign, round_bit, sticky, lsb, mode)) {
            rounded_mant += 1u;
        }
    }

    if (output_subnormal) {
        // IEEE 754 §7.5 (tiny-before-rounding flavour): subnormal output
        // with any rounding loss raises UNDERFLOW+INEXACT. Rounding up into
        // the smallest normal still counts as an underflow event — the
        // pre-round value was tiny.
        if (round_inexact) {
            fe.raise(SF64_FE_UNDERFLOW | SF64_FE_INEXACT);
        }
        // Check if rounding pushed us into the normal range.
        if ((rounded_mant & kImplicitBit) != 0) {
            // Became the smallest normal (exp_biased = 1, frac = 0 or the
            // low 52 bits if we went further).
            return from_bits(pack(sign, 1u, rounded_mant & kFracMask));
        }
        return from_bits(pack(sign, 0u, rounded_mant));
    }

    if (round_inexact) {
        fe.raise(SF64_FE_INEXACT);
    }

    // Normal. rounded_mant has 53 bits (MSB at 52) or 54 bits if rounding
    // caused a carry — in which case shift right and bump exponent.
    if ((rounded_mant & (uint64_t{1} << 53)) != 0) {
        rounded_mant >>= 1;
        biased += 1;
        if (biased >= static_cast<int64_t>(kExpMax)) {
            fe.raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
            switch (mode) {
            case SF64_RTZ:
                return from_bits(pack(sign, kExpMax - 1, kFracMask));
            case SF64_RUP:
                return sign == 0u ? make_signed_inf(0u)
                                  : from_bits(pack(1u, kExpMax - 1, kFracMask));
            case SF64_RDN:
                return sign != 0u ? make_signed_inf(1u)
                                  : from_bits(pack(0u, kExpMax - 1, kFracMask));
            case SF64_RNE:
            case SF64_RNA:
            default:
                return make_signed_inf(sign);
            }
        }
    }
    return from_bits(pack(sign, static_cast<uint32_t>(biased), rounded_mant & kFracMask));
}

SF64_ALWAYS_INLINE double fma_r_impl(double a, double b, double c, sf64_rounding_mode mode,
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

    // IEEE §7.2: the 0×∞ sub-operation is invalid regardless of c. Raise
    // INVALID before propagating NaN so that fma(0, ∞, qNaN) signals it.
    if ((a_inf && b_zero) || (a_zero && b_inf)) {
        fe.raise(SF64_FE_INVALID);
        return canonical_nan();
    }

    // NaN propagation.
    if (a_nan || b_nan || c_nan) {
        // IEEE 754: fma(inf, 0, qNaN) returns qNaN (even though inf*0 would
        // be invalid). We return a quieted NaN from the first NaN operand.
        if (a_nan)
            return from_bits(ba | kQuietNaNBit);
        if (b_nan)
            return from_bits(bb | kQuietNaNBit);
        return from_bits(bc | kQuietNaNBit);
    }

    // If a*b is inf:
    if (a_inf || b_inf) {
        // a*b = signed inf.
        if (c_inf) {
            // inf + inf = inf if signs match; inf - inf = NaN (invalid).
            if (sign_ab == sc)
                return make_signed_inf(sign_ab);
            fe.raise(SF64_FE_INVALID);
            return canonical_nan();
        }
        return make_signed_inf(sign_ab);
    }

    // If c is inf (and a*b is finite), result is c.
    if (c_inf) {
        return make_signed_inf(sc);
    }

    // Handle a*b == 0 separately: result is c, but sign handling for the
    // 0 + 0 case follows IEEE 754-2008 §6.3 — opposite-sign zero sum is
    // -0 under RDN and +0 under all other modes.
    if (a_zero || b_zero) {
        if (is_zero_bits(bc)) {
            if (sign_ab == sc) {
                // Like signs: result keeps that sign.
                return make_signed_zero(sign_ab);
            }
            // Opposite signs: +0 under RNE/RTZ/RUP/RNA, -0 under RDN.
            return make_signed_zero(mode == SF64_RDN ? 1u : 0u);
        }
        return c;
    }

    // Now a, b, c are finite; a*b is finite and nonzero. c may be zero.

    // Extract 53-bit mantissas (with implicit bit for normals) and true
    // exponents. Normalise subnormals.
    auto normalise = [](uint32_t exp_biased, uint64_t frac, uint64_t& mant_out, int64_t& exp_out) {
        if (exp_biased == 0) {
            // Subnormal (we've already ruled out zero for a and b).
            // SAFETY: clz64 UB guard already applied (frac != 0 for subnormals
            // reaching here).
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

    // Product: ma (53 bits, MSB at 52) * mb (53 bits, MSB at 52) = up to
    // 106-bit result with MSB at bit 104 or 105.
    // SAFETY: __uint128_t multiplication on two 53-bit values never overflows
    // 128 bits; it fits comfortably in 106.
    const __uint128_t prod = (__uint128_t)ma * (__uint128_t)mb;
    // `prod` has MSB at bit 104 or 105 depending on whether (ma*mb)'s high
    // bit carried. Compute its true exponent.
    // Exponent of the integer product: expa + expb - 104 (if MSB at 104)
    //                                  expa + expb - 105 (if MSB at 105)
    // It's easier to track: treat `prod` as an integer magnitude, and the
    // product's unbiased exponent is (expa - 52) + (expb - 52) = expa+expb-104
    // in "integer" form (i.e. product_value = prod * 2^(expa+expb-104)).
    int64_t prod_exp = expa + expb - 104; // value = prod * 2^prod_exp

    // Handle c.
    if (is_zero_bits(bc)) {
        // Result is just the signed product; defer all rounding to the
        // single-rounding helper.
        return round_and_pack(sign_ab, prod, prod_exp, mode, fe);
    }

    // c is nonzero, finite. Normalise it.
    uint64_t mc;
    int64_t expc;
    normalise(ec, fc, mc, expc);

    // Two values to combine:
    //   prod_value = prod * 2^prod_exp         (prod up to 106 bits, MSB at 104/105)
    //   c_value    = mc   * 2^(expc - 52)      (mc 53 bits, MSB at bit 52)
    //
    // Choose a common "frame exponent" E such that each value is laid into
    // a 128-bit word as `(value / 2^E)`. The dominant operand's MSB should
    // sit near bit 127 (leaving room for a carry) and the subordinate is
    // right-shifted with jamming. We want at least ~55 bits of precision
    // retained for rounding.
    //
    // Dominant operand determined by comparing each value's "true MSB
    // position" (unbiased exponent):
    //   prod_true_msb = prod_exp + msb_of(prod)  (msb_of in [104, 105])
    //   c_true_msb    = expc                     (mc MSB at 52; value MSB at expc)
    //
    // Set frame bit 127 to the max of those + some headroom for carry.
    const int prod_msb_local = highest_bit_u128(prod); // 104 or 105
    const int64_t prod_true_msb = prod_exp + static_cast<int64_t>(prod_msb_local);
    const int64_t c_true_msb = expc;
    // Headroom of 2 bits above the dominant MSB to catch carries.
    const int64_t frame_top = (prod_true_msb > c_true_msb ? prod_true_msb : c_true_msb) + 2;
    // Frame represents bit 127 = 2^frame_top. So E (bottom of frame) =
    // frame_top - 127. A value v is placed at integer position
    // floor(v / 2^E), optionally with jamming of dropped low bits.
    const int64_t E = frame_top - 127;

    // Place prod into frame. prod's integer value * 2^prod_exp = value, so
    // frame_int(prod_value) = prod * 2^(prod_exp - E). Shift amount:
    const int64_t prod_shift = prod_exp - E;
    __uint128_t prod_frame;
    if (prod_shift >= 0) {
        // left shift — guaranteed safe since frame_top was chosen with 2
        // bits of headroom above prod_true_msb, so prod's MSB lands at bit
        // prod_msb_local + prod_shift <= 125.
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

    // Place c into frame. c's integer value (mc) * 2^(expc - 52) = value.
    // frame_int(c_value) = mc * 2^(expc - 52 - E).
    const int64_t c_shift = (expc - 52) - E;
    __uint128_t c_frame;
    if (c_shift >= 0) {
        // left shift — safe because frame_top >= c_true_msb + 2 = expc + 2,
        // so E <= expc - 125, so c_shift = (expc - 52) - E <= 73 < 128,
        // and mc (53 bits) << 73 = MSB at bit 125. Fine.
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
        mag = prod_frame + c_frame; // fits in 128 bits; headroom bit ensures no overflow
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
            // Exact cancellation: +0 under RNE/RTZ/RUP/RNA, -0 under RDN
            // (IEEE 754-2008 §6.3).
            return make_signed_zero(mode == SF64_RDN ? 1u : 0u);
        }
    }

    // After the add/sub, the sticky bit (bit 0 of each operand that was
    // preserved via jamming) has been added/subtracted. For subtract-with-
    // borrow we must re-jam: if the two sticky contributions differed, the
    // lower bits were not actually zero. However: our jamming ORed the
    // sticky into bit 0 of each operand, and the rest of the dropped bits
    // were represented as this single bit. This is the standard "shift-
    // right-jamming" convention — add/sub of two jammed operands still
    // gives a correctly-jammed result (sticky propagates through addition,
    // and for subtraction the convention is conservative: we may mark a
    // cancellation as inexact when it wasn't exactly zero at the dropped
    // positions, which can only move a borderline round-to-even away from
    // 0 to the correct direction).

    // Single rounding step covering normal, subnormal, and overflow.
    return round_and_pack(result_sign, mag, E, mode, fe);
}

} // namespace

extern "C" double sf64_fma(double a, double b, double c) {
    sf64_internal_fe_acc fe;
    const double r = sf64_internal_fma_rne(a, b, c, fe);
    fe.flush();
    return r;
}

extern "C" double sf64_fma_r(sf64_rounding_mode mode, double a, double b, double c) {
    sf64_internal_fe_acc fe;
    const double r = fma_r_impl(a, b, c, mode, fe);
    fe.flush();
    return r;
}
