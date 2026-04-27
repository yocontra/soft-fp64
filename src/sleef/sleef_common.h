#pragma once
//
// Derived from SLEEF 3.6 (Boost-1.0); see src/sleef/NOTICE for the full
// license text, upstream URL, and pinned commit SHA.
//
// Double-double (DD) arithmetic primitives and bit helpers shared across
// the per-family SLEEF translation units. Every routine here performs ALL
// arithmetic through the `sf64_internal_*_rne` inline helpers (see
// src/internal_arith.h) so no host FPU operations are emitted AND the
// 5-way rounding-mode switch at the public sf64_* entry is skipped — the
// SLEEF layer is always round-to-nearest-even.
//
// SPDX-License-Identifier: BSL-1.0 AND MIT
//
// A DD number represents a value as an unevaluated sum `x = hi + lo`, where
// |lo| <= ulp(hi) / 2. This gives us ~106 bits of precision using two IEEE
// binary64 limbs — enough to carry Cody-Waite and Payne-Hanek range-reduced
// arguments without losing low-order bits.
//
// Flag plumbing. Each DD primitive takes an `sf64_internal_fe_acc&` by
// reference. The SLEEF public entries (`sf64_pow`, `sf64_sin`, …) declare
// one accumulator at entry and flush it once to TLS before returning —
// keeps the TLS get-address call off the ~160-call inner loop of
// transcendentals. Under SOFT_FP64_FENV_MODE==0 (disabled) the accumulator
// is empty and the reference threading DCEs.
//

#include "../../include/soft_fp64/defines.h"
#include "../../include/soft_fp64/soft_f64.h"
#include "../internal_arith.h"
#include "../internal_classify.h"
#include "../internal_fenv.h"

#include <cstdint>

namespace soft_fp64::sleef {

// Pull the fe-acc type into the sleef namespace so headers using
// `sf64_internal_fe_acc&` in their signatures don't need to fully-qualify.
using soft_fp64::internal::sf64_internal_fe_acc;

// ---- bit helpers --------------------------------------------------------

SF64_ALWAYS_INLINE uint64_t bits_of(double x) noexcept {
    // SAFETY: double and uint64_t are the same size with no padding; this
    // is a bit-level reinterpret, not a value conversion.
    return __builtin_bit_cast(uint64_t, x);
}

SF64_ALWAYS_INLINE double from_bits(uint64_t b) noexcept {
    // SAFETY: double and uint64_t are the same size with no padding; this
    // is a bit-level reinterpret, not a value conversion.
    return __builtin_bit_cast(double, b);
}

SF64_ALWAYS_INLINE bool isnan_(double x) noexcept {
    const uint64_t b = bits_of(x);
    return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFULL) != 0;
}

// Signaling NaN per IEEE 754-2008 §6.2.1: NaN bits with the most-significant
// fraction bit (bit 51) clear. Used by IEEE-arithmetic public ops in the
// SLEEF TUs (e.g. sf64_remainder) to raise SF64_FE_INVALID before quieting
// the payload, per IEEE 754 §7.2.
SF64_ALWAYS_INLINE bool is_snan_(double x) noexcept {
    const uint64_t b = bits_of(x);
    return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFULL) != 0 &&
           (b & 0x0008000000000000ULL) == 0;
}

SF64_ALWAYS_INLINE bool isinf_(double x) noexcept {
    const uint64_t b = bits_of(x);
    return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFULL) == 0;
}

SF64_ALWAYS_INLINE bool signbit_(double x) noexcept {
    return (bits_of(x) >> 63) != 0;
}

SF64_ALWAYS_INLINE bool is_neg_zero(double x) noexcept {
    return bits_of(x) == 0x8000000000000000ULL;
}

SF64_ALWAYS_INLINE double neg(double x) noexcept {
    return soft_fp64::internal::sf64_internal_neg(x);
}
SF64_ALWAYS_INLINE double abs_(double x) noexcept {
    return soft_fp64::internal::sf64_internal_fabs(x);
}

// Relational predicates — every comparison on a `double` lvalue inside the
// SLEEF layer goes through `sf64_internal_fcmp` (the hidden-visibility
// inline lift of the public `sf64_fcmp` body) so it stays bit-exact on
// targets without native fp64 AND avoids the cross-TU public-ABI call
// frame on every comparison. LLVM FCmpInst::Predicate encoding (see
// include/soft_fp64/soft_f64.h § fcmp predicates): 1=OEQ, 2=OGT, 3=OGE,
// 4=OLT, 5=OLE, 14=UNE. C++'s relational operators map to the ordered
// predicates (NaN on either side ⇒ false); `!=` matches UNE because
// `NaN != x` is true in C++.
SF64_ALWAYS_INLINE bool lt_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 4) != 0;
}
SF64_ALWAYS_INLINE bool le_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 5) != 0;
}
SF64_ALWAYS_INLINE bool gt_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 2) != 0;
}
SF64_ALWAYS_INLINE bool ge_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 3) != 0;
}
SF64_ALWAYS_INLINE bool eq_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 1) != 0;
}
SF64_ALWAYS_INLINE bool ne_(double a, double b) noexcept {
    return soft_fp64::internal::sf64_internal_fcmp(a, b, 14) != 0;
}

// Short-name aliases so call-site syntax matches the old `sf64_add(a,b)`
// shape but routes through the inline RNE helper. Every SLEEF callsite
// threads its local `sf64_internal_fe_acc& fe` through these.
SF64_ALWAYS_INLINE double add_(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_add_rne(a, b, fe);
}
SF64_ALWAYS_INLINE double sub_(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_sub_rne(a, b, fe);
}
SF64_ALWAYS_INLINE double mul_(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_mul_rne(a, b, fe);
}
SF64_ALWAYS_INLINE double div_(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_div_rne(a, b, fe);
}
SF64_ALWAYS_INLINE double fma_(double a, double b, double c, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_fma_rne(a, b, c, fe);
}
SF64_ALWAYS_INLINE double sqrt_(double x, sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::internal::sf64_internal_sqrt_rne(x, fe);
}

// Integer power-of-two multiplier — uses sf64_internal_ldexp so no host
// FPU and no cross-TU call frame.
SF64_ALWAYS_INLINE double pow2i(int q) noexcept {
    // 1.0 * 2^q via the hidden-visibility inline ldexp lift.
    return soft_fp64::internal::sf64_internal_ldexp(1.0, q);
}

// Extract the high 32 mantissa bits (used by Dekker's upper()).
SF64_ALWAYS_INLINE double upper(double d) noexcept {
    // Clear the low 27 mantissa bits so `upper(d) * upper(d)` fits in a
    // double without rounding — Dekker's classic trick.
    // SAFETY: masking bits of a double and reinterpreting; the exponent
    // field is untouched so the value stays in [d/2, 2d] magnitude range.
    const uint64_t mask = 0xFFFFFFFFF8000000ULL; // clear low 27 bits
    return from_bits(bits_of(d) & mask);
}

// Multiply-add: SLEEF `mla(x, y, z) = x*y + z`.
SF64_ALWAYS_INLINE double mla(double x, double y, double z, sf64_internal_fe_acc& fe) noexcept {
    return fma_(x, y, z, fe);
}

// Multiply-subtract: SLEEF `mlapn(x, y, z) = x*y - z`.
SF64_ALWAYS_INLINE double mlapn(double x, double y, double z, sf64_internal_fe_acc& fe) noexcept {
    return fma_(x, y, neg(z), fe);
}

// Integer-valued rounding: floor, via sf64_floor.
SF64_ALWAYS_INLINE double rint_(double x) noexcept {
    return sf64_rint(x);
}

SF64_ALWAYS_INLINE double trunc_(double x) noexcept {
    return soft_fp64::internal::sf64_internal_trunc(x);
}

// ---- Horner polynomial evaluation --------------------------------------
//
// poly_n(x, c[n-1], c[n-2], …, c[0]) evaluates
//     c[n-1] + x * (c[n-2] + x * ( … + x * c[0] ))
// expressed as nested fma_ calls. The caller supplies the coefficients in
// Horner order (highest-degree first).

SF64_ALWAYS_INLINE double poly2(double x, double c1, double c0, sf64_internal_fe_acc& fe) noexcept {
    return mla(x, c1, c0, fe);
}

SF64_ALWAYS_INLINE double poly3(double x, double c2, double c1, double c0,
                                sf64_internal_fe_acc& fe) noexcept {
    return mla(x, mla(x, c2, c1, fe), c0, fe);
}

SF64_ALWAYS_INLINE double poly4(double x, double c3, double c2, double c1, double c0,
                                sf64_internal_fe_acc& fe) noexcept {
    return mla(x, mla(x, mla(x, c3, c2, fe), c1, fe), c0, fe);
}

SF64_ALWAYS_INLINE double poly_array(double x, const double* coeffs, int n,
                                     sf64_internal_fe_acc& fe) noexcept {
    // coeffs[0] is the highest-degree term.
    double acc = coeffs[0];
    for (int i = 1; i < n; ++i) {
        acc = mla(x, acc, coeffs[i], fe);
    }
    return acc;
}

// ---- double-double --------------------------------------------------

struct DD {
    double hi;
    double lo;
};

SF64_ALWAYS_INLINE DD dd(double h, double l) noexcept {
    return DD{h, l};
}

// ddadd2: Knuth's TwoSum with no precondition on |hi| >= |lo|.
SF64_ALWAYS_INLINE DD ddadd2_dd_dd(DD a, DD b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a.hi, b.hi, fe);
    const double bb = sub_(s, a.hi, fe);
    const double err = add_(sub_(a.hi, sub_(s, bb, fe), fe), sub_(b.hi, bb, fe), fe);
    return DD{s, add_(err, add_(a.lo, b.lo, fe), fe)};
}

SF64_ALWAYS_INLINE DD ddadd2_dd_d_d(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a, b, fe);
    const double bb = sub_(s, a, fe);
    const double err = add_(sub_(a, sub_(s, bb, fe), fe), sub_(b, bb, fe), fe);
    return DD{s, err};
}

SF64_ALWAYS_INLINE DD ddadd2_dd_dd_d(DD a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a.hi, b, fe);
    const double bb = sub_(s, a.hi, fe);
    const double err = add_(sub_(a.hi, sub_(s, bb, fe), fe), sub_(b, bb, fe), fe);
    return DD{s, add_(err, a.lo, fe)};
}

SF64_ALWAYS_INLINE DD ddadd2_dd_d_dd(double a, DD b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a, b.hi, fe);
    const double bb = sub_(s, a, fe);
    const double err = add_(sub_(a, sub_(s, bb, fe), fe), sub_(b.hi, bb, fe), fe);
    return DD{s, add_(err, b.lo, fe)};
}

// ddmul with FMA. Uses Dekker's `hi*lo` via one FMA for the correction.
SF64_ALWAYS_INLINE DD ddmul_dd_d_d(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double hi = mul_(a, b, fe);
    const double lo = fma_(a, b, neg(hi), fe);
    return DD{hi, lo};
}

SF64_ALWAYS_INLINE DD ddmul_dd_dd_d(DD a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double hi = mul_(a.hi, b, fe);
    const double lo = fma_(a.hi, b, neg(hi), fe);
    return DD{hi, fma_(a.lo, b, lo, fe)};
}

SF64_ALWAYS_INLINE DD ddmul_dd_dd_dd(DD a, DD b, sf64_internal_fe_acc& fe) noexcept {
    const double hi = mul_(a.hi, b.hi, fe);
    double lo = fma_(a.hi, b.hi, neg(hi), fe);
    lo = fma_(a.lo, b.hi, lo, fe);
    lo = fma_(a.hi, b.lo, lo, fe);
    return DD{hi, lo};
}

// ddsqu (square) with FMA.
SF64_ALWAYS_INLINE DD ddsqu_dd_dd(DD a, sf64_internal_fe_acc& fe) noexcept {
    const double hi = mul_(a.hi, a.hi, fe);
    double lo = fma_(a.hi, a.hi, neg(hi), fe);
    const double twolo = add_(a.lo, a.lo, fe);
    lo = fma_(a.hi, twolo, lo, fe);
    return DD{hi, lo};
}

// Reciprocal: 1/b expressed as DD using Newton correction.
SF64_ALWAYS_INLINE DD ddrec_dd_d(double b, sf64_internal_fe_acc& fe) noexcept {
    const double t = div_(1.0, b, fe);
    const double u = fma_(neg(t), b, 1.0, fe);
    // lo = t * u  (approximation; adequate for our tolerances)
    return DD{t, mul_(t, u, fe)};
}

SF64_ALWAYS_INLINE DD ddrec_dd_dd(DD b, sf64_internal_fe_acc& fe) noexcept {
    const double t = div_(1.0, b.hi, fe);
    double u = fma_(neg(t), b.hi, 1.0, fe);
    u = fma_(neg(t), b.lo, u, fe);
    return DD{t, mul_(t, u, fe)};
}

// Division.
SF64_ALWAYS_INLINE DD dddiv_dd_dd_dd(DD n, DD d, sf64_internal_fe_acc& fe) noexcept {
    const DD r = ddrec_dd_dd(d, fe);
    return ddmul_dd_dd_dd(n, r, fe);
}

// Normalise (renormalise hi+lo so |lo| <= ulp(hi)/2).
SF64_ALWAYS_INLINE DD ddnormalize_dd_dd(DD a, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a.hi, a.lo, fe);
    const double e = add_(sub_(a.hi, s, fe), a.lo, fe);
    return DD{s, e};
}

// Convert a DD to a single double (just return hi + lo).
SF64_ALWAYS_INLINE double dd_to_d(DD a, sf64_internal_fe_acc& fe) noexcept {
    return add_(a.hi, a.lo, fe);
}

// ddscale_dd_dd_d: multiply a DD by a power of two (lossless since the
// mantissa doesn't change, only the exponent).
SF64_ALWAYS_INLINE DD ddscale_dd_dd_d(DD d, double s, sf64_internal_fe_acc& fe) noexcept {
    return DD{mul_(d.hi, s, fe), mul_(d.lo, s, fe)};
}

SF64_ALWAYS_INLINE DD ddneg_dd_dd(DD a) noexcept {
    return DD{neg(a.hi), neg(a.lo)};
}

// ---- DD helpers with "|a| >= |b|" precondition (fast TwoSum) ----------
// These live in sleef_common.h (not anonymous-ns file-local to trig) so
// sleef_fe_macros.h can append `, fe` uniformly across consumer TUs. The
// `add_` / `sub_` calls are the RNE-specialized inline helpers declared
// above — each raise feeds into the caller's stack-local accumulator.

SF64_ALWAYS_INLINE DD ddadd_dd_d_d(double a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a, b, fe);
    const double v = sub_(s, a, fe);
    const double e = sub_(b, v, fe);
    return DD{s, e};
}

SF64_ALWAYS_INLINE DD ddadd_dd_dd_d(DD a, double b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a.hi, b, fe);
    const double v = sub_(s, a.hi, fe);
    const double e = add_(sub_(b, v, fe), a.lo, fe);
    return DD{s, e};
}

SF64_ALWAYS_INLINE DD ddadd_dd_d_dd(double a, DD b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a, b.hi, fe);
    const double v = sub_(s, a, fe);
    const double e = add_(sub_(b.hi, v, fe), b.lo, fe);
    return DD{s, e};
}

SF64_ALWAYS_INLINE DD ddadd_dd_dd_dd(DD a, DD b, sf64_internal_fe_acc& fe) noexcept {
    const double s = add_(a.hi, b.hi, fe);
    const double v = sub_(s, a.hi, fe);
    const double e = add_(sub_(b.hi, v, fe), add_(a.lo, b.lo, fe), fe);
    return DD{s, e};
}

// ddmul_d_dd_dd — SLEEF's tail reduction in the u1 cores: collapses the
// DD product x*y to a single double via 5 FMAs. No explicit error limb;
// the final round absorbs the low-order residual.
SF64_ALWAYS_INLINE double ddmul_d_dd_dd(DD x, DD y, sf64_internal_fe_acc& fe) noexcept {
    const double xh = upper(x.hi);
    const double xl = sub_(x.hi, xh, fe);
    const double yh = upper(y.hi);
    const double yl = sub_(y.hi, yh, fe);
    double acc = mul_(x.lo, yh, fe);
    acc = fma_(xh, y.lo, acc, fe);
    acc = fma_(xl, yl, acc, fe);
    acc = fma_(xh, yl, acc, fe);
    acc = fma_(xl, yh, acc, fe);
    acc = fma_(xh, yh, acc, fe);
    return acc;
}

} // namespace soft_fp64::sleef
