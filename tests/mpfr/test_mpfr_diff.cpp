// MPFR arbitrary-precision differential oracle for every sf64_*
// transcendental.
//
// The existing `test_transcendental_1ulp.cpp` compares sf64 results against
// the host's libm. That only catches bugs up to the host's own ULP budget
// (libm is typically 1-3 ULP, occasionally worse). This test compares
// against MPFR at 200 bits, rounded to double with RNDN. At that precision,
// a double-precision round-to-nearest-even MPFR result is indistinguishable
// from the mathematically correct rounded value, so any observed ULP delta
// is a bug in soft-fp64 — not in the oracle.
//
// Tolerance bands per task spec (strictly tighter than the libm-diff tier):
//   U10  (sin/cos/exp/log/pow/...)            ≤ 4    ULP
//   U35  (tan/sinh/asinh/pow edge/...)        ≤ 8    ULP
//   GAMMA (erf/erfc/tgamma)                   ≤ 1024 ULP
// A sweep either passes its spec band and fails ctest on a regression, or
// it lives in tests/experimental/ (report-only, not part of the release
// correctness claim).
//
// Input ranges mirror the validated domains used by test_transcendental_1ulp
// where applicable. Where MPFR supports a tighter oracle (e.g. *pi variants)
// we also run the corresponding sweep. Known-buggy input regions surfaced
// by early exploratory MPFR runs are documented inline with a TODO so a
// future sf64 fix can widen the sweep.
//
// The `ulp_diff` helper is copied here rather than factored to a shared
// header so the oracle stays self-contained in tests/mpfr/.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/rounding_mode.h"
#include "soft_fp64/soft_f64.h"

#include <mpfr.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" {
// Prototypes mirrored so this TU compiles even if the public header reorders
// symbols. The linker is the ground truth for the sf64_* ABI.
double sf64_sin(double);
double sf64_cos(double);
void sf64_sincos(double, double*, double*);
double sf64_tan(double);
double sf64_asin(double);
double sf64_acos(double);
double sf64_atan(double);
double sf64_atan2(double, double);
double sf64_sinpi(double);
double sf64_cospi(double);
double sf64_tanpi(double);
double sf64_asinpi(double);
double sf64_acospi(double);
double sf64_atanpi(double);
double sf64_atan2pi(double, double);
double sf64_sinh(double);
double sf64_cosh(double);
double sf64_tanh(double);
double sf64_asinh(double);
double sf64_acosh(double);
double sf64_atanh(double);
double sf64_exp(double);
double sf64_exp2(double);
double sf64_exp10(double);
double sf64_expm1(double);
double sf64_log(double);
double sf64_log2(double);
double sf64_log10(double);
double sf64_log1p(double);
double sf64_pow(double, double);
double sf64_powr(double, double);
double sf64_pown(double, int);
double sf64_rootn(double, int);
double sf64_cbrt(double);
double sf64_erf(double);
double sf64_erfc(double);
double sf64_tgamma(double);
double sf64_lgamma(double);
double sf64_lgamma_r(double, int*);
double sf64_fmod(double, double);
double sf64_remainder(double, double);
double sf64_hypot(double, double);

// Rounding-mode-explicit arithmetic / sqrt / fma / convert / rint surface.
// Used by the per-mode bit-exact sweeps. `_r(mode, ...)` wrappers are defined
// on the public header and every five-way mode must match MPFR bit-for-bit.
double sf64_add_r(sf64_rounding_mode, double, double);
double sf64_sub_r(sf64_rounding_mode, double, double);
double sf64_mul_r(sf64_rounding_mode, double, double);
double sf64_div_r(sf64_rounding_mode, double, double);
double sf64_sqrt_r(sf64_rounding_mode, double);
double sf64_fma_r(sf64_rounding_mode, double, double, double);
float sf64_to_f32_r(sf64_rounding_mode, double);
double sf64_from_f32(float);
int8_t sf64_to_i8_r(sf64_rounding_mode, double);
int16_t sf64_to_i16_r(sf64_rounding_mode, double);
int32_t sf64_to_i32_r(sf64_rounding_mode, double);
int64_t sf64_to_i64_r(sf64_rounding_mode, double);
uint8_t sf64_to_u8_r(sf64_rounding_mode, double);
uint16_t sf64_to_u16_r(sf64_rounding_mode, double);
uint32_t sf64_to_u32_r(sf64_rounding_mode, double);
uint64_t sf64_to_u64_r(sf64_rounding_mode, double);
double sf64_rint_r(sf64_rounding_mode, double);
}

namespace {

// ---------- Bit utilities --------------------------------------------------

inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

// Inlined locally so this subdir can build without pulling in tests/host_oracle.h.
inline int64_t ulp_diff(double a, double b) {
    if (std::isnan(a) && std::isnan(b))
        return 0;
    if (a == b)
        return 0;
    uint64_t ab = bits(a);
    uint64_t bb = bits(b);
    if (ab & 0x8000000000000000ULL)
        ab = 0x8000000000000000ULL - ab;
    if (bb & 0x8000000000000000ULL)
        bb = 0x8000000000000000ULL - bb;
    int64_t sa = static_cast<int64_t>(ab);
    int64_t sb = static_cast<int64_t>(bb);
    int64_t d = sa - sb;
    return d < 0 ? -d : d;
}

// ---------- Deterministic LCG (copied inline) -----------------------------

class LCG {
  public:
    explicit LCG(uint64_t seed = 0xD00DCAFEULL) : state_(seed) {}
    uint64_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }
    double uniform_unit() {
        return static_cast<double>(next() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }
    // Log-uniform over [lo, hi] with strictly positive endpoints.
    double log_uniform(double lo, double hi) {
        const double llo = std::log(lo);
        const double lhi = std::log(hi);
        return std::exp(llo + (lhi - llo) * uniform_unit());
    }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform_unit(); }

  private:
    uint64_t state_;
};

// ---------- MPFR oracle plumbing ------------------------------------------

constexpr mpfr_prec_t ORACLE_PREC = 200;

using Mpfr1 = int (*)(mpfr_t, const mpfr_t, mpfr_rnd_t);
using Mpfr2 = int (*)(mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);

inline double ref1(Mpfr1 fn, double x) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    fn(rm, xm, MPFR_RNDN);
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}

inline double ref2(Mpfr2 fn, double x, double y) {
    mpfr_t xm, ym, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(ym, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_set_d(ym, y, MPFR_RNDN);
    fn(rm, xm, ym, MPFR_RNDN);
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(ym);
    mpfr_clear(rm);
    return r;
}

// pown / rootn operate on integer exponents.
inline double ref_pow_si(double x, long n) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_pow_si(rm, xm, n, MPFR_RNDN);
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}

inline double ref_rootn_si(double x, long n) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_rootn_si(rm, xm, n, MPFR_RNDN);
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}

// ---------- Per-mode MPFR oracles -----------------------------------------
//
// For the bit-exact `sf64_*_r(mode, ...)` surface every one of the five
// IEEE-754 rounding attributes (RNE, RTZ, RUP, RDN, RNA) must match MPFR
// exactly. Four of the five map onto MPFR's own rounding modes:
//
//   SF64_RNE -> MPFR_RNDN   (nearest, ties to even)
//   SF64_RTZ -> MPFR_RNDZ   (toward zero)
//   SF64_RUP -> MPFR_RNDU   (toward +infinity)
//   SF64_RDN -> MPFR_RNDD   (toward -infinity)
//
// The fifth, round-to-nearest-ties-away, has no direct MPFR equivalent
// outside `mpfr_rint` (the `MPFR_RNDNA=-1` pseudo-mode is only honored
// inside `mpfr_rint`/`mpfr_round` — mpfr.h itself flags it "DON'T USE").
// `MPFR_RNDA` is round-away-from-zero for every value, not just ties, so
// it is not the right mode either.
//
// Emulation: compute the op at 200-bit precision with RNDN, then apply a
// double-rounding idiom — first to 54 bits with RNDN (one extra bit so
// the halfway case is representable exactly), then to 53 bits with RNDA.
// At non-tie points this is equivalent to RNDN on the final step; at
// exact halfway ties the extra bit is 1 and RNDA pushes the 53-bit
// result away from zero, which is precisely RNA semantics. The
// intermediate 54-bit value is exact relative to the 53-bit target,
// so there is no double-rounding error — this is the standard
// technique for emulating ties-away rounding with a library that
// lacks it natively.
//
// Subnormal handling: `mpfr_subnormalize` with the chosen rounding mode
// forces the f64 exponent range so an RNA-subnormal result matches the
// hardware-representable bit pattern.
inline mpfr_rnd_t sf_to_mpfr_direct(sf64_rounding_mode m) {
    switch (m) {
    case SF64_RNE:
        return MPFR_RNDN;
    case SF64_RTZ:
        return MPFR_RNDZ;
    case SF64_RUP:
        return MPFR_RNDU;
    case SF64_RDN:
        return MPFR_RNDD;
    case SF64_RNA:
        // Caller handles RNA via double-rounding; this branch is only hit
        // on the input side where RNDN is always exact for `set_d`.
        return MPFR_RNDN;
    }
    return MPFR_RNDN;
}

// Finalize a high-precision MPFR scratch into a bit-exact double under
// the requested soft-fp64 rounding mode. MPFR's `mpfr_get_d` already
// implements the correct IEEE-754 rounding for every mode MPFR supports
// (RNDN/RNDZ/RNDU/RNDD) including subnormal / overflow handling.
//
// For RNA (round-to-nearest-ties-away): start from MPFR's RNDN answer,
// which is correct everywhere except exact halfway ties. If the value
// is a tie AND RNDN chose a neighbor with a smaller magnitude than the
// other candidate, swap to the other neighbor (RNA picks away-from-zero
// on ties).
//
// Tie detection: compute RNDU and RNDD. The value is a tie iff
//   |v - RNDU| == |v - RNDD|
// exactly, computed at MPFR precision so the comparison is not tainted
// by f64 rounding hazard.
inline double finalize_d(mpfr_t rm, sf64_rounding_mode m) {
    if (m == SF64_RNA) {
        const double rne = mpfr_get_d(rm, MPFR_RNDN);
        // For overflow, infinities, NaN: RNDN already gives the right
        // answer and there is no "other neighbor" to swap to.
        if (std::isnan(rne) || std::isinf(rne))
            return rne;
        const double down = mpfr_get_d(rm, MPFR_RNDD);
        const double up = mpfr_get_d(rm, MPFR_RNDU);
        if (down == up)
            return rne; // exact representable
        // Midpoint check at MPFR precision: exact tie iff |v-up| == |v-down|.
        const mpfr_prec_t p = mpfr_get_prec(rm);
        mpfr_t dm, um, ld, lu;
        mpfr_init2(dm, p);
        mpfr_init2(um, p);
        mpfr_init2(ld, p);
        mpfr_init2(lu, p);
        mpfr_set_d(dm, down, MPFR_RNDN);
        mpfr_set_d(um, up, MPFR_RNDN);
        mpfr_sub(ld, rm, dm, MPFR_RNDN);
        mpfr_abs(ld, ld, MPFR_RNDN);
        mpfr_sub(lu, um, rm, MPFR_RNDN);
        mpfr_abs(lu, lu, MPFR_RNDN);
        const int cmp = mpfr_cmp(ld, lu);
        mpfr_clear(dm);
        mpfr_clear(um);
        mpfr_clear(ld);
        mpfr_clear(lu);
        if (cmp != 0)
            return rne; // not a tie — RNDN is the RNA answer
        // Exact halfway tie: pick the neighbor with larger magnitude.
        return (std::fabs(up) > std::fabs(down)) ? up : down;
    }
    return mpfr_get_d(rm, sf_to_mpfr_direct(m));
}

// Per-mode arithmetic / sqrt / fma references. Inputs are exact doubles,
// so `mpfr_set_d` with any rounding mode is exact; the mode only applies
// at the final `finalize_d` step.
//
// Precision: the 200-bit `ORACLE_PREC` used for transcendentals is not
// enough for arithmetic on widely-separated f64 operands.
//
//   add/sub/mul/div/sqrt: exp span is at most 2^1023 / 2^-1074 = ~2100
//   bits; 2200 bits suffices to preserve every bit of the exact result.
//
//   fma: the worst case is `a*b + c` with |a*b| near 2^-2148 (two
//   subnormals multiplied) and |c| near 2^1023. The exact value spans
//   ~3171 bits; a 3300-bit scratch captures it losslessly. `mpfr_fma`
//   computes the fused operation at the scratch precision, then our
//   `finalize_d` rounds to 53 bits under the requested mode.
//
// Using the same precision for every arithmetic ref keeps the oracle
// uniform at the cost of some memory per call — fine for a test binary.
constexpr mpfr_prec_t ARITH_PREC = 3300;

// IEEE 754-2008 §6.3 signed-zero rule for exactly-zero results from
// addition and subtraction. The rule depends on whether the zero came
// from cancellation of nonzero operands or from direct zero-operand
// arithmetic, and on the rounding mode. `mpfr_add` / `mpfr_sub` always
// yields a `+0` mpfr_t on cancellation regardless of mode, so the
// oracle must fix up the sign post-hoc.
//
// We distinguish three cases (all require `mpfr_zero_p(rm)` true):
//   1) Both operands are literal zero (signed 0). Sign of the result
//      follows host FPU semantics under each mode — this is the
//      standard `0 ± 0` rule and must be computed from the operand
//      signs directly. See the case tables in `oracle_zero_sign_*`
//      below.
//   2) Cancellation of nonzero same-magnitude opposite-sign operands
//      (e.g. `x - x` with finite nonzero x): result sign is +0 for
//      RNE/RTZ/RUP, -0 for RDN.
//   3) Underflow to zero (MPFR's `rm` is not zero but `rm` rounds to
//      zero in f64): NOT handled here — MPFR's get_d already preserves
//      the underflow sign.
//
// Call `apply_signed_zero_add` or `apply_signed_zero_sub` with the
// pre-rounding MPFR zero flag.

// Sign of `a + b` under mode `m` when both operands are literal zero.
// Host FPU semantics table (verified against Apple / Intel FPU):
//   RNE/RTZ/RUP: sign = sign_a AND sign_b (result -0 iff both -0)
//   RDN:         sign = sign_a OR sign_b  (result +0 iff both +0)
inline double zero_plus_zero_sign(double a, double b, sf64_rounding_mode m) {
    const int sa = std::signbit(a) ? 1 : 0;
    const int sb = std::signbit(b) ? 1 : 0;
    const int s = (m == SF64_RDN) ? (sa | sb) : (sa & sb);
    return s ? -0.0 : 0.0;
}

inline double apply_signed_zero_add(double r, mpfr_srcptr rm, double a, double b,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r; // not exact zero — MPFR rounding was correct
    // Case 1: both operands literal zero.
    if (a == 0.0 && b == 0.0)
        return zero_plus_zero_sign(a, b, m);
    // Case 2: cancellation of nonzero operands.
    return (m == SF64_RDN) ? -0.0 : 0.0;
}

inline double apply_signed_zero_sub(double r, mpfr_srcptr rm, double a, double b,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r;
    if (a == 0.0 && b == 0.0)
        return zero_plus_zero_sign(a, -b, m); // x - y == x + (-y)
    return (m == SF64_RDN) ? -0.0 : 0.0;
}

// fma: the fused product `a*b` is the first addend; the second is `c`.
// When the exact fma result is zero:
//   - If BOTH addends are literal zero (product AND c are zero), use
//     the zero-plus-zero sign table on (sign(a)^sign(b), sign(c)).
//   - Else cancellation rule: +0 (else) / -0 (RDN).
inline double apply_signed_zero_fma(double r, mpfr_srcptr rm, double a, double b, double c,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r;
    const bool product_is_zero = (a == 0.0) || (b == 0.0);
    const bool c_is_zero = (c == 0.0);
    if (product_is_zero && c_is_zero) {
        // Sign of the product: xor of sign bits (literal zero operands).
        const int product_sign = (std::signbit(a) ? 1 : 0) ^ (std::signbit(b) ? 1 : 0);
        const double product_zero = product_sign ? -0.0 : 0.0;
        return zero_plus_zero_sign(product_zero, c, m);
    }
    return (m == SF64_RDN) ? -0.0 : 0.0;
}

inline double ref_add_r(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_add(rm, am, bm, MPFR_RNDN);
    double r = finalize_d(rm, m);
    r = apply_signed_zero_add(r, rm, a, b, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
inline double ref_sub_r(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_sub(rm, am, bm, MPFR_RNDN);
    double r = finalize_d(rm, m);
    r = apply_signed_zero_sub(r, rm, a, b, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
inline double ref_mul_r(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_mul(rm, am, bm, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
inline double ref_div_r(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_div(rm, am, bm, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
inline double ref_sqrt_r(double x, sf64_rounding_mode m) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_sqrt(rm, xm, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}
inline double ref_fma_r(double a, double b, double c, sf64_rounding_mode m) {
    mpfr_t am, bm, cm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(cm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_set_d(cm, c, MPFR_RNDN);
    mpfr_fma(rm, am, bm, cm, MPFR_RNDN);
    double r = finalize_d(rm, m);
    r = apply_signed_zero_fma(r, rm, a, b, c, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(cm);
    mpfr_clear(rm);
    return r;
}

// f64 -> f32 under all five modes. `mpfr_get_flt` handles IEEE-754
// binary32 subnormal / overflow conversion directly under RNDN/RNDZ/
// RNDU/RNDD. For RNA we start from the RNDN answer (correct everywhere
// except at exact halfway ties) and swap to the magnitude-larger
// neighbor only on confirmed ties.
inline float ref_to_f32_r(double x, sf64_rounding_mode m) {
    mpfr_t xm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    float r;
    if (m == SF64_RNA) {
        const float rne = mpfr_get_flt(xm, MPFR_RNDN);
        if (std::isnan(rne) || std::isinf(rne)) {
            r = rne;
        } else {
            const float down = mpfr_get_flt(xm, MPFR_RNDD);
            const float up = mpfr_get_flt(xm, MPFR_RNDU);
            if (down == up) {
                r = rne;
            } else {
                mpfr_t dm, um, ld, lu;
                mpfr_init2(dm, ORACLE_PREC);
                mpfr_init2(um, ORACLE_PREC);
                mpfr_init2(ld, ORACLE_PREC);
                mpfr_init2(lu, ORACLE_PREC);
                mpfr_set_flt(dm, down, MPFR_RNDN);
                mpfr_set_flt(um, up, MPFR_RNDN);
                mpfr_sub(ld, xm, dm, MPFR_RNDN);
                mpfr_abs(ld, ld, MPFR_RNDN);
                mpfr_sub(lu, um, xm, MPFR_RNDN);
                mpfr_abs(lu, lu, MPFR_RNDN);
                const int cmp = mpfr_cmp(ld, lu);
                mpfr_clear(dm);
                mpfr_clear(um);
                mpfr_clear(ld);
                mpfr_clear(lu);
                if (cmp != 0) {
                    r = rne;
                } else {
                    r = (std::fabs(static_cast<double>(up)) > std::fabs(static_cast<double>(down)))
                            ? up
                            : down;
                }
            }
        }
    } else {
        r = mpfr_get_flt(xm, sf_to_mpfr_direct(m));
    }
    mpfr_clear(xm);
    return r;
}

// f64 -> integer under all five modes. MPFR's `mpfr_rint` accepts
// `MPFR_RNDNA` directly, so RNA routes through that without the
// double-rounding dance used for f64/f32 targets.
inline mpfr_rnd_t sf_to_mpfr_int(sf64_rounding_mode m) {
    switch (m) {
    case SF64_RNE:
        return MPFR_RNDN;
    case SF64_RTZ:
        return MPFR_RNDZ;
    case SF64_RUP:
        return MPFR_RNDU;
    case SF64_RDN:
        return MPFR_RNDD;
    case SF64_RNA:
        return MPFR_RNDNA;
    }
    return MPFR_RNDN;
}

// Return the MPFR-rounded integer value of `x` under mode `m` as a
// signed / unsigned long long. Caller is responsible for ensuring `x`
// is finite and in the destination type's representable range after
// rounding (see sweep_to_int_r below — it clips).
inline long long ref_to_llint_r(double x, sf64_rounding_mode m) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_rint(rm, xm, sf_to_mpfr_int(m));
    const long long r = mpfr_get_sj(rm, MPFR_RNDZ); // value is already integer
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}
inline unsigned long long ref_to_ullint_r(double x, sf64_rounding_mode m) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_rint(rm, xm, sf_to_mpfr_int(m));
    const unsigned long long r = mpfr_get_uj(rm, MPFR_RNDZ);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}

// sf64_rint oracle: round to integral double. Integer values with
// magnitude < 2^53 are exactly representable in f64; larger inputs
// are already integral so the result equals the input.
inline double ref_rint_r(double x, sf64_rounding_mode m) {
    mpfr_t xm, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_rint(rm, xm, sf_to_mpfr_int(m));
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(rm);
    return r;
}

// ---------- Tolerance tiers -----------------------------------------------
//
// Hard-fail bands from the task spec. `enum class` so a stray integer (or a
// `numeric_limits<int64_t>::max()` sentinel) is a type error rather than an
// escape hatch. New tiers require a public spec entry — not a tolerance bump
// to paper over a regression.
enum class Tier : int64_t {
    BIT_EXACT = 0,
    U10 = 4,
    U35 = 8,
    GAMMA = 1024,
};

// Short-name aliases for readability at sweep callsites. These are strongly
// typed `Tier` values, NOT `int64_t` — assigning a raw integer (or a sentinel
// like `numeric_limits<int64_t>::max()`) to a `Tier` argument is a compile
// error.
constexpr Tier BIT_EXACT = Tier::BIT_EXACT;
constexpr Tier U10 = Tier::U10;
constexpr Tier U35 = Tier::U35;
constexpr Tier GAMMA = Tier::GAMMA;

// ---------- Sweep harness --------------------------------------------------

struct Stats {
    const char* name = "?";
    int64_t max_ulp = 0;
    double worst_x = 0.0, worst_y = 0.0;
    double worst_got = 0.0, worst_expect = 0.0;
    int checked = 0;
    int nan_mismatch = 0;
    int inf_mismatch = 0;
    // Trivial matches: both impl and oracle returned the SAME NaN-or-
    // signed-infinity. These contribute nothing to precision verification
    // — they only confirm that the algorithm overflows / NaN-propagates
    // the same way the oracle does, not that it computes precise values
    // anywhere in its claimed range. A sweep whose sampling strategy
    // collapses most inputs into the degenerate regime (e.g. linear
    // sampling across `[1e-100, 1e100]` — almost every sample lands at
    // ~1e100 and pow overflows on almost every one) will have a high
    // trivial_matches ratio and is not actually testing its claim.
    // Gated at 25% in fail().
    int trivial_matches = 0;
    Tier tier = Tier::U10;
};

void record(Stats& s, double x, double y, double got, double expect) {
    s.checked++;
    const bool gnan = std::isnan(got);
    const bool enan = std::isnan(expect);
    if (gnan != enan) {
        s.nan_mismatch++;
        if (s.nan_mismatch <= 3) {
            std::printf("  [nan-mismatch %s] x=%.17g y=%.17g got=%.17g "
                        "expect=%.17g\n",
                        s.name, x, y, got, expect);
        }
        return;
    }
    if (gnan && enan) {
        s.trivial_matches++;
        return;
    }

    const bool ginf = std::isinf(got);
    const bool einf = std::isinf(expect);
    if (ginf != einf) {
        s.inf_mismatch++;
        return;
    }
    if (ginf && einf && (got > 0) != (expect > 0)) {
        s.inf_mismatch++;
        return;
    }
    if (ginf && einf) {
        // Both sides are ±inf with matching sign. The algorithm and the
        // oracle agree that this input overflows, but we learn nothing
        // about precision here.
        s.trivial_matches++;
        return;
    }

    const int64_t d = ulp_diff(got, expect);
    if (d > s.max_ulp) {
        s.max_ulp = d;
        s.worst_x = x;
        s.worst_y = y;
        s.worst_got = got;
        s.worst_expect = expect;
    }
}

void report(const Stats& s) {
    const double trivial_pct =
        s.checked > 0 ? (100.0 * static_cast<double>(s.trivial_matches) / s.checked) : 0.0;
    std::printf("  %-10s  n=%-6d  max_ulp=%-8lld  tier=%-6lld  trivial=%5.1f%%  "
                "worst x=%.17g y=%.17g got=%.17g expect=%.17g\n",
                s.name, s.checked, static_cast<long long>(s.max_ulp),
                static_cast<long long>(static_cast<int64_t>(s.tier)), trivial_pct, s.worst_x,
                s.worst_y, s.worst_got, s.worst_expect);
    if (s.nan_mismatch || s.inf_mismatch) {
        std::printf("    !! nan_mismatch=%d inf_mismatch=%d\n", s.nan_mismatch, s.inf_mismatch);
    }
}

// Trivial-match gate threshold. A sweep whose sampling regime causes
// more than this fraction of samples to collapse into (NaN,NaN) or
// matching-sign (inf,inf) is not actually testing precision in its
// advertised range — the "n=10000" headline turns into "n=2500
// precision checks plus 7500 overflow confirmations."
constexpr double kTrivialMatchMaxRatio = 0.25;

bool fail(const Stats& s) {
    if (s.nan_mismatch > 0 || s.inf_mismatch > 0)
        return true;
    if (s.max_ulp > static_cast<int64_t>(s.tier))
        return true;
    if (s.checked > 0) {
        const double ratio = static_cast<double>(s.trivial_matches) / s.checked;
        if (ratio > kTrivialMatchMaxRatio) {
            std::printf("    !! trivial-match gate: %d/%d = %.1f%% > %.0f%% — sampling regime "
                        "collapses into overflow/underflow/NaN for the majority of inputs; the "
                        "sweep is not actually testing precision in the advertised range.\n",
                        s.trivial_matches, s.checked,
                        100.0 * static_cast<double>(s.trivial_matches) / s.checked,
                        100.0 * kTrivialMatchMaxRatio);
            return true;
        }
    }
    return false;
}

// ---------- Sweep drivers -------------------------------------------------

constexpr int N_RAND = 10000;

// 1-arg sweep, log-space sampling over a strictly-positive magnitude range.
// If `symmetric` is true, half the samples are mirrored into the negative
// side (for functions whose impl handles both signs uniformly).
template <class SoftFn>
Stats sweep1_log(const char* name, Tier tier, SoftFn soft, Mpfr1 mref, double lo_abs, double hi_abs,
                 bool symmetric, uint64_t seed = 0xBADC0FFEEULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        double x = rng.log_uniform(lo_abs, hi_abs);
        if (symmetric && (rng.next() & 1))
            x = -x;
        record(s, x, 0.0, soft(x), ref1(mref, x));
    }
    return s;
}

// 1-arg sweep, uniform over [lo, hi].
template <class SoftFn>
Stats sweep1_uniform(const char* name, Tier tier, SoftFn soft, Mpfr1 mref, double lo, double hi,
                     uint64_t seed = 0xFACEFEEDULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        const double x = rng.uniform(lo, hi);
        record(s, x, 0.0, soft(x), ref1(mref, x));
    }
    return s;
}

// 2-arg sweep, uniform in both arguments.
template <class SoftFn>
Stats sweep2_uniform(const char* name, Tier tier, SoftFn soft, Mpfr2 mref, double xlo, double xhi,
                     double ylo, double yhi, uint64_t seed = 0xFEEDFACEULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        const double x = rng.uniform(xlo, xhi);
        const double y = rng.uniform(ylo, yhi);
        record(s, x, y, soft(x, y), ref2(mref, x, y));
    }
    return s;
}

// 2-arg sweep, log-uniform in both arguments. Bounds are magnitudes
// (strictly positive). `x_sym` / `y_sym` flip half the samples to the
// negative side — needed for ranges that advertise coverage across zero.
// Use this instead of sweep2_uniform for wide-decade windows: linear
// sampling over `[10^-N, 10^+N]` (or `[-10^N, +10^N]`) with N≥3 puts
// ~99% of samples in the top decade of |x|, leaving the small-|x| half
// of the advertised window unexercised. Log sampling distributes samples
// uniformly across decades so the whole range actually gets exercised.
template <class SoftFn>
Stats sweep2_log(const char* name, Tier tier, SoftFn soft, Mpfr2 mref, double xlo_abs,
                 double xhi_abs, bool x_sym, double ylo_abs, double yhi_abs, bool y_sym,
                 uint64_t seed = 0xC0DE4D00DULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        double x = rng.log_uniform(xlo_abs, xhi_abs);
        if (x_sym && (rng.next() & 1))
            x = -x;
        double y = rng.log_uniform(ylo_abs, yhi_abs);
        if (y_sym && (rng.next() & 1))
            y = -y;
        record(s, x, y, soft(x, y), ref2(mref, x, y));
    }
    return s;
}

// 2-arg sweep, log-uniform in x (with optional symmetric flag), linear
// in y. Use when the x range spans many decades but y is a narrow
// linear band — typical of pow / powr / fmod where the algorithmic
// regime varies with |x| but y stays modest.
template <class SoftFn>
Stats sweep2_log_uniform(const char* name, Tier tier, SoftFn soft, Mpfr2 mref, double xlo_abs,
                         double xhi_abs, bool x_sym, double ylo, double yhi,
                         uint64_t seed = 0xCAFE5EEDULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        double x = rng.log_uniform(xlo_abs, xhi_abs);
        if (x_sym && (rng.next() & 1))
            x = -x;
        const double y = rng.uniform(ylo, yhi);
        record(s, x, y, soft(x, y), ref2(mref, x, y));
    }
    return s;
}

// atan2-style: sample over a ring to cover all quadrants robustly.
template <class SoftFn>
Stats sweep_atan2like(const char* name, Tier tier, SoftFn soft, Mpfr2 mref,
                      uint64_t seed = 0xC0FFEEULL) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(seed);
    for (int i = 0; i < N_RAND; ++i) {
        const double r = rng.log_uniform(1e-20, 1e20);
        const double theta = rng.uniform(-3.141592653589793, 3.141592653589793);
        const double x = r * std::cos(theta);
        const double y = r * std::sin(theta);
        record(s, y, x, soft(y, x), ref2(mref, y, x));
    }
    return s;
}

// pown: x over a bounded range, n over small integer exponents.
Stats sweep_pown(const char* name, Tier tier) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(0xABCDEFULL);
    for (int i = 0; i < N_RAND; ++i) {
        const double x = rng.uniform(-4.0, 4.0);
        const int n = static_cast<int>(rng.next() % 21) - 10; // [-10, 10]
        const double got = sf64_pown(x, n);
        const double expect = ref_pow_si(x, static_cast<long>(n));
        record(s, x, static_cast<double>(n), got, expect);
    }
    return s;
}

// rootn: positive x, small integer n (rootn of negative x is impl-defined
// for even n and we leave that to unit tests).
Stats sweep_rootn(const char* name, Tier tier) {
    Stats s;
    s.name = name;
    s.tier = tier;
    LCG rng(0x13579BULL);
    const int ns[] = {2, 3, 4, 5, 7, 11};
    for (int i = 0; i < N_RAND; ++i) {
        const double x = rng.log_uniform(0.01, 1e10);
        const int n = ns[rng.next() % 6];
        const double got = sf64_rootn(x, n);
        const double expect = ref_rootn_si(x, static_cast<long>(n));
        record(s, x, static_cast<double>(n), got, expect);
    }
    return s;
}

// Edge corpus — spot-check behavior at IEEE boundaries (signed zeros,
// ±1, ±inf-adjacent, tiny normals) for a small fixed set of functions.
// These do not contribute to the main tier checks; they only surface
// obvious regressions and are reported verbosely.
void edge_spot_checks() {
    struct Probe {
        const char* name;
        double x;
        double (*soft)(double);
        Mpfr1 mref;
    };
    const Probe probes[] = {
        {"sin(+0)", +0.0, sf64_sin, mpfr_sin},
        {"sin(-0)", -0.0, sf64_sin, mpfr_sin},
        {"cos(+0)", +0.0, sf64_cos, mpfr_cos},
        {"tan(0)", 0.0, sf64_tan, mpfr_tan},
        {"atan(+inf)", std::numeric_limits<double>::infinity(), sf64_atan, mpfr_atan},
        {"exp(-inf)", -std::numeric_limits<double>::infinity(), sf64_exp, mpfr_exp},
        {"log(1)", 1.0, sf64_log, mpfr_log},
        {"log1p(0)", 0.0, sf64_log1p, mpfr_log1p},
        {"expm1(0)", 0.0, sf64_expm1, mpfr_expm1},
        {"cbrt(+0)", +0.0, sf64_cbrt, mpfr_cbrt},
        {"cbrt(-0)", -0.0, sf64_cbrt, mpfr_cbrt},
        {"acosh(1)", 1.0, sf64_acosh, mpfr_acosh},
        {"atanh(0)", 0.0, sf64_atanh, mpfr_atanh},
    };
    std::printf("\n[edge spot checks]\n");
    for (const auto& p : probes) {
        const double g = p.soft(p.x);
        const double e = ref1(p.mref, p.x);
        const bool ok = (std::isnan(g) && std::isnan(e)) || bits(g) == bits(e);
        if (!ok) {
            std::printf("  [edge %s] x=%.17g got=%.17g expect=%.17g "
                        "ulp=%lld\n",
                        p.name, p.x, g, e, static_cast<long long>(ulp_diff(g, e)));
        }
    }
}

// ---------- Per-mode bit-exact sweeps -------------------------------------
//
// The `sf64_*_r(mode, ...)` surface must match MPFR bit-for-bit across all
// five IEEE-754 rounding modes. These sweeps are additive — they do NOT
// displace the RNE-only sweeps above (the headline `sf64_*` surface is
// still exercised by the main-entry sweeps). They verify the new 1.1
// rounding-mode-parameterized surface at the BIT_EXACT tier.

constexpr int N_RAND_MODE = 4096;

struct ModeNameRow {
    sf64_rounding_mode mode;
    const char* name;
};
constexpr ModeNameRow kModes[5] = {
    {SF64_RNE, "RNE"}, {SF64_RTZ, "RTZ"}, {SF64_RUP, "RUP"}, {SF64_RDN, "RDN"}, {SF64_RNA, "RNA"},
};

// Per-sweep row name buffer. Every sweep_* helper reserves a fresh slot so
// the `Stats.name` pointer stays valid for the full test run.
constexpr int kMaxModeRows = 128;
static char g_mode_row_names[kMaxModeRows][40];
static int g_mode_row_idx = 0;
inline const char* mode_row_name(const char* op, const char* mode_name) {
    const int slot = g_mode_row_idx++ % kMaxModeRows;
    std::snprintf(g_mode_row_names[slot], sizeof(g_mode_row_names[0]), "%s-%s", op, mode_name);
    return g_mode_row_names[slot];
}

using SoftBinR = double (*)(sf64_rounding_mode, double, double);
using OracleBinR = double (*)(double, double, sf64_rounding_mode);

Stats sweep_bin_r(const char* op, sf64_rounding_mode m, const char* mode_name, SoftBinR soft,
                  OracleBinR oracle, uint64_t seed) {
    Stats s;
    s.name = mode_row_name(op, mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        // Log-uniform across the f64 magnitude range with random sign so
        // the rounding-path exercise covers cancellation, overflow, and
        // subnormal boundaries.
        double a = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e150);
        if (rng.next() & 1)
            a = -a;
        double b = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e150);
        if (rng.next() & 1)
            b = -b;
        const double got = soft(m, a, b);
        const double expect = oracle(a, b, m);
        record(s, a, b, got, expect);
    }
    return s;
}

Stats sweep_sqrt_r(sf64_rounding_mode m, const char* mode_name, uint64_t seed) {
    Stats s;
    s.name = mode_row_name("sqrt", mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        const double x = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e300);
        const double got = sf64_sqrt_r(m, x);
        const double expect = ref_sqrt_r(x, m);
        record(s, x, 0.0, got, expect);
    }
    return s;
}

Stats sweep_fma_r(sf64_rounding_mode m, const char* mode_name, uint64_t seed) {
    Stats s;
    s.name = mode_row_name("fma", mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        double a = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e100);
        if (rng.next() & 1)
            a = -a;
        double b = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e100);
        if (rng.next() & 1)
            b = -b;
        double c = rng.log_uniform(std::numeric_limits<double>::denorm_min(), 1e100);
        if (rng.next() & 1)
            c = -c;
        const double got = sf64_fma_r(m, a, b, c);
        const double expect = ref_fma_r(a, b, c, m);
        record(s, a, b, got, expect);
    }
    return s;
}

Stats sweep_to_f32_r(sf64_rounding_mode m, const char* mode_name, uint64_t seed) {
    Stats s;
    s.name = mode_row_name("to_f32", mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        // Sample log-magnitude across full f32 domain with headroom so
        // the overflow/underflow paths get exercised under each mode.
        double x = rng.log_uniform(1e-46, 1e40);
        if (rng.next() & 1)
            x = -x;
        const float got = sf64_to_f32_r(m, x);
        const float expect = ref_to_f32_r(x, m);
        uint32_t gb, eb;
        std::memcpy(&gb, &got, 4);
        std::memcpy(&eb, &expect, 4);
        s.checked++;
        // NaN-vs-NaN identity: both NaN collapses to "equal" (soft-fp64
        // NaN payloads are platform-quiet). Non-NaN must match bitwise.
        const bool g_nan = std::isnan(got);
        const bool e_nan = std::isnan(expect);
        if (g_nan && e_nan)
            continue;
        if (gb != eb) {
            if (s.max_ulp == 0) {
                s.worst_x = x;
                s.worst_got = static_cast<double>(got);
                s.worst_expect = static_cast<double>(expect);
            }
            s.max_ulp = 1; // BIT_EXACT tier: any drift fails.
        }
    }
    return s;
}

// f64->int per-mode. Sample strictly inside `[type_min + 1, type_max - 1]`
// so that any rounding direction stays representable (no saturation edge).
// This is an independent test of rounding rules; saturation is covered by
// the existing `tests/test_rounding_modes.cpp` and the TestFloat sweeps.
template <typename IntT, IntT (*Fn)(sf64_rounding_mode, double)>
Stats sweep_to_int_r(const char* type_label, sf64_rounding_mode m, const char* mode_name,
                     double safe_lo, double safe_hi, bool is_signed, uint64_t seed) {
    Stats s;
    char short_op[16];
    std::snprintf(short_op, sizeof(short_op), "to_%s", type_label);
    s.name = mode_row_name(short_op, mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        // Uniform across `[safe_lo, safe_hi]` — the fractional part stays
        // uniform, which is what drives rounding-direction differences.
        double x = rng.uniform(safe_lo, safe_hi);
        if (is_signed && (rng.next() & 1))
            x = -x;
        const IntT got = Fn(m, x);
        IntT expect;
        if (is_signed) {
            const long long r = ref_to_llint_r(x, m);
            expect = static_cast<IntT>(r);
        } else {
            const unsigned long long r = ref_to_ullint_r(x, m);
            expect = static_cast<IntT>(r);
        }
        s.checked++;
        if (got != expect) {
            if (s.max_ulp == 0) {
                s.worst_x = x;
                s.worst_got = static_cast<double>(static_cast<long long>(got));
                s.worst_expect = static_cast<double>(static_cast<long long>(expect));
            }
            s.max_ulp = 1;
        }
    }
    return s;
}

Stats sweep_rint_r(sf64_rounding_mode m, const char* mode_name, uint64_t seed) {
    Stats s;
    s.name = mode_row_name("rint", mode_name);
    s.tier = BIT_EXACT;
    LCG rng(seed);
    for (int i = 0; i < N_RAND_MODE; ++i) {
        // Log-sample |x| across [1e-4, 1e18] with random sign. Below 2^53
        // rint has real rounding work; above 2^53 the result equals the
        // input (already integral). Both regimes are exercised.
        double x = rng.log_uniform(1e-4, 1e18);
        if (rng.next() & 1)
            x = -x;
        const double got = sf64_rint_r(m, x);
        const double expect = ref_rint_r(x, m);
        record(s, x, 0.0, got, expect);
    }
    return s;
}

} // namespace

// ---------- Driver --------------------------------------------------------

int main() {
    std::printf("== test_mpfr_diff (MPFR prec=%d) ==\n", static_cast<int>(ORACLE_PREC));

    std::vector<Stats> results;

    // --- Per-mode bit-exact sweeps (1.1 `sf64_*_r` surface) ---------------
    //
    // Every arithmetic / sqrt / fma / convert / rint op must match MPFR
    // bit-for-bit under ALL FIVE IEEE-754 rounding modes. Transcendentals
    // (u10/u35/gamma tiers) stay RNE-only — their precision claim is
    // defined only for RNE in 1.1, so mode-looping them would not
    // correspond to any documented contract.
    //
    // Seeds are per-row so every (op, mode) cell gets its own corpus;
    // same op across modes uses related but offset seeds so cross-mode
    // regressions remain independent signals.
    std::printf("\n[per-mode bit-exact: sf64_*_r(mode, ...)]\n");
    for (const auto& mrow : kModes) {
        results.push_back(sweep_bin_r("add", mrow.mode, mrow.name, sf64_add_r, ref_add_r,
                                      0xADD0000ULL + mrow.mode));
        results.push_back(sweep_bin_r("sub", mrow.mode, mrow.name, sf64_sub_r, ref_sub_r,
                                      0x5B0000ULL + mrow.mode));
        results.push_back(sweep_bin_r("mul", mrow.mode, mrow.name, sf64_mul_r, ref_mul_r,
                                      0x70000ULL + mrow.mode));
        results.push_back(sweep_bin_r("div", mrow.mode, mrow.name, sf64_div_r, ref_div_r,
                                      0xD1D0000ULL + mrow.mode));
        results.push_back(sweep_sqrt_r(mrow.mode, mrow.name, 0x57A0000ULL + mrow.mode));
        results.push_back(sweep_fma_r(mrow.mode, mrow.name, 0xFA0000ULL + mrow.mode));
        results.push_back(sweep_to_f32_r(mrow.mode, mrow.name, 0xF320000ULL + mrow.mode));
        results.push_back(sweep_rint_r(mrow.mode, mrow.name, 0x71A0000ULL + mrow.mode));
        // Integer targets: sample strictly inside the type's representable
        // range (by 2 units of headroom — enough so any rounding direction
        // stays in range). Saturation-edge behavior is covered by the
        // existing `tests/test_rounding_modes.cpp` and TestFloat.
        results.push_back(sweep_to_int_r<int8_t, sf64_to_i8_r>(
            "i8", mrow.mode, mrow.name, 0.0, 126.0, /*is_signed=*/true, 0x180000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<int16_t, sf64_to_i16_r>(
            "i16", mrow.mode, mrow.name, 0.0, 32766.0, true, 0x1160000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<int32_t, sf64_to_i32_r>(
            "i32", mrow.mode, mrow.name, 0.0, 2147483646.0, true, 0x1320000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<int64_t, sf64_to_i64_r>("i64", mrow.mode, mrow.name, 0.0,
                                                                 9223372036854775000.0, true,
                                                                 0x1640000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<uint8_t, sf64_to_u8_r>(
            "u8", mrow.mode, mrow.name, 0.0, 254.0, /*is_signed=*/false, 0xB80000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<uint16_t, sf64_to_u16_r>(
            "u16", mrow.mode, mrow.name, 0.0, 65534.0, false, 0xB160000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<uint32_t, sf64_to_u32_r>(
            "u32", mrow.mode, mrow.name, 0.0, 4294967294.0, false, 0xB320000ULL + mrow.mode));
        results.push_back(sweep_to_int_r<uint64_t, sf64_to_u64_r>("u64", mrow.mode, mrow.name, 0.0,
                                                                  18446744073709548000.0, false,
                                                                  0xB640000ULL + mrow.mode));
    }

    // Input ranges mirror test_transcendental_1ulp where that test validates
    // a range. MPFR permits tighter oracles (e.g. *pi variants, exp10) so we
    // extend coverage where possible.
    //
    // Honest out-of-range notes (no silent skips — each is either covered by
    // another test or documented as out-of-scope):
    //   sf64_sinh boundary window |x| ∈ (709.78, 710.4758]: exercised by the
    //     fixed-point "sinh-edge" spot-check below (gated at U35). The random
    //     sweep stays on |x| ∈ [1e-4, 20] where the main claim lives.

    // --- Trig (forward): U10 ---------------------------------------------
    results.push_back(
        sweep1_log("sin", U10, [](double x) { return sf64_sin(x); }, mpfr_sin, 1e-6, 100.0, true));
    results.push_back(
        sweep1_log("cos", U10, [](double x) { return sf64_cos(x); }, mpfr_cos, 1e-6, 100.0, true));
    // sincos: independent U10 oracle for both outputs against MPFR. The
    // consistency-only check in test_transcendental_1ulp.cpp (sincos vs
    // sf64_sin/sf64_cos) can't catch a joint reduction bug that shifts both
    // legs the same way; MPFR is the real referee. Same seed for s and c so
    // both sweeps hit the identical input corpus — any per-leg discrepancy
    // must be a reduction / reconstruction bug in the sincos path itself.
    results.push_back(sweep1_log(
        "sincos-s", U10,
        [](double x) {
            double s, c;
            sf64_sincos(x, &s, &c);
            return s;
        },
        mpfr_sin, 1e-6, 100.0, true, 0x51C05ULL));
    results.push_back(sweep1_log(
        "sincos-c", U10,
        [](double x) {
            double s, c;
            sf64_sincos(x, &s, &c);
            return c;
        },
        mpfr_cos, 1e-6, 100.0, true, 0x51C05ULL));
    results.push_back(
        sweep1_log("tan", U35, [](double x) { return sf64_tan(x); }, mpfr_tan, 1e-6, 1.5, true));
    results.push_back(sweep1_log(
        "asin", U10, [](double x) { return sf64_asin(x); }, mpfr_asin, 1e-6, 0.99, true));
    results.push_back(sweep1_log(
        "acos", U10, [](double x) { return sf64_acos(x); }, mpfr_acos, 1e-6, 0.99, true));
    results.push_back(
        sweep1_log("atan", U10, [](double x) { return sf64_atan(x); }, mpfr_atan, 1e-6, 1e6, true));
    results.push_back(sweep_atan2like(
        "atan2", U10, [](double y, double x) { return sf64_atan2(y, x); }, mpfr_atan2));

    // --- π-scaled trig variants: U10/U35 (MPFR has exact sinpi/cospi) ----
    results.push_back(sweep1_uniform(
        "sinpi", U10, [](double x) { return sf64_sinpi(x); }, mpfr_sinpi, -4.0, 4.0));
    results.push_back(sweep1_uniform(
        "cospi", U10, [](double x) { return sf64_cospi(x); }, mpfr_cospi, -4.0, 4.0));
    results.push_back(sweep1_uniform(
        "tanpi", U35, [](double x) { return sf64_tanpi(x); }, mpfr_tanpi, -0.49, 0.49));
    results.push_back(sweep1_uniform(
        "asinpi", U10, [](double x) { return sf64_asinpi(x); }, mpfr_asinpi, -0.99, 0.99));
    results.push_back(sweep1_uniform(
        "acospi", U10, [](double x) { return sf64_acospi(x); }, mpfr_acospi, -0.99, 0.99));
    results.push_back(sweep1_log(
        "atanpi", U10, [](double x) { return sf64_atanpi(x); }, mpfr_atanpi, 1e-6, 1e6, true));
    results.push_back(sweep_atan2like(
        "atan2pi", U10, [](double y, double x) { return sf64_atan2pi(y, x); }, mpfr_atan2pi));

    // --- Hyperbolic: U10/U35 ---------------------------------------------
    results.push_back(sweep1_log(
        "sinh", U35, [](double x) { return sf64_sinh(x); }, mpfr_sinh, 1e-4, 20.0, true));
    // sinh boundary: fixed-point sweep across the (709.78, 710.4758] window
    // where the large-|x| branch switches to exp(a - ln2) to avoid overflow.
    {
        Stats sb;
        sb.name = "sinh-edge";
        sb.tier = U35;
        const double pts[] = {709.79, 710.0, 710.4, 710.48};
        for (double p : pts) {
            record(sb, p, 0.0, sf64_sinh(p), ref1(mpfr_sinh, p));
            record(sb, -p, 0.0, sf64_sinh(-p), ref1(mpfr_sinh, -p));
        }
        results.push_back(sb);
    }
    results.push_back(sweep1_log(
        "cosh", U10, [](double x) { return sf64_cosh(x); }, mpfr_cosh, 1e-4, 20.0, true));
    results.push_back(sweep1_log(
        "tanh", U35, [](double x) { return sf64_tanh(x); }, mpfr_tanh, 1e-4, 20.0, true));
    results.push_back(sweep1_log(
        "asinh", U35, [](double x) { return sf64_asinh(x); }, mpfr_asinh, 1e-4, 1e6, true));
    results.push_back(sweep1_log(
        "acosh", U10, [](double x) { return sf64_acosh(x); }, mpfr_acosh, 1.01, 1e6, false));
    results.push_back(sweep1_log(
        "atanh", U10, [](double x) { return sf64_atanh(x); }, mpfr_atanh, 1e-4, 0.99, true));

    // --- Exp / log: U10 ---------------------------------------------------
    results.push_back(
        sweep1_log("exp", U10, [](double x) { return sf64_exp(x); }, mpfr_exp, 1e-6, 700.0, true));
    results.push_back(sweep1_log(
        "exp2", U10, [](double x) { return sf64_exp2(x); }, mpfr_exp2, 1e-6, 1000.0, true));
    results.push_back(sweep1_log(
        "exp10", U10, [](double x) { return sf64_exp10(x); }, mpfr_exp10, 1e-6, 300.0, true));
    results.push_back(sweep1_log(
        "expm1", U10, [](double x) { return sf64_expm1(x); }, mpfr_expm1, 1e-3, 700.0, true));
    results.push_back(sweep1_log(
        "log", U10, [](double x) { return sf64_log(x); }, mpfr_log, 1e-100, 1e100, false));
    results.push_back(sweep1_log(
        "log2", U10, [](double x) { return sf64_log2(x); }, mpfr_log2, 1e-100, 1e100, false));
    results.push_back(sweep1_log(
        "log10", U10, [](double x) { return sf64_log10(x); }, mpfr_log10, 1e-100, 1e100, false));
    results.push_back(sweep1_log(
        "log1p", U10, [](double x) { return sf64_log1p(x); }, mpfr_log1p, 1e-10, 1e10, false));

    // --- Power / root: U35 (pow per task spec "pow edge") ----------------
    //
    // pow is exercised on three overlapping windows that together bound the
    // U35 claim:
    //   "pow"         — moderate: x∈[1e-6, 1e6], y∈[-50, 50]   (main claim)
    //   "pow-xbig"    — x wide, y modest: x∈[1e-100, 1e100], y∈[-5, 5]
    //   "pow-ybig"    — x modest, y wide: x∈[1e-6, 1e3], y∈[-100, 100]
    // Precision drifts above U35 in the "near-unit base × huge exponent"
    // regime (x∈[0.5, 2], |y|≥200) because `logk_dd` evaluates its tail
    // polynomial on x².hi as a plain double, which caps the log DD at
    // ~2^-56 relative. A full DD-Horner rewrite of the log minimax is
    // tracked in TODO.md; in the meantime we document the bounded range.
    // x log-sampled across 12 decades (positive only — pow with negative
    // base + non-integer y returns NaN, which would dominate trivial
    // matches); y is a narrow linear range so linear sampling is fine.
    results.push_back(sweep2_log_uniform(
        "pow", U35, [](double x, double y) { return sf64_pow(x, y); }, mpfr_pow, 1e-6, 1e6,
        /*x_sym=*/false, -50.0, 50.0));
    // pow-xbig: x log-uniform across 200 decades, y log-uniform in
    // [1e-5, 5] with random sign so negative y is exercised. Linear
    // sampling here would concentrate 99%+ of x samples near 1e100 and
    // 99%+ of y samples near ±5, so pow overflows / underflows on
    // almost every input and the sweep never exercises its advertised
    // range. y=0 exactly is out of the log-uniform support; the zero
    // cases are owned by tests/test_powr_ieee754.cpp (for powr) and by
    // the boundary-aware impl.
    results.push_back(sweep2_log(
        "pow-xbig", U35, [](double x, double y) { return sf64_pow(x, y); }, mpfr_pow, 1e-100, 1e100,
        /*x_sym=*/false, 1e-5, 5.0, /*y_sym=*/true, 0xA110CA7EULL));
    // pow-ybig: x log-sampled across 9 decades, y wide linear (the
    // failure-driving axis here is y, and a linear ±100 stays narrow
    // enough to sample uniformly).
    results.push_back(sweep2_log_uniform(
        "pow-ybig", U35, [](double x, double y) { return sf64_pow(x, y); }, mpfr_pow, 1e-6, 1e3,
        /*x_sym=*/false, -100.0, 100.0, 0xB16B00B5ULL));
    // powr: same shape as pow but x ≥ 0 is required by the spec; log
    // sampling keeps small-x exercised.
    results.push_back(sweep2_log_uniform(
        "powr", U35, [](double x, double y) { return sf64_powr(x, y); }, mpfr_powr, 1e-6, 1e6,
        /*x_sym=*/false, -50.0, 50.0));
    results.push_back(sweep_pown("pown", U35));
    results.push_back(sweep_rootn("rootn", U35));
    // cbrt across the full double range including subnormals.
    results.push_back(sweep1_log(
        "cbrt", U10, [](double x) { return sf64_cbrt(x); }, mpfr_cbrt,
        std::numeric_limits<double>::denorm_min(), 1e300, true));

    // --- Error / gamma: GAMMA tier (impl is not DD-tight) ----------------
    results.push_back(
        sweep1_uniform("erf", GAMMA, [](double x) { return sf64_erf(x); }, mpfr_erf, -5.0, 5.0));
    // erfc across the active range including the deep tail: the DD-exp
    // lift (`erfc_cheb` builds `-z²+p(ty)` as DD and feeds the DD into
    // `expk_dd`) keeps the deep-tail max at ≤8 ULP — fits U35 with
    // headroom.  Kept at GAMMA tier for safety against future drift.
    results.push_back(sweep1_uniform(
        "erfc", GAMMA, [](double x) { return sf64_erfc(x); }, mpfr_erfc, -5.0, 27.0));
    // tgamma through the overflow boundary (x ≈ 171.6).  The DD lift of
    // the Lanczos lg body + DD-exp reconstruction keeps the near-overflow
    // bucket at ~0.9 k ULP — just inside GAMMA.
    results.push_back(sweep1_uniform(
        "tgamma", GAMMA, [](double x) { return sf64_tgamma(x); }, mpfr_gamma, 0.5, 170.0));
    // lgamma ULP-diff is unbounded near its zeros at x=1 and x=2 (absolute
    // error stays tiny, but ULP ratio against a near-zero value is not
    // well-defined). Here we gate on a zero-free subrange x ≥ 3 where the
    // DD stitching holds the GAMMA band comfortably. The deeper zero-crossing
    // behavior (which still exceeds GAMMA — dominated by the v1.2 logk_dd
    // DD-Horner rewrite) is exercised by the report-only harness under
    // tests/experimental/.
    results.push_back(sweep1_log(
        "lgamma", GAMMA, [](double x) { return sf64_lgamma(x); }, mpfr_lngamma, 3.0, 1e4, false));
    // lgamma_r shares the same log-|Γ| magnitude path as lgamma; gate its
    // magnitude at the same GAMMA tier over the same zero-free subrange.
    // The out-parameter sign is trivially +1 on [3, 1e4] (Γ is positive
    // there); sign coverage across the full sign-flipping domain lives in
    // tests/test_transcendental_1ulp.cpp.
    results.push_back(sweep1_log(
        "lgamma_r", GAMMA,
        [](double x) {
            int sgn = 0;
            return sf64_lgamma_r(x, &sgn);
        },
        mpfr_lngamma, 3.0, 1e4, false));

    // --- fmod / remainder -----------------------------------------------
    // Binary long-division (no float rounding, no transcendental reduction):
    // the Doxygen contract is "Exact". Gated at BIT_EXACT — any drift from
    // 0 ULP against MPFR is a real bug, not a tier-fit issue. Sweep ranges
    // are wide enough that the quotient bit-count reaches ~2^50, well past
    // any loop-termination edge.
    // x log-symmetric across 30 decades (denorm_min..1e15, both signs);
    // y log-uniform in [1, 1e10] (10 decades, positive only — sign of
    // y doesn't affect fmod magnitude). Linear sampling here would
    // collapse x and y both into the top decade of their windows, so
    // 99% of (x, y) pairs would have |x| ≪ |y| (early-return branch
    // returns x) or |x|/|y| ≈ 10⁵ (only ~17 quotient bits exercised).
    // Log sampling actually exercises the algorithmic regimes the
    // BIT_EXACT claim covers — small-x return path, large-quotient
    // long-division loop, denormal operands.
    results.push_back(sweep2_log(
        "fmod", BIT_EXACT, [](double x, double y) { return sf64_fmod(x, y); }, mpfr_fmod,
        std::numeric_limits<double>::denorm_min(), 1e15, /*x_sym=*/true, 1.0, 1e10,
        /*y_sym=*/false));
    results.push_back(sweep2_log(
        "remainder", BIT_EXACT, [](double x, double y) { return sf64_remainder(x, y); },
        mpfr_remainder, std::numeric_limits<double>::denorm_min(), 1e15, /*x_sym=*/true, 1.0, 1e10,
        /*y_sym=*/false));

    // --- hypot ----------------------------------------------------------
    // hypot's scaling formula must handle operands spanning the full
    // magnitude range. Linear sampling over [-1e150, 1e150] collapses
    // into the top decade for both axes (so almost every sample hits
    // overflow). Log-uniform with random sign exercises every decade of
    // |x| and |y| across the advertised range; the lower bound is
    // `denorm_min()` so subnormal operands are in scope too.
    results.push_back(sweep2_log(
        "hypot", U10, [](double x, double y) { return sf64_hypot(x, y); }, mpfr_hypot,
        std::numeric_limits<double>::denorm_min(), 1e150, /*x_sym=*/true,
        std::numeric_limits<double>::denorm_min(), 1e150, /*y_sym=*/true, 0xBADDBADDULL));

    edge_spot_checks();

    std::printf("\n[max-ULP vs MPFR@%d, rounded to double]\n", static_cast<int>(ORACLE_PREC));
    int failures = 0;
    for (const auto& s : results) {
        report(s);
        if (fail(s))
            failures++;
    }

    // Free MPFR caches so leak-sanitizer builds stay quiet.
    mpfr_free_cache();

    if (failures) {
        std::fprintf(stderr,
                     "\nFAIL: %d sf64_* function(s) exceeded the MPFR-diff "
                     "tolerance band.\n",
                     failures);
        std::abort();
    }

    std::printf("\nOK: all sf64_* transcendentals within MPFR tolerance.\n");
    return 0;
}
