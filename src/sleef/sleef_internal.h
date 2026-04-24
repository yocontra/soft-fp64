#pragma once
//
// Private header shared across the per-family SLEEF TUs
// (`sleef_exp_log.cpp`, `sleef_trig.cpp`, `sleef_inv_hyp_pow.cpp`,
// `sleef_stubs.cpp`). Holds constants and helpers used in more than one
// translation unit, plus forward decls for the `sf64_internal_exp_core` /
// `sf64_internal_log_core` cores that pow / hyperbolic / cbrt pull in.
// The cores are hidden-visibility on ELF/Mach-O — they must never escape
// the archive as public ABI.
//
// SPDX-License-Identifier: BSL-1.0 AND MIT
//

#include "sleef_common.h"

#include "../../include/soft_fp64/soft_f64.h"

namespace soft_fp64::sleef::detail {

// ---- shared constants (SLEEF 3.6 sleefdp.c) -----------------------------

inline constexpr double kPI = 3.141592653589793238462643383279502884;
inline constexpr double kPI_2 = 1.570796326794896619231321691639751442;
inline constexpr double kPI_4 = 0.785398163397448309615660845819875721;
inline constexpr double k2_PI = 0.636619772367581343075535053490057448;
inline constexpr double kM_2_PI_H = 0.63661977236758138243;

// Cody-Waite π splitting — SLEEF PI_A / PI_B / PI_C / PI_D.
inline constexpr double kPI_A = 3.1415926218032836914;
inline constexpr double kPI_B = 3.1786509424591713469e-08;
inline constexpr double kPI_C = 1.2246467864107188502e-16;
inline constexpr double kPI_D = 1.2736634327021899816e-24;

// PI_A2/B2 — for the smaller range (|x| < 15).
inline constexpr double kPI_A2 = 3.141592653589793116;
inline constexpr double kPI_B2 = 1.2246467991473532072e-16;

inline constexpr double kTRIGRANGEMAX = 1e14;
inline constexpr double kTRIGRANGEMAX2 = 15.0;

inline constexpr double kL2U = 0.69314718055966295651160180568695068359375;
inline constexpr double kL2L = 0.28235290563031577122588448175013436025525412068e-12;
inline constexpr double kR_LN2 =
    1.442695040888963407359924681001892137426645954152985934135449406931;

inline constexpr double kInf = __builtin_huge_val();

// ---- canonical quiet NaN -----------------------------------------------

SF64_ALWAYS_INLINE double qNaN() noexcept {
    // SAFETY: constant bit pattern for a canonical quiet NaN (exp=all 1s,
    // MSB of mantissa set, payload bits zero, sign=0). Bit-reinterpret only.
    return from_bits(0x7FF8000000000000ULL);
}

// ---- integer predicates (no host FPU arithmetic in the body) ------------

SF64_ALWAYS_INLINE bool is_int(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    return soft_fp64::sleef::eq_(soft_fp64::sleef::sub_(x, sf64_trunc(x), fe), 0.0);
}

SF64_ALWAYS_INLINE bool is_odd_int(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    if (!is_int(x, fe))
        return false;
    const double half = soft_fp64::sleef::mul_(x, 0.5, fe);
    return soft_fp64::sleef::ne_(soft_fp64::sleef::sub_(half, sf64_trunc(half), fe), 0.0);
}

} // namespace soft_fp64::sleef::detail

// ---- cross-TU SLEEF cores ----------------------------------------------
//
// `sf64_internal_exp_core` and `sf64_internal_log_core` are the raw cores
// used by exp/exp2/exp10/expm1 and log/log2/log10/log1p respectively. They
// are also pulled in by pow, cbrt, asinh, acosh, atanh — hence non-static
// and declared here so the consuming TUs can link against them.
//
// The `sf64_internal_` prefix and hidden visibility make it a link error
// for any external consumer to take a dependency on these symbols. If the
// visibility attribute is dropped, the `install-smoke` CI job's `nm -g`
// check will fail on the archive.

namespace soft_fp64::sleef {

// Cross-TU internals take the caller's stack-local fenv accumulator by
// reference so the flag raise never rotates through TLS inside the hot
// transcendental inner loops. The SLEEF public entry that invoked them
// flushes once at return.
[[gnu::visibility("hidden")]] double sf64_internal_exp_core(double d, sf64_internal_fe_acc& fe);
[[gnu::visibility("hidden")]] double sf64_internal_log_core(double d, sf64_internal_fe_acc& fe);

// DD-carrying exp/log cores. `logk_dd(d)` returns log|d| as a DD pair
// (target ≈ 2^-106 relative); `expk_dd(d)` takes a DD argument and returns
// exp(d) as a plain double with the DD accuracy carried through reduction.
//
// These are the SLEEF 3.6 `logk` and `expk` families, lifted out of their
// former anonymous namespace in `sleef_inv_hyp_pow.cpp` so `sleef_stubs.cpp`
// (erfc deep tail, tgamma near-overflow) can build its exp argument in DD
// too. Hidden visibility so they stay out of the public ABI — the `nm -g`
// CI check on `install-smoke` depends on this.
[[gnu::visibility("hidden")]] DD sf64_internal_logk_dd(double d, sf64_internal_fe_acc& fe);
[[gnu::visibility("hidden")]] double sf64_internal_expk_dd(DD d, sf64_internal_fe_acc& fe);

} // namespace soft_fp64::sleef
