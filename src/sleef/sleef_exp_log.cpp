//
// Derived from SLEEF 3.6 `src/libm/sleefdp.c` (Boost-1.0); see
// `src/sleef/NOTICE` for the upstream URL, the pinned commit SHA, and the
// full Boost-1.0 license text.
//
// exp / log family scalar transcendentals with SLEEF minimax polynomials
// and double-double (DD) accumulation in the tight-cancellation branches.
// Every `+`, `-`, `*`, `/`, `fma`, `sqrt`, `ldexp` in this file is a call
// into the `sf64_*` ABI. No host FPU arithmetic appears in any function
// body (literal constants are fine — the compiler folds them at
// translation time).
//
// Owns: sf64_exp / sf64_exp2 / sf64_exp10 / sf64_expm1 /
//       sf64_log / sf64_log2 / sf64_log10 / sf64_log1p
//
// Exports `sf64_internal_exp_core` and `sf64_internal_log_core` (see
// `sleef_internal.h`) so that sf64_pow / sf64_cbrt / sf64_asinh /
// sf64_acosh / sf64_atanh can build on them without going through the
// public ABI. Both carry `[[gnu::visibility("hidden")]]` — they are link-
// time private to the archive; the `install-smoke` CI job's `nm -g`
// check fails if that invariant regresses.
//
// SPDX-License-Identifier: BSL-1.0 AND MIT
//

#include "sleef_internal.h"
// IMPORTANT: must come after sleef_internal.h — rewrites `sf64_*(...)`,
// the DD primitives, and the `*_core` cross-TU cores to their fe-threaded
// forms. Each helper and entry must have a local `fe` accumulator in scope.
#include "sleef_fe_macros.h"

using soft_fp64::sleef::bits_of;
using soft_fp64::sleef::DD;
using soft_fp64::sleef::dd;
using soft_fp64::sleef::dd_to_d;
using soft_fp64::sleef::ddadd2_dd_d_d;
using soft_fp64::sleef::ddadd2_dd_d_dd;
using soft_fp64::sleef::ddadd2_dd_dd;
using soft_fp64::sleef::ddadd2_dd_dd_d;
using soft_fp64::sleef::dddiv_dd_dd_dd;
using soft_fp64::sleef::ddmul_dd_d_d;
using soft_fp64::sleef::ddmul_dd_dd_d;
using soft_fp64::sleef::ddmul_dd_dd_dd;
using soft_fp64::sleef::ddneg_dd_dd;
using soft_fp64::sleef::ddnormalize_dd_dd;
using soft_fp64::sleef::ddscale_dd_dd_d;
using soft_fp64::sleef::ddsqu_dd_dd;
using soft_fp64::sleef::eq_;
using soft_fp64::sleef::from_bits;
using soft_fp64::sleef::ge_;
using soft_fp64::sleef::gt_;
using soft_fp64::sleef::isinf_;
using soft_fp64::sleef::isnan_;
using soft_fp64::sleef::le_;
using soft_fp64::sleef::lt_;
using soft_fp64::sleef::mla;
using soft_fp64::sleef::ne_;
using soft_fp64::sleef::rint_;
using soft_fp64::sleef::detail::kInf;
using soft_fp64::sleef::detail::kL2L;
using soft_fp64::sleef::detail::kL2U;
using soft_fp64::sleef::detail::kR_LN2;
using soft_fp64::sleef::detail::qNaN;

// ========================================================================
// TU-local helpers: SLEEF-style bit-field exponent extraction / injection.
// Pure bit manipulation — no FP arithmetic — so they satisfy the sf64_*
// ABI contract trivially.
// ========================================================================

namespace {

// ilogb2k(d) = unbiased exponent field of |d|.  SLEEF source:
//   (int)((doubleToRawLongBits(d) >> 52) & 0x7ff) - 0x3ff
SF64_ALWAYS_INLINE int ilogb2k_bits(double d) noexcept {
    const uint64_t b = bits_of(d);
    const int e = static_cast<int>((b >> 52) & 0x7ff) - 0x3ff;
    return e;
}

// ldexp2k / ldexp3k: scale by 2^q via sf64_ldexp. The two
// flavours differ in SLEEF by how large |q| can be before an intermediate
// underflow — but sf64_ldexp handles the full range already.
SF64_ALWAYS_INLINE double ldexp2k_f(double d, int q) noexcept {
    return sf64_ldexp(d, q);
}

SF64_ALWAYS_INLINE double ldexp3k_f(double d, int q) noexcept {
    return sf64_ldexp(d, q);
}

// ---- SLEEF minimax polynomial coefficients (verbatim from sleefdp.c) ----
//
// Order below matches SLEEF's `POLY10(..., c9, c8, …, c0)` convention: the
// highest-degree coefficient is first so `poly_array` (Horner highest-first)
// consumes them directly.

// xexp: degree-10 minimax in s over |s| <= ln(2)/2 (c9 … c0).
constexpr double kExpPoly[10] = {
    2.08860621107283687536341e-09, 2.51112930892876518610661e-08, 2.75573911234900471893338e-07,
    2.75572362911928827629423e-06, 2.4801587159235472998791e-05,  0.000198412698960509205564975,
    0.00138888888889774492207962,  0.00833333333331652721664984,  0.0416666666666665047591422,
    0.166666666666666851703837,
};

// xexp2: degree-10 minimax in s over |s| <= 0.5 (c9 … c0). Leading
// coefficient is the degree-10 term; the degree-0 term (ln2) is applied
// separately via a final fma that is done in DD (SLEEF does `u = mla(u, s,
// 0.6931…)` then combines with `ddadd + ddmul + ldexp2k`).
constexpr double kExp2Poly[10] = {
    +0.4434359082926529454e-9, +0.7073164598085707425e-8, +0.1017819260921760451e-6,
    +0.1321543872511327615e-5, +0.1525273353517584730e-4, +0.1540353045101147808e-3,
    +0.1333355814670499073e-2, +0.9618129107597600536e-2, +0.5550410866482046596e-1,
    +0.2402265069591012214e+0,
};

// xexp10: degree-10 minimax in s over |s| <= log10(2)/2 (c9 … c0). Leading
// coefficient is the degree-10 term. The degree-0 constant (ln 10) is
// combined via SLEEF's final DD fma.
constexpr double kExp10Poly[10] = {
    +0.2411463498334267652e-3, +0.1157488415217187375e-2, +0.5013975546789733659e-2,
    +0.1959762320720533080e-1, +0.6808936399446784138e-1, +0.2069958494722676234e+0,
    +0.5393829292058536229e+0, +0.1171255148908541655e+1, +0.2034678592293432953e+1,
    +0.2650949055239205876e+1,
};

// xlog_u1 / xlog1p: degree-7 minimax in x² over the reduced argument
// (m-1)/(m+1) (c6 … c0). Same coefficients reused by log1p.
constexpr double kLogPoly[7] = {
    0.1532076988502701353e+0, 0.1525629051003428716e+0, 0.1818605932937785996e+0,
    0.2222214519839380009e+0, 0.2857142932794299317e+0, 0.3999999999635251990e+0,
    0.6666666666667333541e+0,
};

// xlog2: degree-7 minimax in x² (c6 … c0).
constexpr double kLog2Poly[7] = {
    +0.2211941750456081490e+0, +0.2200768693152277689e+0, +0.2623708057488514656e+0,
    +0.3205977477944495502e+0, +0.4121985945485324709e+0, +0.5770780162997058982e+0,
    +0.96179669392608091449,
};

// xlog10: degree-7 minimax in x² (c6 … c0).
constexpr double kLog10Poly[7] = {
    +0.6653725819576758460e-1, +0.6625722782820833712e-1, +0.7898105214313944078e-1,
    +0.9650955035715275132e-1, +0.1240841409721444993e+0, +0.1737177927454605086e+0,
    +0.2895296546021972617e+0,
};

// Constants matching SLEEF's misc.h:
//   L10U  = 0.30102999566383914498       // log_10(2) high
//   L10L  = 1.4205023227266099418e-13    // log_10(2) low
//   LOG10_2 = 3.3219280948873623478703   // log2(10)
constexpr double kL10U = 0.30102999566383914498;
constexpr double kL10L = 1.4205023227266099418e-13;
constexpr double kLog10_2 = 3.3219280948873623478703194294893901758648313930;
constexpr double kLN2 = 0.693147180559945286226764;        // DD hi
constexpr double kLN2_LO = 2.319046813846299558417771e-17; // DD lo

// SLEEF's Horner eval for a POLY10 expressed as nested fmas (all via sf64_fma).
SF64_ALWAYS_INLINE double poly10_horner(double x, const double c[10],
                                        soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    double u = c[0];
    u = mla(u, x, c[1]);
    u = mla(u, x, c[2]);
    u = mla(u, x, c[3]);
    u = mla(u, x, c[4]);
    u = mla(u, x, c[5]);
    u = mla(u, x, c[6]);
    u = mla(u, x, c[7]);
    u = mla(u, x, c[8]);
    u = mla(u, x, c[9]);
    return u;
}

SF64_ALWAYS_INLINE double poly7_horner(double x, const double c[7],
                                       soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    double u = c[0];
    u = mla(u, x, c[1]);
    u = mla(u, x, c[2]);
    u = mla(u, x, c[3]);
    u = mla(u, x, c[4]);
    u = mla(u, x, c[5]);
    u = mla(u, x, c[6]);
    return u;
}

// expk2: e^d for d given as a DD. Used by expm1.
// Direct port of SLEEF's `expk2(Sleef_double2 d)`.
DD expk2(DD d, soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    // q = round((d.x + d.y) * R_LN2)
    const double sum = sf64_add(d.hi, d.lo);
    const double qf = rint_(sf64_mul(sum, kR_LN2));
    const int q = sf64_to_i32(qf);

    // s = d - q*L2U - q*L2L  (in DD)
    DD s = ddadd2_dd_dd_d(d, sf64_mul(qf, sf64_neg(kL2U)));
    s = ddadd2_dd_dd_d(s, sf64_mul(qf, sf64_neg(kL2L)));

    // Horner: u = Σ c_i * s.x^i, i = 2..11  (SLEEF computes up through
    // degree 11 for expk2 using 10 fma's starting at 1/11!).
    double u = +0.1602472219709932072e-9;
    u = mla(u, s.hi, +0.2092255183563157007e-8);
    u = mla(u, s.hi, +0.2505230023782644465e-7);
    u = mla(u, s.hi, +0.2755724800902135303e-6);
    u = mla(u, s.hi, +0.2755731892386044373e-5);
    u = mla(u, s.hi, +0.2480158735605815065e-4);
    u = mla(u, s.hi, +0.1984126984148071858e-3);
    u = mla(u, s.hi, +0.1388888888886763255e-2);
    u = mla(u, s.hi, +0.8333333333333347095e-2);
    u = mla(u, s.hi, +0.4166666666666669905e-1);

    // t = s*u + 1/6!    (degree-6 coefficient, 1/720 = 0.1388...e-2)
    //  followed by the DD-precise unrolling of
    //    t = 1 + s + s²*(1/2 + s*(1/6 + s²*u_poly))
    DD t = ddadd2_dd_dd_d(ddmul_dd_dd_d(s, u), +0.1666666666666666574e+0);
    t = ddadd2_dd_dd_d(ddmul_dd_dd_dd(s, t), 0.5);
    t = ddadd2_dd_dd(s, ddmul_dd_dd_dd(ddsqu_dd_dd(s), t));
    t = ddadd2_dd_d_dd(1.0, t);

    // Scale by 2^q (apply to both limbs of the DD — ldexp is lossless).
    t.hi = ldexp2k_f(t.hi, q);
    t.lo = ldexp2k_f(t.lo, q);

    // Very negative argument: underflow to 0.
    if (lt_(d.hi, -1000.0))
        return DD{0.0, 0.0};
    return t;
}

} // namespace

// ========================================================================
// Public / cross-TU cores
// ========================================================================

namespace soft_fp64::sleef {

// ----------------------------------------------------------------------
// sf64_internal_exp_core — SLEEF xexp (u10) — degree-10 minimax.
// ----------------------------------------------------------------------
[[gnu::visibility("hidden")]] double sf64_internal_exp_core(double d, sf64_internal_fe_acc& fe) {
    if (isnan_(d))
        return qNaN();
    if (gt_(d, 709.782712893383996732223))
        return kInf; // overflow
    if (lt_(d, -1000.0))
        return 0.0; // hard underflow

    // q = round(d * R_LN2)
    const double qf = rint_(sf64_mul(d, kR_LN2));
    const int q = sf64_to_i32(qf);

    // Cody-Waite argument reduction in two steps:  s = d - q*L2U - q*L2L.
    double s = sf64_fma(qf, sf64_neg(kL2U), d);
    s = sf64_fma(qf, sf64_neg(kL2L), s);

    // SLEEF minimax: u = c9 + s*(c8 + s*(…)) for the coefficients 1/11!
    // down through 1/3! (xexp runs Horner through degree 10).
    double u = poly10_horner(s, kExpPoly, fe);
    // u = mla(u, s, 0.5)   — fold in the 1/2 coefficient.
    u = mla(u, s, 0.5);
    // u = s*s*u + s + 1    — SLEEF's final reconstruction identity.
    u = sf64_fma(sf64_mul(s, s), u, sf64_add(s, 1.0));

    // Rescale by 2^q.
    double r = sf64_ldexp(u, q);

    if (gt_(d, 709.0) && (isinf_(r) || gt_(r, 1.7976931348623157e+308)))
        return kInf;
    return r;
}

// ----------------------------------------------------------------------
// sf64_internal_log_core — SLEEF xlog_u1 (u10) — degree-7 minimax + DD re-assembly.
// ----------------------------------------------------------------------
[[gnu::visibility("hidden")]] double sf64_internal_log_core(double d, sf64_internal_fe_acc& fe) {
    if (isnan_(d) || lt_(d, 0.0))
        return qNaN();
    if (eq_(d, 0.0))
        return sf64_neg(kInf);
    if (isinf_(d))
        return kInf;

    // Subnormal scaling: if d < 2^-1022, boost it by 2^64 to keep frexp /
    // ilogb accurate, then subtract 64 from the exponent after the poly.
    constexpr double kDblMin = 2.2250738585072014e-308;
    constexpr double kScale = 4.294967296e9 * 4.294967296e9; // 2^64 (folded)
    bool subnormal_bump = false;
    if (lt_(d, kDblMin)) {
        d = sf64_mul(d, kScale);
        subnormal_bump = true;
    }

    // e = unbiased exponent of (d * 4/3) — this centres m into
    // [0.75, 1.5) rather than [1, 2), which shortens the polynomial range.
    int e = ilogb2k_bits(sf64_mul(d, 1.3333333333333333));
    double m = ldexp3k_f(d, -e);
    if (subnormal_bump)
        e -= 64;

    // x = (m - 1) / (m + 1)  as a DD to avoid cancellation near m = 1.
    const DD xnum = ddadd2_dd_d_d(-1.0, m);
    const DD xden = ddadd2_dd_d_d(1.0, m);
    const DD x = dddiv_dd_dd_dd(xnum, xden);

    // x² as a plain double (sufficient since x is small near m=1).
    const double x2 = sf64_mul(x.hi, x.hi);

    // t = POLY7(x², …)  (SLEEF kLogPoly coefficients, c6..c0).
    const double t = poly7_horner(x2, kLogPoly, fe);

    // s = ln(2) * e  (carried as DD)  +  2*x  +  x² * x * t
    DD s = ddmul_dd_dd_d(DD{kLN2, kLN2_LO}, sf64_from_i32(e));
    s = ddadd2_dd_dd(s, ddscale_dd_dd_d(x, 2.0));
    s = ddadd2_dd_dd_d(s, sf64_mul(sf64_mul(x2, x.hi), t));

    return dd_to_d(s);
}

} // namespace soft_fp64::sleef

// ========================================================================
// Public C ABI entry points.
// ========================================================================

using soft_fp64::sleef::sf64_internal_exp_core;
using soft_fp64::sleef::sf64_internal_log_core;

extern "C" double sf64_exp(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double r = sf64_internal_exp_core(x, fe);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_exp2 — SLEEF xexp2 (u10) native impl.
// ----------------------------------------------------------------------
extern "C" double sf64_exp2(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (ge_(x, 1024.0))
        return kInf;
    if (le_(x, -2000.0))
        return 0.0;

    // q = round(x), s = x - q    (|s| <= 0.5)
    const double qf = rint_(x);
    const int q = sf64_to_i32(qf);
    const double s = sf64_sub(x, qf);

    // u = POLY10(s, …)   then fold in the degree-1 ln(2) coefficient.
    double u = poly10_horner(s, kExp2Poly, fe);
    u = mla(u, s, +0.6931471805599452862e+0);

    // t = 1 + u*s, all in DD, then renormalize; SLEEF's trick for the 1.0.
    DD t = ddadd2_dd_d_dd(1.0, ddmul_dd_d_d(u, s));
    double r = ddnormalize_dd_dd(t).hi;

    r = sf64_ldexp(r, q);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_exp10 — SLEEF xexp10 (u10) native impl.
// ----------------------------------------------------------------------
extern "C" double sf64_exp10(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (gt_(x, 308.25471555991671))
        return kInf;
    if (lt_(x, -350.0))
        return 0.0;

    // q = round(x * log2(10))
    const double qf = rint_(sf64_mul(x, kLog10_2));
    const int q = sf64_to_i32(qf);

    // s = x - q*L10U - q*L10L  (Cody-Waite in base 10).
    double s = sf64_fma(qf, sf64_neg(kL10U), x);
    s = sf64_fma(qf, sf64_neg(kL10L), s);

    // u = POLY10(s, …) then fold in the degree-1 ln(10) coefficient.
    double u = poly10_horner(s, kExp10Poly, fe);
    u = mla(u, s, +0.2302585092994045901e+1);

    // t = 1 + u*s in DD, renormalize.
    DD t = ddadd2_dd_d_dd(1.0, ddmul_dd_d_d(u, s));
    double r = ddnormalize_dd_dd(t).hi;

    r = sf64_ldexp(r, q);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_expm1 — SLEEF xexpm1 via DD exp core (expk2).
// ----------------------------------------------------------------------
extern "C" double sf64_expm1(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (gt_(x, 709.782712893383996732223))
        return kInf;
    if (lt_(x, -36.736800569677101399113302437))
        return -1.0;

    // d = expk2(x) - 1 in DD, then collapse.
    DD d = expk2(DD{x, 0.0}, fe);
    d = ddadd2_dd_dd_d(d, -1.0);
    double r = dd_to_d(d);

    // Signed-zero preservation: -0 in → -0 out.
    if (bits_of(x) == 0x8000000000000000ULL)
        r = -0.0;
    fe.flush();
    return r;
}

extern "C" double sf64_log(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double r = sf64_internal_log_core(x, fe);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_log2 — SLEEF xlog2 (u10) — same reduction, different DD fold-in.
// ----------------------------------------------------------------------
extern "C" double sf64_log2(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || lt_(x, 0.0))
        return qNaN();
    if (eq_(x, 0.0))
        return sf64_neg(kInf);
    if (isinf_(x))
        return kInf;

    constexpr double kDblMin = 2.2250738585072014e-308;
    constexpr double kScale = 4.294967296e9 * 4.294967296e9;
    bool sub = false;
    if (lt_(x, kDblMin)) {
        x = sf64_mul(x, kScale);
        sub = true;
    }

    int e = ilogb2k_bits(sf64_mul(x, 1.3333333333333333));
    double m = ldexp3k_f(x, -e);
    if (sub)
        e -= 64;

    const DD xdd = dddiv_dd_dd_dd(ddadd2_dd_d_d(-1.0, m), ddadd2_dd_d_d(1.0, m));
    const double x2 = sf64_mul(xdd.hi, xdd.hi);
    const double t = poly7_horner(x2, kLog2Poly, fe);

    // s = e  +  x * log2(e)_dd  +  x² * x.hi * t     (all DD)
    DD s = ddadd2_dd_d_dd(sf64_from_i32(e),
                          ddmul_dd_dd_dd(xdd, DD{2.885390081777926774, 6.0561604995516736434e-18}));
    s = ddadd2_dd_dd_d(s, sf64_mul(sf64_mul(x2, xdd.hi), t));
    const double r = dd_to_d(s);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_log10 — SLEEF xlog10 (u10).
// ----------------------------------------------------------------------
extern "C" double sf64_log10(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || lt_(x, 0.0))
        return qNaN();
    if (eq_(x, 0.0))
        return sf64_neg(kInf);
    if (isinf_(x))
        return kInf;

    constexpr double kDblMin = 2.2250738585072014e-308;
    constexpr double kScale = 4.294967296e9 * 4.294967296e9;
    bool sub = false;
    if (lt_(x, kDblMin)) {
        x = sf64_mul(x, kScale);
        sub = true;
    }

    int e = ilogb2k_bits(sf64_mul(x, 1.3333333333333333));
    double m = ldexp3k_f(x, -e);
    if (sub)
        e -= 64;

    const DD xdd = dddiv_dd_dd_dd(ddadd2_dd_d_d(-1.0, m), ddadd2_dd_d_d(1.0, m));
    const double x2 = sf64_mul(xdd.hi, xdd.hi);
    const double t = poly7_horner(x2, kLog10Poly, fe);

    // s = log10(2)_dd * e  +  x * log10(e)_dd  +  x²*x.hi*t
    DD s = ddmul_dd_dd_d(DD{0.30102999566398119802, -2.803728127785170339e-18}, sf64_from_i32(e));
    s = ddadd2_dd_dd(s, ddmul_dd_dd_dd(xdd, DD{0.86858896380650363334, 1.1430059694096389311e-17}));
    s = ddadd2_dd_dd_d(s, sf64_mul(sf64_mul(x2, xdd.hi), t));
    const double r = dd_to_d(s);
    fe.flush();
    return r;
}

// ----------------------------------------------------------------------
// sf64_log1p — SLEEF xlog1p (u10).
// ----------------------------------------------------------------------
extern "C" double sf64_log1p(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || lt_(x, -1.0))
        return qNaN();
    if (eq_(x, -1.0))
        return sf64_neg(kInf);
    if (isinf_(x) && gt_(x, 0.0))
        return kInf;

    // dp1 = d + 1; exponent comes from dp1 * 4/3.
    const double dp1 = sf64_add(x, 1.0);
    constexpr double kDblMin = 2.2250738585072014e-308;
    constexpr double kScale = 4.294967296e9 * 4.294967296e9;
    bool sub = false;
    double dp1s = dp1;
    if (lt_(dp1s, kDblMin)) {
        dp1s = sf64_mul(dp1s, kScale);
        sub = true;
    }

    int e = ilogb2k_bits(sf64_mul(dp1s, 1.3333333333333333));
    double t = ldexp3k_f(1.0, -e); // 2^-e  (exact)
    // m = d*t + (t - 1)    (this is the key cancellation-safe m = (1+d)*2^-e - 1)
    double m = sf64_fma(x, t, sf64_sub(t, 1.0));
    if (sub)
        e -= 64;

    // x_dd = m / (2 + m)   as DD.
    const DD xdd = dddiv_dd_dd_dd(DD{m, 0.0}, ddadd2_dd_d_d(2.0, m));
    const double x2 = sf64_mul(xdd.hi, xdd.hi);
    const double tp = poly7_horner(x2, kLogPoly, fe);

    // s = ln(2)_dd * e  +  2*x  +  x² * x.hi * tp
    DD s = ddmul_dd_dd_d(DD{kLN2, kLN2_LO}, sf64_from_i32(e));
    s = ddadd2_dd_dd(s, ddscale_dd_dd_d(xdd, 2.0));
    s = ddadd2_dd_dd_d(s, sf64_mul(sf64_mul(x2, xdd.hi), tp));

    double r = dd_to_d(s);
    if (bits_of(x) == 0x8000000000000000ULL)
        r = -0.0;
    fe.flush();
    return r;
}
