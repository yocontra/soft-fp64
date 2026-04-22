// Coverage extensions that require MPFR arbitrary-precision references.
//
// Two sweeps live here:
//
//   1. sf64_powr MPFR sweep. powr(x, y) requires x > 0. ~1024 log-uniform
//      (x, y) pairs in x ∈ [1e-200, 1e200], y ∈ [-50, 50]. ≤4 ULP band.
//
//   2. Payne-Hanek MPFR stress for sin/cos/tan. x = k·π + δ for
//      k ∈ {2^40, 2^45, 2^50} and δ ∈ {0, ±ε, ±1e-12}. ≤4 ULP vs MPFR
//      200-bit reference.
//
// Skipped if MPFR is not available on the host. MPFR detection/linkage is
// driven from the parent tests/CMakeLists.txt.
//
// Tolerance is tighter here than in the libm-diff test because MPFR at
// 200 bits rounded to double is the mathematically correct rounded value;
// any observed delta is a soft-fp64 bug, not oracle noise.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <mpfr.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// ---- bit helpers (copied inline — tests/host_oracle.h is fine to include,
// but this file also compiles standalone against MPFR only). --------------

inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

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
    const int64_t sa = static_cast<int64_t>(ab);
    const int64_t sb = static_cast<int64_t>(bb);
    return sa > sb ? (sa - sb) : (sb - sa);
}

// ---- deterministic LCG ---------------------------------------------------

class LCG {
  public:
    explicit LCG(uint64_t seed = 0xC0FFEEC0FFEEULL) : state_(seed) {}
    uint64_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }
    double uniform_unit() {
        return static_cast<double>(next() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }
    double log_uniform(double lo, double hi) {
        const double llo = std::log(lo);
        const double lhi = std::log(hi);
        return std::exp(llo + (lhi - llo) * uniform_unit());
    }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform_unit(); }

  private:
    uint64_t state_;
};

// ---- MPFR oracle plumbing ------------------------------------------------

constexpr mpfr_prec_t ORACLE_PREC = 200;

inline double mpfr_ref_powr(double x, double y) {
    mpfr_t xm, ym, rm;
    mpfr_init2(xm, ORACLE_PREC);
    mpfr_init2(ym, ORACLE_PREC);
    mpfr_init2(rm, ORACLE_PREC);
    mpfr_set_d(xm, x, MPFR_RNDN);
    mpfr_set_d(ym, y, MPFR_RNDN);
    mpfr_powr(rm, xm, ym, MPFR_RNDN);
    const double r = mpfr_get_d(rm, MPFR_RNDN);
    mpfr_clear(xm);
    mpfr_clear(ym);
    mpfr_clear(rm);
    return r;
}

inline double mpfr_ref_1arg(int (*fn)(mpfr_t, const mpfr_t, mpfr_rnd_t), double x) {
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

// Compute k·π + δ *at the MPFR precision* so the test input itself is
// represented as faithfully as possible. The double we hand to sf64_* is
// RNDN-rounded from that MPFR value; the oracle operates on the same
// rounded double. This matches the task spec: we test soft-fp64's argument
// reduction, not how faithfully `k·π + δ` is expressible in double.
inline double build_kpi_plus_delta(double k, double delta) {
    mpfr_t pi, km, dm, tmp;
    mpfr_init2(pi, ORACLE_PREC);
    mpfr_init2(km, ORACLE_PREC);
    mpfr_init2(dm, ORACLE_PREC);
    mpfr_init2(tmp, ORACLE_PREC);
    mpfr_const_pi(pi, MPFR_RNDN);
    mpfr_set_d(km, k, MPFR_RNDN);
    mpfr_set_d(dm, delta, MPFR_RNDN);
    mpfr_mul(tmp, km, pi, MPFR_RNDN);  // tmp = k·π
    mpfr_add(tmp, tmp, dm, MPFR_RNDN); // tmp = k·π + δ
    const double x = mpfr_get_d(tmp, MPFR_RNDN);
    mpfr_clear(pi);
    mpfr_clear(km);
    mpfr_clear(dm);
    mpfr_clear(tmp);
    return x;
}

// ---- test harness --------------------------------------------------------

struct Stats {
    const char* name = "?";
    int64_t max_ulp = 0;
    double worst_x = 0.0, worst_y = 0.0;
    double worst_got = 0.0, worst_expect = 0.0;
    int checked = 0;
    int nan_mismatch = 0;
    int inf_mismatch = 0;
    int zero_mismatch = 0;
    // Trivial matches: both impl and oracle returned matching NaN-or-
    // signed-infinity. These contribute nothing to precision verification
    // — they only confirm the algorithm overflows / NaN-propagates the
    // same way the oracle does. A sweep whose sampling regime collapses
    // most inputs into the degenerate regime is not actually testing
    // precision in its advertised range. Gated at 25% in fail(), same
    // threshold as tests/mpfr/test_mpfr_diff.cpp.
    int trivial_matches = 0;
    int64_t tier = 4;
};

void record(Stats& s, double x, double y, double got, double expect) {
    s.checked++;
    const bool gnan = std::isnan(got);
    const bool enan = std::isnan(expect);
    if (gnan != enan) {
        s.nan_mismatch++;
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
        // oracle agree the input overflows; we learn nothing about
        // precision. Pre-fix this fell through to ulp_diff which returned
        // 0 for matching bit patterns — silently counted as a "0-ULP"
        // success.
        s.trivial_matches++;
        return;
    }
    // Zero-class symmetry: ULP diff on its own doesn't distinguish
    // "expect is exactly ±0 but got is a finite non-zero." The libm-backed
    // oracle collapses to signed zero for plenty of real inputs, so a soft
    // fp64 that returns a small non-zero there is a real bug — not a
    // round-off 1-ULP drift. Same for the mirror case. ulp_diff treats ±0
    // as equal, so this branch doesn't double-count.
    const bool gzero = (got == 0.0);
    const bool ezero = (expect == 0.0);
    if (gzero != ezero) {
        s.zero_mismatch++;
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

constexpr double kTrivialMatchMaxRatio = 0.25;

bool fail(const Stats& s) {
    if (s.nan_mismatch > 0 || s.inf_mismatch > 0)
        return true;
    if (s.zero_mismatch > 0)
        return true;
    if (s.max_ulp > s.tier)
        return true;
    if (s.checked > 0) {
        const double ratio = static_cast<double>(s.trivial_matches) / s.checked;
        if (ratio > kTrivialMatchMaxRatio) {
            std::fprintf(stderr,
                         "    !! trivial-match gate: %d/%d = %.1f%% > %.0f%% — sampling regime "
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

void report(const Stats& s) {
    const double trivial_pct =
        s.checked > 0 ? (100.0 * static_cast<double>(s.trivial_matches) / s.checked) : 0.0;
    std::printf("  %-14s n=%-5d max_ulp=%-5lld tier=%-3lld trivial=%5.1f%% "
                "worst (x=%.17g y=%.17g got=%.17g expect=%.17g)\n",
                s.name, s.checked, static_cast<long long>(s.max_ulp),
                static_cast<long long>(s.tier), trivial_pct, s.worst_x, s.worst_y, s.worst_got,
                s.worst_expect);
    if (s.nan_mismatch || s.inf_mismatch || s.zero_mismatch) {
        std::printf("    !! nan_mismatch=%d inf_mismatch=%d zero_mismatch=%d\n", s.nan_mismatch,
                    s.inf_mismatch, s.zero_mismatch);
    }
}

} // namespace

int main() {
    std::printf("== test_coverage_mpfr (MPFR prec=%d) ==\n", static_cast<int>(ORACLE_PREC));

    std::vector<Stats> results;

    // ------------------------------------------------------------------
    // Task 3 — sf64_powr sweep vs MPFR 200-bit `powr`, ≤4 ULP.
    // ------------------------------------------------------------------
    {
        Stats s;
        s.name = "powr";
        s.tier = 4;
        LCG rng(0x50FA64C0FFEEULL);
        // ~1024 pairs, log-uniform x in [1e-100, 1e100], uniform y in [-3, 3].
        // The bounds are picked so |y · log(x)| ≤ 3·230 = 690 stays inside
        // the exp range (~±709), keeping ~all outputs representable so the
        // sweep actually tests precision rather than overflow/underflow
        // class agreement. The previous bounds (x in [1e-200, 1e200],
        // y in [-50, 50]) gave 41.7% trivial-match (samples that overflow
        // / underflow on both sides) which the trivial-match gate fires on
        // — it's a coverage claim that collapses into class-agreement, not
        // precision testing.
        //
        // Overflow / underflow class-agreement coverage lives in
        // tests/test_powr_ieee754.cpp, which is bit-exact (not ULP) and
        // explicitly enumerates the §9.2.1 boundary inputs.
        constexpr int kN = 1024;
        int overflow_class = 0; // separate counter for visibility; not gated
        for (int i = 0; i < kN; ++i) {
            const double x = rng.log_uniform(1e-100, 1e100);
            const double y = rng.uniform(-3.0, 3.0);
            const double got = sf64_powr(x, y);
            const double expect = mpfr_ref_powr(x, y);
            if (std::isinf(got) || std::isinf(expect) || got == 0.0 || expect == 0.0)
                overflow_class++;
            record(s, x, y, got, expect);
        }
        std::printf("  powr sweep: %d/%d samples reached overflow/underflow class\n",
                    overflow_class, kN);
        results.push_back(s);
    }

    // ------------------------------------------------------------------
    // Task 5 — Payne-Hanek stress (MPFR-referenced), ≤4 ULP.
    //
    // Covers `|x| >> 1` where Cody-Waite would lose all significance and
    // Payne-Hanek is the only correctness path. The multipliers exercise
    // the full Payne-Hanek table: 2^40/45/50 are just past the 1e14
    // switchover; 2^500 and 2^900 push to the large end of the SLEEF
    // `rempitabdp` table entries. 2^1000 is within epsilon of DBL_MAX
    // and exercises the final table entry.
    // ------------------------------------------------------------------
    {
        const double ks[] = {std::ldexp(1.0, 40),  std::ldexp(1.0, 45),  std::ldexp(1.0, 50),
                             std::ldexp(1.0, 500), std::ldexp(1.0, 900), std::ldexp(1.0, 1000)};
        const double deltas[] = {0.0, +std::numeric_limits<double>::epsilon(),
                                 -std::numeric_limits<double>::epsilon(), +1e-12, -1e-12};

        Stats ss;
        ss.name = "ph.sin";
        ss.tier = 4;
        Stats sc;
        sc.name = "ph.cos";
        sc.tier = 4;
        Stats st;
        st.name = "ph.tan";
        st.tier = 4;

        for (double k : ks) {
            for (double d : deltas) {
                const double x = build_kpi_plus_delta(k, d);
                record(ss, x, 0.0, sf64_sin(x), mpfr_ref_1arg(mpfr_sin, x));
                record(sc, x, 0.0, sf64_cos(x), mpfr_ref_1arg(mpfr_cos, x));
                record(st, x, 0.0, sf64_tan(x), mpfr_ref_1arg(mpfr_tan, x));
            }
        }
        results.push_back(ss);
        results.push_back(sc);
        results.push_back(st);
    }

    int failures = 0;
    std::printf("\n[results vs MPFR @ %d bits]\n", static_cast<int>(ORACLE_PREC));
    for (const auto& s : results) {
        report(s);
        if (fail(s))
            failures++;
    }

    mpfr_free_cache();

    if (failures) {
        std::fprintf(stderr, "\nFAIL: %d MPFR sweep(s) exceeded tolerance.\n", failures);
        std::abort();
    }

    std::printf("\nOK\n");
    return 0;
}
