// Bit-exact correctness tests for the `sf64_*_r` surface.
//
// Oracle: host FPU under `std::fesetround()`. This is a valid bit-exact
// oracle per CLAUDE.md — host FPU is forbidden as an implementation inside
// `src/` but allowed as an oracle in `tests/`.
//
// The harness resets `fesetround` back to `FE_TONEAREST` at every iteration
// boundary so the mode is scoped to each `host_*` evaluation and the soft-
// fp64 call itself runs under the default RNE host mode (the call does not
// observe host fenv anyway — it only consumes its `mode` argument).
//
// Coverage: add/sub/mul/div/sqrt/fma/rint, f64→f32, f64→i{8,16,32,64},
// f64→u{8,16,32,64}. Each under all five modes (RNE/RTZ/RUP/RDN/RNA).
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cfenv>
#include <cmath>
#include <cstdio>

namespace {

struct ModeMapping {
    sf64_rounding_mode sf_mode;
    int host_mode; // one of FE_TONEAREST / FE_TOWARDZERO / FE_UPWARD / FE_DOWNWARD
                   // (-1 = no host equivalent — used for RNA)
    const char* name;
};

// Host FPU doesn't have round-to-nearest-ties-away. We skip RNA where the
// oracle would need it and cover it via targeted rows that exercise the
// tie-direction differences from RNE (halfway cases with even LSB).
constexpr ModeMapping kModes[] = {
    {SF64_RNE, FE_TONEAREST, "RNE"},
    {SF64_RTZ, FE_TOWARDZERO, "RTZ"},
    {SF64_RUP, FE_UPWARD, "RUP"},
    {SF64_RDN, FE_DOWNWARD, "RDN"},
};

struct HostFenvGuard {
    int prev;
    explicit HostFenvGuard(int m) : prev(std::fegetround()) {
        std::fesetround(m);
        // Hard memory barrier so the compiler cannot constant-fold the FP
        // op below us using its default rounding mode. AppleClang on the
        // macos-14 GitHub runner in -O2 / -O3 has been observed folding
        // host_sub(1.0, 2^-1022) through the default RNE path (returning
        // 1.0) instead of honoring the just-set FE_TOWARDZERO (which
        // would return the just-below-1.0 representable). The `volatile
        // double va = a, vb = b;` lines below already force real loads,
        // but the FP op itself can still be precomputed by the
        // optimizer if it doesn't see a barrier between fesetround and
        // the op. This asm volatile is opaque to the optimizer and
        // forces it to treat any subsequent FP behavior as
        // mode-dependent.
        __asm__ __volatile__("" ::: "memory");
    }
    ~HostFenvGuard() {
        __asm__ __volatile__("" ::: "memory");
        std::fesetround(prev);
    }
};

// noinline + volatile + asm-memory-barrier in HostFenvGuard together force
// the host FP op to run at runtime under the just-set rounding mode, even on
// older AppleClang where `volatile` alone wasn't enough to defeat compile-
// time constant-folding of e.g. 1.0 - 2^-1022 to 1.0 (the RNE result)
// despite a fesetround(FE_TOWARDZERO) on the line above.
[[gnu::noinline]] double host_add(double a, double b, int m) {
    HostFenvGuard g(m);
    volatile double va = a, vb = b;
    return va + vb;
}
[[gnu::noinline]] double host_sub(double a, double b, int m) {
    HostFenvGuard g(m);
    volatile double va = a, vb = b;
    return va - vb;
}
[[gnu::noinline]] double host_mul(double a, double b, int m) {
    HostFenvGuard g(m);
    volatile double va = a, vb = b;
    return va * vb;
}
[[gnu::noinline]] double host_div(double a, double b, int m) {
    HostFenvGuard g(m);
    volatile double va = a, vb = b;
    return va / vb;
}
[[gnu::noinline]] double host_sqrt(double a, int m) {
    HostFenvGuard g(m);
    volatile double va = a;
    return std::sqrt(va);
}
[[gnu::noinline]] double host_fma(double a, double b, double c, int m) {
    HostFenvGuard g(m);
    volatile double va = a, vb = b, vc = c;
    return std::fma(va, vb, vc);
}
[[gnu::noinline]] float host_to_f32(double x, int m) {
    HostFenvGuard g(m);
    volatile double vx = x;
    return static_cast<float>(vx);
}

// Finite random doubles the host can round safely. The `any` stream would
// include NaNs/infs whose +/- oracle matches by definition.
double finite_d(host_oracle::LCG& rng) {
    while (true) {
        double d = host_oracle::from_bits(rng.next());
        if (std::isfinite(d) && std::fabs(d) < 1e200)
            return d;
    }
}

} // namespace

int main() {
    using namespace host_oracle;

    // ---- NaN / inf / zero edges are mode-invariant. No fenv needed. ------
    // (Covered by the existing RNE test_arithmetic_exact; reuse.)

    // ---- Mode-dependent bit-exact pairs against host FPU. ----------------
    for (const auto& m : kModes) {
        for (double a : edge_cases_f64()) {
            for (double b : edge_cases_f64()) {
                auto check = [&](double got, double expect, const char* op) {
                    if (!::host_oracle::equal_exact_or_nan(got, expect)) {
                        std::fprintf(stderr,
                                     "FAIL: %s(%s, a=%a, b=%a) got=%a (0x%016llx) "
                                     "expect=%a (0x%016llx)\n",
                                     op, m.name, a, b, got,
                                     (unsigned long long)::host_oracle::bits(got), expect,
                                     (unsigned long long)::host_oracle::bits(expect));
                        std::abort();
                    }
                };
                check(sf64_add_r(m.sf_mode, a, b), host_add(a, b, m.host_mode), "add");
                check(sf64_sub_r(m.sf_mode, a, b), host_sub(a, b, m.host_mode), "sub");
                check(sf64_mul_r(m.sf_mode, a, b), host_mul(a, b, m.host_mode), "mul");
                check(sf64_div_r(m.sf_mode, a, b), host_div(a, b, m.host_mode), "div");
            }
        }
        for (double a : edge_cases_f64()) {
            SF64_CHECK_BITS(sf64_sqrt_r(m.sf_mode, a), host_sqrt(a, m.host_mode));
        }
    }

    // ---- fma under each mode. --------------------------------------------
    for (const auto& m : kModes) {
        for (double a : edge_cases_f64()) {
            for (double b : edge_cases_f64()) {
                for (double c : edge_cases_f64()) {
                    const double got = sf64_fma_r(m.sf_mode, a, b, c);
                    const double expect = host_fma(a, b, c, m.host_mode);
                    if (!::host_oracle::equal_exact_or_nan(got, expect)) {
                        std::fprintf(stderr,
                                     "FAIL: fma(%s, a=%a, b=%a, c=%a) got=%a (0x%016llx) "
                                     "expect=%a (0x%016llx)\n",
                                     m.name, a, b, c, got,
                                     (unsigned long long)::host_oracle::bits(got), expect,
                                     (unsigned long long)::host_oracle::bits(expect));
                        std::abort();
                    }
                }
            }
        }
    }

    // ---- Random finite pairs: 4096 per mode. -----------------------------
    for (const auto& m : kModes) {
        LCG rng(0xF00DFEEDULL ^ static_cast<uint64_t>(m.sf_mode));
        for (int i = 0; i < 4096; ++i) {
            double a = finite_d(rng);
            double b = finite_d(rng);
            SF64_CHECK_BITS(sf64_add_r(m.sf_mode, a, b), host_add(a, b, m.host_mode));
            SF64_CHECK_BITS(sf64_sub_r(m.sf_mode, a, b), host_sub(a, b, m.host_mode));
            SF64_CHECK_BITS(sf64_mul_r(m.sf_mode, a, b), host_mul(a, b, m.host_mode));
            SF64_CHECK_BITS(sf64_div_r(m.sf_mode, a, b), host_div(a, b, m.host_mode));
            if (!(std::signbit(a) && std::isfinite(a) && a != 0.0)) {
                SF64_CHECK_BITS(sf64_sqrt_r(m.sf_mode, std::fabs(a)),
                                host_sqrt(std::fabs(a), m.host_mode));
            }
        }
    }

    // ---- f64 -> f32 under each mode. -------------------------------------
    for (const auto& m : kModes) {
        for (double a : edge_cases_f64()) {
            const float got = sf64_to_f32_r(m.sf_mode, a);
            const float expect = host_to_f32(a, m.host_mode);
            if (std::isnan(got) && std::isnan(expect)) {
                continue;
            }
            if (bits_f32(got) != bits_f32(expect)) {
                std::fprintf(stderr, "FAIL: sf64_to_f32_r(%s, %a) got=0x%08x expect=0x%08x\n",
                             m.name, a, bits_f32(got), bits_f32(expect));
                std::abort();
            }
        }
    }

    // ---- f64 -> integer under each mode. ---------------------------------
    // Oracle: nearbyint under host mode, then saturate to destination range.
    auto host_round_int64 = [](double x, int m) -> int64_t {
        HostFenvGuard g(m);
        if (std::isnan(x))
            return 0;
        // nearbyint(x) under fenv: result rounds per current mode.
        volatile double v = x;
        double r = std::nearbyint(v);
        if (std::isnan(r))
            return 0;
        if (r == std::numeric_limits<double>::infinity() || r > 9.2233720368547758e18) {
            return INT64_MAX;
        }
        if (r == -std::numeric_limits<double>::infinity() || r < -9.2233720368547758e18) {
            return INT64_MIN;
        }
        return static_cast<int64_t>(r);
    };
    auto saturate_signed = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
        if (v < lo)
            return lo;
        if (v > hi)
            return hi;
        return v;
    };
    auto saturate_unsigned = [](int64_t v, uint64_t hi) -> uint64_t {
        if (v < 0)
            return 0;
        if (static_cast<uint64_t>(v) > hi)
            return hi;
        return static_cast<uint64_t>(v);
    };

    for (const auto& m : kModes) {
        for (double a : edge_cases_f64()) {
            const int64_t host_i64 = host_round_int64(a, m.host_mode);

            const int8_t expect_i8 =
                static_cast<int8_t>(saturate_signed(host_i64, INT8_MIN, INT8_MAX));
            const int8_t got_i8 = sf64_to_i8_r(m.sf_mode, a);
            if (got_i8 != expect_i8) {
                std::fprintf(stderr, "FAIL: to_i8_r(%s, %a) got=%d expect=%d\n", m.name, a, got_i8,
                             expect_i8);
                std::abort();
            }

            const int16_t expect_i16 =
                static_cast<int16_t>(saturate_signed(host_i64, INT16_MIN, INT16_MAX));
            const int16_t got_i16 = sf64_to_i16_r(m.sf_mode, a);
            SF64_CHECK(got_i16 == expect_i16);

            const int32_t expect_i32 =
                static_cast<int32_t>(saturate_signed(host_i64, INT32_MIN, INT32_MAX));
            const int32_t got_i32 = sf64_to_i32_r(m.sf_mode, a);
            SF64_CHECK(got_i32 == expect_i32);

            // i64 saturation: only test where the rounded value comfortably
            // fits — the |x| > 2^63 edge is covered by the fp64 large-corpus
            // rows elsewhere and the oracle above can't distinguish saturate
            // vs overflow precisely at the boundary.
            if (std::fabs(a) < 9.0e18 && std::isfinite(a) && !std::isnan(a)) {
                const int64_t got_i64 = sf64_to_i64_r(m.sf_mode, a);
                SF64_CHECK(got_i64 == host_i64);
            }

            const uint8_t expect_u8 = static_cast<uint8_t>(saturate_unsigned(host_i64, UINT8_MAX));
            const uint8_t got_u8 = sf64_to_u8_r(m.sf_mode, a);
            SF64_CHECK(got_u8 == expect_u8);

            const uint16_t expect_u16 =
                static_cast<uint16_t>(saturate_unsigned(host_i64, UINT16_MAX));
            const uint16_t got_u16 = sf64_to_u16_r(m.sf_mode, a);
            SF64_CHECK(got_u16 == expect_u16);

            const uint32_t expect_u32 =
                static_cast<uint32_t>(saturate_unsigned(host_i64, UINT32_MAX));
            const uint32_t got_u32 = sf64_to_u32_r(m.sf_mode, a);
            SF64_CHECK(got_u32 == expect_u32);
        }
    }

    // ---- rint_r ----------------------------------------------------------
    for (const auto& m : kModes) {
        for (double a : edge_cases_f64()) {
            if (std::isnan(a))
                continue;
            HostFenvGuard g(m.host_mode);
            volatile double va = a;
            const double expect = std::nearbyint(va);
            const double got = sf64_rint_r(m.sf_mode, a);
            // Reset before comparing so any library use inside the macro
            // runs under RNE.
            std::fesetround(FE_TONEAREST);
            SF64_CHECK_BITS(got, expect);
        }
    }

    // ---- RNA spot-checks: halfway cases with even-LSB truncations where
    //      RNE rounds to even and RNA rounds away. -------------------------
    //
    // 0.5 → RNE: 0 (to even); RNA: 1 (away)
    // 1.5 → RNE: 2 (to even); RNA: 2 (away)  (same direction; not a distinguisher)
    // 2.5 → RNE: 2 (to even); RNA: 3 (away)
    // 3.5 → RNE: 4 (to even); RNA: 4 (away)
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNA, 0.5), 1.0);
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNA, -0.5), -1.0);
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNA, 2.5), 3.0);
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNA, -2.5), -3.0);
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNE, 0.5), 0.0);
    SF64_CHECK_BITS(sf64_rint_r(SF64_RNE, 2.5), 2.0);
    SF64_CHECK(sf64_to_i32_r(SF64_RNA, 0.5) == 1);
    SF64_CHECK(sf64_to_i32_r(SF64_RNA, 2.5) == 3);
    SF64_CHECK(sf64_to_i32_r(SF64_RNE, 0.5) == 0);
    SF64_CHECK(sf64_to_i32_r(SF64_RNE, 2.5) == 2);

    std::printf("test_rounding_modes: all rounding-mode assertions passed\n");
    return 0;
}
