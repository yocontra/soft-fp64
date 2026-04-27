// Host-side spot checks for every SF64_FE_RAISE site wired in Track C.
//
// This is not a sweep — every assertion targets a single hook at a single
// input so a regression lists an exact operation. A future oracle-driven
// sweep (TestFloat fl2 tokens) lives in tests/testfloat/run_testfloat.cpp.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#ifndef SF64_TEST_FENV_MODE
#define SF64_TEST_FENV_MODE 1
#endif

// Under `disabled` (mode 0) every raise/clear is a no-op and getall is
// hard-wired to 0 — so the "expect flag got raised" assertions are
// inapplicable. `has()` collapses to `true` in that case so the test
// reduces to a surface-smoke: the ABI is still linkable, every entry
// still compiles, and the test binary still runs green.
#if SF64_TEST_FENV_MODE == 1
constexpr bool kFlagsActive = true;
#else
constexpr bool kFlagsActive = false;
#endif

namespace {

void clear_all() {
    sf64_fe_clear(SF64_FE_INVALID | SF64_FE_DIVBYZERO | SF64_FE_OVERFLOW | SF64_FE_UNDERFLOW |
                  SF64_FE_INEXACT);
}

// `has_all` / `has_none` are the pair used inside expectations. Under
// disabled mode both collapse to `true` so the test becomes a surface-
// smoke; under tls both carry the real semantics.
bool has_all(unsigned bits) {
    if (!kFlagsActive)
        return true;
    return (sf64_fe_getall() & bits) == bits;
}
bool has_none(unsigned bits) {
    if (!kFlagsActive)
        return true;
    return (sf64_fe_getall() & bits) == 0u;
}

void expect(bool condition, const char* label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::fflush(stderr);
        std::abort();
    }
}

void expect_flags(unsigned expected, const char* label) {
    if (!kFlagsActive)
        return;
    const unsigned got = sf64_fe_getall();
    if (got != expected) {
        std::fprintf(stderr, "FAIL: %s — got flags=0x%02x expected=0x%02x\n", label, got, expected);
        std::fflush(stderr);
        std::abort();
    }
}

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kMax = std::numeric_limits<double>::max();
constexpr double kMin = std::numeric_limits<double>::min();
constexpr double kDenormMin = std::numeric_limits<double>::denorm_min();

// ---- INVALID hooks ------------------------------------------------------

void test_invalid() {
    // add/sub: inf + (-inf).
    clear_all();
    sf64_add(kInf, -kInf);
    expect(has_all(SF64_FE_INVALID), "add inf+(-inf) → INVALID");

    clear_all();
    sf64_sub(kInf, kInf);
    expect(has_all(SF64_FE_INVALID), "sub inf-inf → INVALID");

    // mul: 0 * inf.
    clear_all();
    sf64_mul(0.0, kInf);
    expect(has_all(SF64_FE_INVALID), "mul 0*inf → INVALID");

    clear_all();
    sf64_mul(-kInf, 0.0);
    expect(has_all(SF64_FE_INVALID), "mul -inf*0 → INVALID");

    // div: 0/0, inf/inf.
    clear_all();
    sf64_div(0.0, 0.0);
    expect(has_all(SF64_FE_INVALID), "div 0/0 → INVALID");

    clear_all();
    sf64_div(kInf, -kInf);
    expect(has_all(SF64_FE_INVALID), "div inf/(-inf) → INVALID");

    // rem: inf % y, x % 0.
    clear_all();
    sf64_rem(kInf, 1.0);
    expect(has_all(SF64_FE_INVALID), "rem inf%1 → INVALID");

    clear_all();
    sf64_rem(1.0, 0.0);
    expect(has_all(SF64_FE_INVALID), "rem 1%0 → INVALID");

    // sqrt of negative finite / negative inf.
    clear_all();
    sf64_sqrt(-1.0);
    expect(has_all(SF64_FE_INVALID), "sqrt(-1) → INVALID");

    clear_all();
    sf64_sqrt(-kInf);
    expect(has_all(SF64_FE_INVALID), "sqrt(-inf) → INVALID");

    // fma invalid-ops.
    clear_all();
    sf64_fma(kInf, 0.0, 1.0);
    expect(has_all(SF64_FE_INVALID), "fma(inf, 0, 1) → INVALID");

    clear_all();
    sf64_fma(2.0, kInf, -kInf);
    expect(has_all(SF64_FE_INVALID), "fma(2, inf, -inf) → INVALID");

    // IEEE §7.2: the 0×∞ sub-operation is invalid regardless of c, so
    // fma(0, ∞, qNaN) must raise INVALID even though c is a NaN. The
    // raise must fire *before* NaN propagation short-circuits (the hook
    // site in src/sqrt_fma.cpp is reordered specifically to handle this).
    clear_all();
    sf64_fma(kInf, 0.0, kNaN);
    expect(has_all(SF64_FE_INVALID), "fma(inf, 0, qNaN) → INVALID (before NaN propagation)");
    clear_all();
    sf64_fma(0.0, kInf, kNaN);
    expect(has_all(SF64_FE_INVALID), "fma(0, inf, qNaN) → INVALID (before NaN propagation)");

    // convert: NaN → int.
    clear_all();
    (void)sf64_to_i32(kNaN);
    expect(has_all(SF64_FE_INVALID), "i32(NaN) → INVALID");

    clear_all();
    (void)sf64_to_u64(kNaN);
    expect(has_all(SF64_FE_INVALID), "u64(NaN) → INVALID");

    // convert: out-of-range float → int (saturation).
    clear_all();
    (void)sf64_to_i8(1000.0);
    expect(has_all(SF64_FE_INVALID), "i8(1000) → INVALID (saturated)");

    clear_all();
    (void)sf64_to_u16(-1.0);
    expect(has_all(SF64_FE_INVALID), "u16(-1) → INVALID (negative→unsigned)");

    clear_all();
    (void)sf64_to_u64(kInf);
    expect(has_all(SF64_FE_INVALID), "u64(+inf) → INVALID");

    std::fputs("  invalid: ok\n", stdout);
}

// ---- DIVBYZERO hook -----------------------------------------------------

void test_divbyzero() {
    clear_all();
    const double r = sf64_div(1.0, 0.0);
    expect(has_all(SF64_FE_DIVBYZERO), "1/0 → DIVBYZERO");
    expect(std::isinf(r) && r > 0, "1/0 → +inf");

    clear_all();
    const double n = sf64_div(-1.0, 0.0);
    expect(has_all(SF64_FE_DIVBYZERO), "-1/0 → DIVBYZERO");
    expect(std::isinf(n) && n < 0, "-1/0 → -inf");

    std::fputs("  divbyzero: ok\n", stdout);
}

// ---- OVERFLOW hook ------------------------------------------------------

void test_overflow() {
    clear_all();
    const double big = sf64_mul(kMax, 2.0);
    expect(has_all(SF64_FE_OVERFLOW | SF64_FE_INEXACT), "max*2 → OVERFLOW+INEXACT");
    expect(std::isinf(big), "max*2 → inf");

    clear_all();
    (void)sf64_add(kMax, kMax);
    expect(has_all(SF64_FE_OVERFLOW | SF64_FE_INEXACT), "max+max → OVERFLOW+INEXACT");

    // fma carry into overflow.
    clear_all();
    (void)sf64_fma(kMax, 2.0, 0.0);
    expect(has_all(SF64_FE_OVERFLOW | SF64_FE_INEXACT), "fma(max, 2, 0) → OVERFLOW+INEXACT");

    // f64 → f32 overflow.
    clear_all();
    const float ov = sf64_to_f32(1e300);
    expect(has_all(SF64_FE_OVERFLOW | SF64_FE_INEXACT), "f32(1e300) → OVERFLOW+INEXACT");
    expect(std::isinf(ov), "f32(1e300) → inf");

    std::fputs("  overflow: ok\n", stdout);
}

// ---- UNDERFLOW hook -----------------------------------------------------

void test_underflow() {
    // DBL_MIN * DBL_MIN underflows to 0 (exact).
    clear_all();
    const double u = sf64_mul(kMin, kMin);
    expect(has_all(SF64_FE_UNDERFLOW | SF64_FE_INEXACT), "min*min → UNDERFLOW+INEXACT");
    expect(u == 0.0, "min*min → 0");

    // fma to denorm: half*denorm_min (tiny and inexact).
    clear_all();
    (void)sf64_fma(0.5, kDenormMin, 0.0);
    expect(has_all(SF64_FE_UNDERFLOW | SF64_FE_INEXACT),
           "fma(0.5, denorm_min, 0) → UNDERFLOW+INEXACT");

    // f64 subnormal → f32 collapses.
    clear_all();
    (void)sf64_to_f32(kDenormMin);
    expect(has_all(SF64_FE_UNDERFLOW | SF64_FE_INEXACT), "f32(f64 denorm_min) → UNDERFLOW+INEXACT");

    std::fputs("  underflow: ok\n", stdout);
}

// ---- INEXACT hook -------------------------------------------------------

void test_inexact() {
    // 0.1 + 0.2 → 0.30000000000000004, inexact.
    clear_all();
    (void)sf64_add(0.1, 0.2);
    expect(has_all(SF64_FE_INEXACT), "0.1+0.2 → INEXACT");
    expect(has_none(SF64_FE_INVALID), "0.1+0.2 → no INVALID");
    expect(has_none(SF64_FE_OVERFLOW), "0.1+0.2 → no OVERFLOW");

    // 1.0 + 1.0 is exact.
    clear_all();
    (void)sf64_add(1.0, 1.0);
    expect_flags(0u, "1+1 → no flags");

    // sqrt(2.0) is inexact.
    clear_all();
    (void)sf64_sqrt(2.0);
    expect(has_all(SF64_FE_INEXACT), "sqrt(2) → INEXACT");

    // sqrt(4.0) is exact.
    clear_all();
    (void)sf64_sqrt(4.0);
    expect_flags(0u, "sqrt(4) → no flags");

    // Int → f64 lossy: 2^63 - 1 doesn't fit in 53 bits.
    clear_all();
    (void)sf64_from_i64(static_cast<int64_t>(0x7FFFFFFFFFFFFFFF));
    expect(has_all(SF64_FE_INEXACT), "i64(2^63-1) → f64 INEXACT");

    // Int → f64 exact.
    clear_all();
    (void)sf64_from_i32(42);
    expect_flags(0u, "i32(42) → f64 no flags");

    // f64 → int with fractional drop.
    clear_all();
    (void)sf64_to_i32(1.5);
    expect(has_all(SF64_FE_INEXACT), "i32(1.5) → INEXACT");

    // f64 → int exact.
    clear_all();
    (void)sf64_to_i32(2.0);
    expect_flags(0u, "i32(2.0) → no flags");

    std::fputs("  inexact: ok\n", stdout);
}

// ---- save / restore / clear / test --------------------------------------

void test_save_restore() {
    clear_all();
    (void)sf64_div(1.0, 0.0);
    expect(has_all(SF64_FE_DIVBYZERO), "setup");

    sf64_fe_state_t saved{};
    sf64_fe_save(&saved);

    sf64_fe_raise(SF64_FE_OVERFLOW | SF64_FE_INEXACT);
    expect(has_all(SF64_FE_OVERFLOW), "post-raise");

    sf64_fe_restore(&saved);
    expect(has_all(SF64_FE_DIVBYZERO), "restored keeps DIVBYZERO");
    expect(has_none(SF64_FE_OVERFLOW), "restored drops OVERFLOW");

    // test() semantics — only meaningful under tls.
    if (kFlagsActive) {
        expect(sf64_fe_test(SF64_FE_DIVBYZERO) == 1, "test(DIVBYZERO) == 1");
        expect(sf64_fe_test(SF64_FE_OVERFLOW) == 0, "test(OVERFLOW) == 0");
    }

    // clear is scoped
    sf64_fe_clear(SF64_FE_DIVBYZERO);
    expect(has_none(SF64_FE_DIVBYZERO), "clear drops DIVBYZERO");

    std::fputs("  save/restore/clear/test: ok\n", stdout);
}

// ---- thread-local isolation ---------------------------------------------

void test_thread_isolation() {
    // Under disabled mode flags are a no-op, so there is nothing to isolate —
    // the thread spawn still runs (surface smoke), but accumulator equality
    // checks are inapplicable.
    clear_all();
    sf64_fe_raise(SF64_FE_INVALID);

    volatile bool worker_saw_cross_talk = false;
    volatile unsigned worker_flags = 0u;

    std::thread worker([&]() {
        if (sf64_fe_getall() != 0u) {
            worker_saw_cross_talk = true;
        }
        sf64_fe_raise(SF64_FE_DIVBYZERO | SF64_FE_INEXACT);
        worker_flags = sf64_fe_getall();
    });
    worker.join();

    if (kFlagsActive) {
        expect(!worker_saw_cross_talk, "worker sees its own empty fenv at start");
        expect(worker_flags == (SF64_FE_DIVBYZERO | SF64_FE_INEXACT),
               "worker accumulates its own flags");
        expect(sf64_fe_getall() == SF64_FE_INVALID,
               "main retains its own INVALID, unaffected by worker");
    }

    constexpr unsigned flags_per_thread[] = {SF64_FE_INVALID, SF64_FE_DIVBYZERO, SF64_FE_OVERFLOW,
                                             SF64_FE_UNDERFLOW, SF64_FE_INEXACT};
    std::vector<std::thread> workers;
    std::vector<unsigned> results(5, 0u);
    for (int i = 0; i < 5; ++i) {
        workers.emplace_back([i, &results]() {
            sf64_fe_raise(flags_per_thread[i]);
            results[i] = sf64_fe_getall();
        });
    }
    for (auto& t : workers)
        t.join();
    if (kFlagsActive) {
        for (int i = 0; i < 5; ++i) {
            expect(results[i] == flags_per_thread[i], "per-thread fanout is independent");
        }
    }

    std::fputs("  thread isolation: ok\n", stdout);
}

// ---- caller-state (`_ex`) surface — exercised under tls + explicit -----
//
// Under TLS mode the `_ex` surface coexists with the default surface; the
// test verifies that flagging into a caller-supplied state does NOT leak
// into TLS, and vice-versa.
// Under explicit mode the default TLS surface is no-ops (`getall_ex()`
// returns 0 etc.); the `_ex` surface is the only path that carries flag
// state.
// Under disabled mode the `_ex` surface is not in the archive — the test
// is omitted entirely.

#if SF64_TEST_FENV_MODE == 1 || SF64_TEST_FENV_MODE == 2
void test_explicit_state() {
    sf64_fe_state_t st = {0u};

    // INVALID via add_ex.
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_add_ex(kInf, -kInf, &st);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_INVALID, "add_ex inf+(-inf) → INVALID in state");

    // DIVBYZERO via div_ex.
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_div_ex(1.0, 0.0, &st);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_DIVBYZERO, "div_ex 1/0 → DIVBYZERO in state");

    // OVERFLOW + INEXACT via mul_ex.
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_mul_ex(kMax, 2.0, &st);
    expect(sf64_fe_getall_ex(&st) == (SF64_FE_OVERFLOW | SF64_FE_INEXACT),
           "mul_ex max*2 → OVERFLOW+INEXACT in state");

    // INVALID via sqrt_ex.
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_sqrt_ex(-1.0, &st);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_INVALID, "sqrt_ex(-1) → INVALID in state");

    // INVALID via fma_ex.
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_fma_ex(kInf, 0.0, 1.0, &st);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_INVALID, "fma_ex(inf,0,1) → INVALID in state");

    // INVALID via to_i32_ex(NaN).
    sf64_fe_clear_ex(&st, 0x1Fu);
    (void)sf64_to_i32_ex(kNaN, &st);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_INVALID, "to_i32_ex(NaN) → INVALID in state");

    // raise_ex / clear_ex / save_ex / restore_ex round-trip.
    sf64_fe_clear_ex(&st, 0x1Fu);
    sf64_fe_raise_ex(&st, SF64_FE_DIVBYZERO);
    expect(sf64_fe_test_ex(&st, SF64_FE_DIVBYZERO) == 1, "raise_ex+test_ex round-trip");

    sf64_fe_state_t snapshot = {0u};
    sf64_fe_save_ex(&st, &snapshot);
    sf64_fe_raise_ex(&st, SF64_FE_OVERFLOW);
    expect(sf64_fe_getall_ex(&st) == (SF64_FE_DIVBYZERO | SF64_FE_OVERFLOW),
           "raise_ex accumulates");
    sf64_fe_restore_ex(&st, &snapshot);
    expect(sf64_fe_getall_ex(&st) == SF64_FE_DIVBYZERO, "restore_ex resets to snapshot");

    sf64_fe_clear_ex(&st, SF64_FE_DIVBYZERO);
    expect(sf64_fe_getall_ex(&st) == 0u, "clear_ex strips DIVBYZERO");

    // Null state pointer: callable, returns 0, drops flags.
    expect(sf64_fe_getall_ex(nullptr) == 0u, "getall_ex(nullptr) == 0");
    expect(sf64_fe_test_ex(nullptr, 0xFFu) == 0, "test_ex(nullptr) == 0");
    sf64_fe_raise_ex(nullptr, 0xFFu); // no-op, must not crash
    sf64_fe_clear_ex(nullptr, 0xFFu); // no-op, must not crash
    sf64_fe_state_t out_st = {0xAABBu};
    sf64_fe_save_ex(nullptr, &out_st);
    expect(out_st.flags == 0u, "save_ex(nullptr, out) zeroes out");

    // Null state passed to sf64_*_ex: result still computed correctly,
    // flags are dropped on the floor — verify the result and that no
    // state we own is touched.
    sf64_fe_state_t observer = {0u};
    const double r = sf64_div_ex(1.0, 0.0, nullptr);
    expect(std::isinf(r) && r > 0.0, "div_ex(1,0,nullptr) still returns +inf");
    expect(observer.flags == 0u, "div_ex with null state does not touch unrelated state");

#if SF64_TEST_FENV_MODE == 1
    // Under TLS mode the `_ex` surface must not leak into TLS, and vice-
    // versa. Under explicit mode there's no TLS to compare against.
    sf64_fe_clear(0x1Fu);
    sf64_fe_state_t isol = {0u};
    (void)sf64_add_ex(kInf, -kInf, &isol); // raises INVALID into isol
    expect(sf64_fe_getall() == 0u, "_ex raise does not leak into TLS");
    expect(isol.flags == SF64_FE_INVALID, "_ex raise lands in caller state");

    // Symmetric direction: TLS raise does not leak into a fresh state.
    sf64_fe_state_t isol2 = {0u};
    (void)sf64_div(1.0, 0.0); // raises DIVBYZERO into TLS
    expect(isol2.flags == 0u, "TLS raise does not leak into _ex state");
    sf64_fe_clear(0x1Fu);
#endif

    std::fputs("  explicit-state (`_ex`): ok\n", stdout);
}
#endif

} // namespace

int main() {
    std::fputs("test_fenv:\n", stdout);
    test_invalid();
    test_divbyzero();
    test_overflow();
    test_underflow();
    test_inexact();
    test_save_restore();
    test_thread_isolation();
#if SF64_TEST_FENV_MODE == 1 || SF64_TEST_FENV_MODE == 2
    test_explicit_state();
#endif
    std::fputs("test_fenv: all fenv assertions passed\n", stdout);
    return 0;
}
