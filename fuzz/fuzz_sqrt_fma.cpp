// libFuzzer target for sf64_sqrt and sf64_fma.
//
// Consumes up to 24 bytes (three f64 bit-patterns).  Exercises sqrt(x)
// and fma(a,b,c) and checks IEEE-754 special-case invariants.  ULP
// regressions are out of scope here (test_sqrt_fma_exact.cpp covers
// that); we're looking for sanitizer findings, traps, NaN-payload bugs,
// and qualitative divergences (e.g., sqrt(+inf) returning a finite value).
//
// Why 2^20 is the ULP budget (not the BIT_EXACT release tier of 0 ULP):
//   sf64_sqrt and sf64_fma are contracted BIT_EXACT vs the host FPU —
//   precision regressions are gated by test_sqrt_fma_exact.cpp against
//   host std::sqrt / std::fma, and by the TestFloat / MPFR oracle
//   stacks. This fuzz target is a CRASH-HUNT, not a precision oracle:
//   it chases sanitizer findings, UB-hit subnormal paths, NaN-payload
//   corruption, wild-pointer scribbles, and qualitative IEEE-754
//   invariant breaks (sqrt(+inf) finite, fma(NaN,*,*) non-NaN, etc.).
//   A 0-ULP budget would duplicate test_sqrt_fma_exact.cpp's work while
//   also firing on legitimate platform libm jitter (some older libm
//   builds return an FMA oracle that drifts 1 ULP under specific
//   rounding-mode regimes) — neither outcome is useful here. The 2^20
//   budget exists purely to catch "result shape completely wrong"
//   regressions that escape the structural invariant checks above;
//   exact-bit grading lives in test_sqrt_fma_exact.cpp.

#include <soft_fp64/soft_f64.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

double bits_to_double(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

uint64_t double_to_bits(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

bool is_nan(double d) {
    uint64_t b = double_to_bits(d);
    return ((b >> 52) & 0x7ffu) == 0x7ffu && (b & 0x000f'ffff'ffff'ffffULL) != 0;
}

[[noreturn]] void fuzz_fail(const char* /*msg*/) {
    __builtin_trap();
}

// |ulp_diff| helper using the host libm as oracle.  Only meaningful for
// finite, non-NaN inputs on both sides.
uint64_t ulp_diff(double x, double y) {
    if (is_nan(x) || is_nan(y))
        return 0;
    uint64_t xb = double_to_bits(x);
    uint64_t yb = double_to_bits(y);
    // Monotone ordering in unsigned space (avoid signed overflow UB).
    auto to_mono_u = [](uint64_t b) -> uint64_t {
        return (b & 0x8000'0000'0000'0000ULL)
                   ? (~b)                            // negative: bit-flip
                   : (b | 0x8000'0000'0000'0000ULL); // non-negative: set high bit
    };
    const uint64_t xm = to_mono_u(xb);
    const uint64_t ym = to_mono_u(yb);
    return (xm > ym) ? (xm - ym) : (ym - xm);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 24)
        return 0;

    uint64_t ab, bb, cb;
    std::memcpy(&ab, data, 8);
    std::memcpy(&bb, data + 8, 8);
    std::memcpy(&cb, data + 16, 8);

    const double a = bits_to_double(ab);
    const double b = bits_to_double(bb);
    const double c = bits_to_double(cb);

    // ---- sqrt ------------------------------------------------------------
    const double r_sqrt_a = sf64_sqrt(a);

    volatile uint64_t sink = 0;
    sink ^= double_to_bits(r_sqrt_a);

    // IEEE-754 specials for sqrt:
    //   sqrt(NaN)   = NaN
    //   sqrt(+inf)  = +inf
    //   sqrt(-0)    = -0
    //   sqrt(x<0)   = NaN (for finite negative x)
    if (is_nan(a)) {
        if (!is_nan(r_sqrt_a))
            fuzz_fail("sqrt(NaN) != NaN");
    } else if (__builtin_isinf(a) && a > 0.0) {
        if (!(__builtin_isinf(r_sqrt_a) && r_sqrt_a > 0.0))
            fuzz_fail("sqrt(+inf) != +inf");
    } else if (a < 0.0 && !__builtin_isinf(a)) {
        // Finite negative: must be NaN.
        if (!is_nan(r_sqrt_a))
            fuzz_fail("sqrt(<0 finite) not NaN");
    } else if (a == 0.0) {
        // sqrt(+/-0) = +/-0 (same sign).  Accept either magnitude zero.
        if (r_sqrt_a != 0.0)
            fuzz_fail("sqrt(0) not zero");
    }

    // sqrt(x*x) ≈ |x| for finite non-NaN x with |x| in a safe range (avoid
    // overflow in x*x).  We accept up to 2 ULP vs |x| — sqrt is correctly
    // rounded but x*x is not, so the round-trip can drift one ULP each
    // way.  This is a crash-hunt, not an accuracy test.
    if (!is_nan(a) && !__builtin_isinf(a) && std::fabs(a) > 0x1p-500 && std::fabs(a) < 0x1p500) {
        const double x2 = sf64_mul(a, a);
        const double s = sf64_sqrt(x2);
        const double abs_a = std::fabs(a);
        if (ulp_diff(s, abs_a) > 2) {
            // Non-fatal for legitimate boundary cases; the MPFR harness
            // will catch systematic drift.  Drop into a hard trap only on
            // egregious divergence (>1024 ULP) so we don't false-positive
            // on known double-rounding near subnormals.
            if (ulp_diff(s, abs_a) > 1024)
                fuzz_fail("sqrt(x*x) wildly != |x|");
        }
    }

    // ---- fma -------------------------------------------------------------
    const double r_fma = sf64_fma(a, b, c);
    sink ^= double_to_bits(r_fma);

    // NaN propagation: if any input is NaN, result is NaN.
    if (is_nan(a) || is_nan(b) || is_nan(c)) {
        if (!is_nan(r_fma))
            fuzz_fail("fma(NaN,*,*) not NaN");
    }

    // fma(a,b,c) vs libm std::fma oracle: must agree within a loose ULP
    // budget on finite inputs.  (Comparing against naive `a*b + c` is a
    // classic mistake — under genuine cancellation the two legitimately
    // differ by many ULPs; fma is the correctly-rounded answer, naive is
    // not.  So we use the libm oracle, which is correctly rounded on
    // modern platforms, and keep the budget loose because this is a
    // crash-hunt, not ULP grading.)
    if (!is_nan(a) && !is_nan(b) && !is_nan(c) && !__builtin_isinf(a) && !__builtin_isinf(b) &&
        !__builtin_isinf(c)) {
        const double oracle = std::fma(a, b, c);
        if (!is_nan(oracle) && !__builtin_isinf(oracle) && !__builtin_isinf(r_fma) &&
            !is_nan(r_fma)) {
            // Budget: 2^20 ULP.  On well-conditioned inputs sf64_fma
            // should be within 1 ULP of libm; the budget exists purely
            // to catch "result shape completely wrong" not 1-ULP bugs.
            if (ulp_diff(r_fma, oracle) > (uint64_t(1) << 20)) {
                fuzz_fail("fma vs libm::fma wildly divergent");
            }
        }
    }

    (void)sink;
    return 0;
}
