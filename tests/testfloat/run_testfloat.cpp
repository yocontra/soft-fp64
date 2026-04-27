// Berkeley TestFloat-3 oracle runner for soft-fp64.
//
// For each operation we spawn the vendored `testfloat_gen` binary (built
// by this subdir's CMakeLists) via popen(), parse its hex-formatted stdout
// vectors, call the corresponding `sf64_*` entry point, and compare
// bit-exact against the reference result that testfloat_gen produces via
// berkeley-softfloat-3.
//
// Output line formats produced by testfloat_gen (see
// berkeley-testfloat-3/source/writeCase_*.c):
//
//   unary  float → float     "<a16> <z16> <fl2>"            (e.g. f64_sqrt)
//   binary float × float → f "<a16> <b16> <z16> <fl2>"      (e.g. f64_add)
//   ternary f × f × f → f    "<a16> <b16> <c16> <z16> <fl2>" (f64_mulAdd)
//   cmp    f × f → bool      "<a16> <b16> <z1> <fl2>"       (e.g. f64_lt)
//   int   → float            "<aN> <z16> <fl2>"             (e.g. i32_to_f64)
//   float → int              "<a16> <zN> <fl2>"             (e.g. f64_to_i32)
//   f32   → f64              "<a8>  <z16> <fl2>"
//   f64   → f32              "<a16> <z8>  <fl2>"
//
// Exception flags (`fl2`) are parsed from each vector and compared against
// `sf64_fe_getall()` after the op. The runner clears flags between vectors
// so each comparison is local. Underflow convention is aligned by passing
// `-tininessbefore` to testfloat_gen (soft-fp64 detects tininess before
// rounding — the MIPS / RISC-V / RV64G / POWER default). Vectors where any
// input is a signaling NaN have their flag check skipped: `sf64_*` currently
// quiets sNaN inputs silently; sNaN→INVALID wiring is deferred to 1.2
// alongside `SOFT_FP64_SNAN_PROPAGATE` (see TODO.md).
//
// Any quiet-NaN result counts as "equal" to any expected quiet-NaN
// (matches the existing test_arithmetic_exact.cpp convention). Non-NaN
// results must match bit-exactly.
//
// Usage: ctest -R test_testfloat   (invokes this binary with no args)
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/rounding_mode.h"
#include "soft_fp64/soft_f64.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifndef SF64_TESTFLOAT_GEN_PATH
#error "SF64_TESTFLOAT_GEN_PATH must be defined by CMake"
#endif

// Under `disabled` fenv mode every sf64_fe_* is a no-op; fl2 comparison is
// inapplicable. The CMake build pipes the mode selection here; default to
// TLS-active so local one-off builds still exercise flag checks.
//
// Mode wiring:
//   1 (tls)      — use the TLS-backed sf64_* surface (bodies untouched).
//   0 (disabled) — every sf64_fe_* is a no-op; flag comparison is skipped
//                  and the runner reduces to a result-bit check.
//   2 (explicit) — TLS storage is gone, so the runner exercises the
//                  parallel sf64_*_ex / sf64_fe_*_ex ABI against a file-
//                  scope sf64_fe_state_t. The shim macros below redirect
//                  every call site without touching the body of each
//                  run_* helper. This is the production wiring frontends
//                  like Metal / WebGPU use — the runner mirrors it so
//                  the same 7.16M-vector corpus that gates TLS mode
//                  also gates explicit mode.
#ifndef SF64_TEST_FENV_MODE
#define SF64_TEST_FENV_MODE 1
#endif
#if SF64_TEST_FENV_MODE == 1
static constexpr bool kFlagsActive = true;
#elif SF64_TEST_FENV_MODE == 2
static constexpr bool kFlagsActive = true;
#else
static constexpr bool kFlagsActive = false;
#endif

#if SF64_TEST_FENV_MODE == 2
// File-scope state struct that every shim writes into. The runner is
// single-threaded so a non-thread-local global is safe.
static sf64_fe_state_t g_state = {0u};

// Wrapper functions that match the signatures of the original sf64_*
// surface so they can be both called directly AND passed as function
// pointers. These forward to the parallel `_ex` ABI plus the file-scope
// `g_state`. The runner's helper functions (`run_f64_binop_rmode`, etc.)
// receive these wrappers as `f64_binop_r fn` and call `fn(m, a, b)`,
// which preserves the existing call shape. Function-pointer storage of
// the wrapper is what closes the explicit-mode coverage gap that the
// macro-redirect approach (used for direct call sites) cannot reach.
namespace ex_shim {
inline void clear(unsigned mask) {
    sf64_fe_clear_ex(&g_state, mask);
}
inline unsigned getall(void) {
    return sf64_fe_getall_ex(&g_state);
}

inline double add(double a, double b) {
    return sf64_add_ex(a, b, &g_state);
}
inline double sub(double a, double b) {
    return sf64_sub_ex(a, b, &g_state);
}
inline double mul(double a, double b) {
    return sf64_mul_ex(a, b, &g_state);
}
inline double div_(double a, double b) {
    return sf64_div_ex(a, b, &g_state);
}
inline double sqrt_(double x) {
    return sf64_sqrt_ex(x, &g_state);
}
inline double fma_(double a, double b, double c) {
    return sf64_fma_ex(a, b, c, &g_state);
}
inline double from_f32(float x) {
    return sf64_from_f32_ex(x, &g_state);
}
inline float to_f32(double x) {
    return sf64_to_f32_ex(x, &g_state);
}
inline double from_i8(int8_t x) {
    return sf64_from_i8_ex(x, &g_state);
}
inline double from_i16(int16_t x) {
    return sf64_from_i16_ex(x, &g_state);
}
inline double from_i32(int32_t x) {
    return sf64_from_i32_ex(x, &g_state);
}
inline double from_i64(int64_t x) {
    return sf64_from_i64_ex(x, &g_state);
}
inline double from_u8(uint8_t x) {
    return sf64_from_u8_ex(x, &g_state);
}
inline double from_u16(uint16_t x) {
    return sf64_from_u16_ex(x, &g_state);
}
inline double from_u32(uint32_t x) {
    return sf64_from_u32_ex(x, &g_state);
}
inline double from_u64(uint64_t x) {
    return sf64_from_u64_ex(x, &g_state);
}
inline int8_t to_i8(double x) {
    return sf64_to_i8_ex(x, &g_state);
}
inline int16_t to_i16(double x) {
    return sf64_to_i16_ex(x, &g_state);
}
inline int32_t to_i32(double x) {
    return sf64_to_i32_ex(x, &g_state);
}
inline int64_t to_i64(double x) {
    return sf64_to_i64_ex(x, &g_state);
}
inline uint8_t to_u8(double x) {
    return sf64_to_u8_ex(x, &g_state);
}
inline uint16_t to_u16(double x) {
    return sf64_to_u16_ex(x, &g_state);
}
inline uint32_t to_u32(double x) {
    return sf64_to_u32_ex(x, &g_state);
}
inline uint64_t to_u64(double x) {
    return sf64_to_u64_ex(x, &g_state);
}
inline double add_r(sf64_rounding_mode m, double a, double b) {
    return sf64_add_r_ex(m, a, b, &g_state);
}
inline double sub_r(sf64_rounding_mode m, double a, double b) {
    return sf64_sub_r_ex(m, a, b, &g_state);
}
inline double mul_r(sf64_rounding_mode m, double a, double b) {
    return sf64_mul_r_ex(m, a, b, &g_state);
}
inline double div_r(sf64_rounding_mode m, double a, double b) {
    return sf64_div_r_ex(m, a, b, &g_state);
}
inline double sqrt_r(sf64_rounding_mode m, double x) {
    return sf64_sqrt_r_ex(m, x, &g_state);
}
inline double fma_r(sf64_rounding_mode m, double a, double b, double c) {
    return sf64_fma_r_ex(m, a, b, c, &g_state);
}
inline float to_f32_r(sf64_rounding_mode m, double x) {
    return sf64_to_f32_r_ex(m, x, &g_state);
}
inline int8_t to_i8_r(sf64_rounding_mode m, double x) {
    return sf64_to_i8_r_ex(m, x, &g_state);
}
inline int16_t to_i16_r(sf64_rounding_mode m, double x) {
    return sf64_to_i16_r_ex(m, x, &g_state);
}
inline int32_t to_i32_r(sf64_rounding_mode m, double x) {
    return sf64_to_i32_r_ex(m, x, &g_state);
}
inline int64_t to_i64_r(sf64_rounding_mode m, double x) {
    return sf64_to_i64_r_ex(m, x, &g_state);
}
inline uint8_t to_u8_r(sf64_rounding_mode m, double x) {
    return sf64_to_u8_r_ex(m, x, &g_state);
}
inline uint16_t to_u16_r(sf64_rounding_mode m, double x) {
    return sf64_to_u16_r_ex(m, x, &g_state);
}
inline uint32_t to_u32_r(sf64_rounding_mode m, double x) {
    return sf64_to_u32_r_ex(m, x, &g_state);
}
inline uint64_t to_u64_r(sf64_rounding_mode m, double x) {
    return sf64_to_u64_r_ex(m, x, &g_state);
}
} // namespace ex_shim

// Now redirect every TU-internal call site by macro. The wrappers above
// are referenced both by direct call (where the macro substitution
// fires) and by bare-identifier function-pointer passing (where the
// macro fires too because there's no following parenthesis test in the
// preprocessor — bare-identifier macros without args do expand). Note
// each macro is defined as a bare token (`sf64_fe_clear` →
// `ex_shim::clear`) rather than a function-call shape, so it expands
// uniformly in both contexts. That's exactly the case we need: the
// rmode dispatch in main() calls `run_f64_binop_rmode("f64_add",
// sf64_add_r, ...)` with `sf64_add_r` as a function pointer; under
// explicit mode this expands to `ex_shim::add_r` — the wrapper —
// which has the same signature.
#define sf64_fe_clear ex_shim::clear
#define sf64_fe_getall ex_shim::getall
#define sf64_add ex_shim::add
#define sf64_sub ex_shim::sub
#define sf64_mul ex_shim::mul
#define sf64_div ex_shim::div_
#define sf64_sqrt ex_shim::sqrt_
#define sf64_fma ex_shim::fma_
#define sf64_from_f32 ex_shim::from_f32
#define sf64_to_f32 ex_shim::to_f32
#define sf64_from_i8 ex_shim::from_i8
#define sf64_from_i16 ex_shim::from_i16
#define sf64_from_i32 ex_shim::from_i32
#define sf64_from_i64 ex_shim::from_i64
#define sf64_from_u8 ex_shim::from_u8
#define sf64_from_u16 ex_shim::from_u16
#define sf64_from_u32 ex_shim::from_u32
#define sf64_from_u64 ex_shim::from_u64
#define sf64_to_i8 ex_shim::to_i8
#define sf64_to_i16 ex_shim::to_i16
#define sf64_to_i32 ex_shim::to_i32
#define sf64_to_i64 ex_shim::to_i64
#define sf64_to_u8 ex_shim::to_u8
#define sf64_to_u16 ex_shim::to_u16
#define sf64_to_u32 ex_shim::to_u32
#define sf64_to_u64 ex_shim::to_u64
#define sf64_add_r ex_shim::add_r
#define sf64_sub_r ex_shim::sub_r
#define sf64_mul_r ex_shim::mul_r
#define sf64_div_r ex_shim::div_r
#define sf64_sqrt_r ex_shim::sqrt_r
#define sf64_fma_r ex_shim::fma_r
#define sf64_to_f32_r ex_shim::to_f32_r
#define sf64_to_i8_r ex_shim::to_i8_r
#define sf64_to_i16_r ex_shim::to_i16_r
#define sf64_to_i32_r ex_shim::to_i32_r
#define sf64_to_i64_r ex_shim::to_i64_r
#define sf64_to_u8_r ex_shim::to_u8_r
#define sf64_to_u16_r ex_shim::to_u16_r
#define sf64_to_u32_r ex_shim::to_u32_r
#define sf64_to_u64_r ex_shim::to_u64_r
#endif

// ---- bit-cast helpers --------------------------------------------------

static inline uint64_t bits(double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}
static inline double from_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}
static inline uint32_t bits_f32(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}
static inline float f32_from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

static inline bool nan_equiv(double got, double expect) {
    // NaN payloads can legitimately differ between impls; collapse.
    if (std::isnan(got) && std::isnan(expect))
        return true;
    return bits(got) == bits(expect);
}
static inline bool nan_equiv_f32(float got, float expect) {
    if (std::isnan(got) && std::isnan(expect))
        return true;
    return bits_f32(got) == bits_f32(expect);
}

// ---- fl2 helpers -------------------------------------------------------

// Berkeley softfloat fl2 bit layout (softfloat.h:85):
//   inexact=1, underflow=2, overflow=4, infinite(divbyzero)=8, invalid=16.
// sf64 layout (soft_f64.h): invalid=1, divbyzero=2, overflow=4, underflow=8,
// inexact=16 — bit-compatible with <fenv.h>. Map between the two.
static inline unsigned sf_flags_from_softfloat(uint64_t fl2) {
    unsigned out = 0u;
    if (fl2 & 0x01u)
        out |= SF64_FE_INEXACT;
    if (fl2 & 0x02u)
        out |= SF64_FE_UNDERFLOW;
    if (fl2 & 0x04u)
        out |= SF64_FE_OVERFLOW;
    if (fl2 & 0x08u)
        out |= SF64_FE_DIVBYZERO;
    if (fl2 & 0x10u)
        out |= SF64_FE_INVALID;
    return out;
}

// ---- process/stream launch --------------------------------------------

struct Proc {
    FILE* fp = nullptr;
    std::string cmd;

    bool open(const std::string& c) {
        cmd = c;
        fp = popen(c.c_str(), "r");
        return fp != nullptr;
    }
    ~Proc() {
        if (fp)
            pclose(fp);
    }
};

// Build shell command. We trust our own path (CMake-generated absolute
// path to testfloat_gen). Extra flags are single tokens.
static std::string mk_cmd(const char* op, const std::vector<const char*>& extra) {
    std::string s = "\"" SF64_TESTFLOAT_GEN_PATH "\"";
    for (const char* f : extra) {
        s += " ";
        s += f;
    }
    s += " ";
    s += op;
    // Discard stderr so a progress banner from upstream (if any) doesn't
    // interleave; we only care about stdout hex vectors.
    s += " 2>/dev/null";
    return s;
}

// ---- failure reporting -------------------------------------------------

struct Stats {
    uint64_t total = 0;
    uint64_t per_op = 0;
};

[[noreturn]] static void fail(const char* op, uint64_t lineno, const std::string& line,
                              const char* detail) {
    std::fprintf(stderr, "FAIL[%s] line=%" PRIu64 " detail=%s\n  input=%s\n", op, lineno, detail,
                 line.c_str());
    std::abort();
}

// ---- hex parsing -------------------------------------------------------

// Parse N whitespace-separated hex tokens from `line`. Returns false on
// malformed input. Each token written as uint64 (caller truncates).
static bool parse_hex_n(const char* line, int n, uint64_t out[]) {
    const char* p = line;
    for (int i = 0; i < n; ++i) {
        while (*p == ' ' || *p == '\t')
            ++p;
        if (!*p || *p == '\n')
            return false;
        char* end = nullptr;
        unsigned long long v = std::strtoull(p, &end, 16);
        if (end == p)
            return false;
        out[i] = (uint64_t)v;
        p = end;
    }
    return true;
}

// ---- rounding-mode table -----------------------------------------------
//
// TestFloat-gen accepts five rounding-mode flags matching IEEE-754's five
// modes. `sf64_rounding_mode` maps onto them verbatim:
//
//   SF64_RNE -> -rnear_even     (nearest, ties to even)
//   SF64_RTZ -> -rminMag        (toward zero)
//   SF64_RUP -> -rmax           (toward +infinity)
//   SF64_RDN -> -rmin           (toward -infinity)
//   SF64_RNA -> -rnear_maxMag   (nearest, ties away from zero)
//
// The label is used in failure messages so a per-mode regression
// identifies itself unambiguously.
struct ModeRow {
    sf64_rounding_mode mode;
    const char* tf_flag; // testfloat_gen -rXXX flag
    const char* label;   // short mode name for printouts
};
static const ModeRow kModes[] = {
    {SF64_RNE, "-rnear_even", "RNE"},   {SF64_RTZ, "-rminMag", "RTZ"},
    {SF64_RUP, "-rmax", "RUP"},         {SF64_RDN, "-rmin", "RDN"},
    {SF64_RNA, "-rnear_maxMag", "RNA"},
};

// ---- per-op runners ----------------------------------------------------

// f64 binary: add/sub/mul/div/rem
typedef double (*f64_binop)(double, double);
typedef double (*f64_binop_r)(sf64_rounding_mode, double, double);
typedef double (*f64_unop_r)(sf64_rounding_mode, double);
typedef double (*f64_fma_r_fn)(sf64_rounding_mode, double, double, double);
typedef float (*f64_to_f32_r_fn)(sf64_rounding_mode, double);

static uint64_t run_f64_binop(const char* op, f64_binop fn, const std::vector<const char*>& flags) {
    Proc p;
    if (!p.open(mk_cmd(op, flags))) {
        std::fprintf(stderr, "FAIL[%s]: cannot spawn testfloat_gen\n", op);
        std::abort();
    }
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[4];
        if (!parse_hex_n(line, 4, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        double z = from_bits(v[2]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[3]);
        sf64_fe_clear(0x1Fu);
        double got = fn(a, b);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[2]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail),
                              "flags got=0x%02x expected=0x%02x result=0x%016" PRIx64, got_flags,
                              expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// f64 binary, value-only — no flag verification. Used for ops whose
// flag plumbing isn't reachable under the current build mode (e.g.
// sf64_remainder under SOFT_FP64_FENV=explicit, which lives in the sleef
// TU and uses the TLS SF64_FE_RAISE macro). The bit-exact result corpus
// still runs at full size; only the fl2 comparison is skipped.
static uint64_t run_f64_binop_no_flags(const char* op, f64_binop fn,
                                       const std::vector<const char*>& flags) {
    Proc p;
    if (!p.open(mk_cmd(op, flags))) {
        std::fprintf(stderr, "FAIL[%s]: cannot spawn testfloat_gen\n", op);
        std::abort();
    }
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[4];
        if (!parse_hex_n(line, 4, v))
            fail(op, n, line, "parse");
        const double a = from_bits(v[0]);
        const double b = from_bits(v[1]);
        const double z = from_bits(v[2]);
        const double got = fn(a, b);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[2]);
            fail(op, n, line, detail);
        }
        ++n;
    }
    return n;
}

// f64 unary float→float: sqrt
static uint64_t run_f64_unop(const char* op, double (*fn)(double),
                             const std::vector<const char*>& flags) {
    Proc p;
    if (!p.open(mk_cmd(op, flags)))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double z = from_bits(v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        double got = fn(a);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[1]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail),
                              "flags got=0x%02x expected=0x%02x result=0x%016" PRIx64, got_flags,
                              expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// f64_mulAdd: (a*b)+c
static uint64_t run_f64_mulAdd(uint64_t n_cases) {
    const char* op = "f64_mulAdd";
    char ncount[32];
    std::snprintf(ncount, sizeof(ncount), "%" PRIu64, n_cases);
    Proc p;
    if (!p.open(mk_cmd(op, {"-n", ncount, "-tininessbefore"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[5];
        if (!parse_hex_n(line, 5, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        double c = from_bits(v[2]);
        double z = from_bits(v[3]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[4]);
        sf64_fe_clear(0x1Fu);
        double got = sf64_fma(a, b, c);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[3]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail),
                              "flags got=0x%02x expected=0x%02x result=0x%016" PRIx64, got_flags,
                              expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// Compare: f64 × f64 → bool. sf64_fcmp predicate supplied by caller.
static uint64_t run_f64_cmp(const char* op, int pred) {
    Proc p;
    if (!p.open(mk_cmd(op, {})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        // For cmp the expected result is a 1-hex-digit bool in field 2.
        // The format still parses as hex via parse_hex_n — 3 tokens: a, b, z.
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        int z = (int)(v[2] & 1);
        int got = sf64_fcmp(a, b, pred);
        if (got != z) {
            char detail[160];
            std::snprintf(detail, sizeof(detail),
                          "a=0x%016" PRIx64 " b=0x%016" PRIx64 " pred=%d got=%d expect=%d", v[0],
                          v[1], pred, got, z);
            fail(op, n, line, detail);
        }
        ++n;
    }
    return n;
}

// Compare: f64 × f64 → bool driven by a *derived* oracle that combines one
// or two TestFloat-generated primitives with a pure-arithmetic composition.
// Used for the 10 FCmp predicates that don't land directly on an eq/lt/le
// testfloat op:
//
//   OGT    = !UNO && !OLE          (from f64_le: ogt = !unordered && !le)
//   OGE    = !UNO && !OLT          (from f64_lt)
//   ONE    = !UNO && !OEQ          (from f64_eq)
//   ORD    =  !isnan(a) && !isnan(b)
//   UNO    =   isnan(a) ||  isnan(b)
//   UEQ    =   UNO || OEQ
//   UGT    =   UNO || !OLE
//   UGE    =   UNO || !OLT
//   ULT    =   UNO || !OGE  =  UNO || !(ORD && !OLT)  =  UNO || OLT
//   ULE    =   UNO || OLE
//   UNE    =   UNO || !OEQ
//
// We build the oracle from raw inputs (NaN check + TestFloat's f64_eq /
// f64_lt / f64_le scalar booleans). TestFloat's f64_le generator emits the
// oracle for OLE directly, so we stream it and post-process.
enum class CmpSource {
    Eq, // op = f64_eq  -> oracle bool is OEQ(a,b)
    Lt, // op = f64_lt  -> oracle bool is OLT(a,b)
    Le, // op = f64_le  -> oracle bool is OLE(a,b)
};

static bool is_nan_bits(uint64_t bits) {
    const uint64_t exp = (bits >> 52) & 0x7ffULL;
    const uint64_t mant = bits & 0x000fffffffffffffULL;
    return exp == 0x7ffULL && mant != 0;
}

// Compute the expected fcmp bool for predicate `pred` from the raw inputs
// and a single source oracle bit. `src_bool` is the boolean TestFloat
// emitted for the generator op identified by `src`.
static int derive_pred(int pred, uint64_t a_bits, uint64_t b_bits, CmpSource src, int src_bool) {
    const bool uno = is_nan_bits(a_bits) || is_nan_bits(b_bits);
    const bool ord = !uno;
    bool oeq = false, olt = false, ole = false;
    switch (src) {
    case CmpSource::Eq:
        oeq = (src_bool != 0);
        break;
    case CmpSource::Lt:
        olt = (src_bool != 0);
        break;
    case CmpSource::Le:
        ole = (src_bool != 0);
        break;
    }
    switch (pred) {
    case 2 /*OGT*/:
        return (ord && !ole) ? 1 : 0;
    case 3 /*OGE*/:
        return (ord && !olt) ? 1 : 0;
    case 6 /*ONE*/:
        return (ord && !oeq) ? 1 : 0;
    case 7 /*ORD*/:
        return ord ? 1 : 0;
    case 8 /*UNO*/:
        return uno ? 1 : 0;
    case 9 /*UEQ*/:
        return (uno || oeq) ? 1 : 0;
    case 10 /*UGT*/:
        return (uno || !ole) ? 1 : 0;
    case 11 /*UGE*/:
        return (uno || !olt) ? 1 : 0;
    case 12 /*ULT*/:
        return (uno || olt) ? 1 : 0;
    case 13 /*ULE*/:
        return (uno || ole) ? 1 : 0;
    case 14 /*UNE*/:
        return (uno || !oeq) ? 1 : 0;
    default:
        std::abort();
    }
}

static uint64_t run_f64_cmp_derived(const char* label, int pred, const char* gen_op,
                                    CmpSource src) {
    Proc p;
    if (!p.open(mk_cmd(gen_op, {})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(label, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        int src_bool = (int)(v[2] & 1);
        int z = derive_pred(pred, v[0], v[1], src, src_bool);
        int got = sf64_fcmp(a, b, pred);
        if (got != z) {
            char detail[192];
            std::snprintf(detail, sizeof(detail),
                          "a=0x%016" PRIx64 " b=0x%016" PRIx64
                          " pred=%d got=%d expect=%d (via %s=%d)",
                          v[0], v[1], pred, got, z, gen_op, src_bool);
            fail(label, n, line, detail);
        }
        ++n;
    }
    return n;
}

// FCMP_FALSE (0) and FCMP_TRUE (15) have no TestFloat counterpart — assert
// the soft-fp64 output is the corresponding constant on a deterministic
// sweep that covers each IEEE-754 class (normal, subnormal, zero, inf, NaN,
// signed variants, boundary values).
static uint64_t run_f64_cmp_const(int pred, int expect) {
    static const uint64_t kSeeds[] = {
        0x0000000000000000ULL, // +0
        0x8000000000000000ULL, // -0
        0x0000000000000001ULL, // +denorm_min
        0x8000000000000001ULL, // -denorm_min
        0x000fffffffffffffULL, // +max subnormal
        0x0010000000000000ULL, // +min normal
        0x3ff0000000000000ULL, //  +1.0
        0xbff0000000000000ULL, //  -1.0
        0x3fe0000000000000ULL, //  +0.5
        0x4000000000000000ULL, //  +2.0
        0x7fefffffffffffffULL, // +max normal
        0xffefffffffffffffULL, // -max normal
        0x7ff0000000000000ULL, // +inf
        0xfff0000000000000ULL, // -inf
        0x7ff8000000000000ULL, // quiet NaN
        0xfff8000000000000ULL, // quiet NaN (sign)
        0x7ff0000000000001ULL, // signaling NaN
        0x4024000000000000ULL, // +10
        0xc024000000000000ULL, // -10
    };
    const size_t N = sizeof(kSeeds) / sizeof(kSeeds[0]);
    uint64_t n = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double a = from_bits(kSeeds[i]);
            double b = from_bits(kSeeds[j]);
            int got = sf64_fcmp(a, b, pred);
            if (got != expect) {
                char line[96];
                std::snprintf(line, sizeof(line), "a=0x%016" PRIx64 " b=0x%016" PRIx64, kSeeds[i],
                              kSeeds[j]);
                char detail[160];
                std::snprintf(detail, sizeof(detail), "pred=%d got=%d expect=%d (constant)", pred,
                              got, expect);
                fail(pred == 0 ? "FCMP_FALSE" : "FCMP_TRUE", n, line, detail);
            }
            ++n;
        }
    }
    return n;
}

// Integer → f64.
template <typename IntT> static uint64_t run_int_to_f64(const char* op, double (*fn)(IntT)) {
    Proc p;
    if (!p.open(mk_cmd(op, {})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        IntT a = (IntT)v[0];
        double z = from_bits(v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        double got = fn(a);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[1]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail), "flags got=0x%02x expected=0x%02x", got_flags,
                              expected_flags);
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// f64 → integer.
//
// sf64_to_{i,u}{32,64} is defined as C-style truncation toward zero
// (`(IntT)double` semantics; also what LLVM `fptosi`/`fptoui` emits).
// TestFloat's default rounding mode is round-to-nearest-even, so we pass
// `-rminMag` to make its reference match truncation.
//
// For inputs outside the destination range (±inf, NaN, magnitude beyond
// the type's representable bounds), Berkeley softfloat returns a
// platform-defined "invalid conversion" sentinel — for 8086-SSE that's
// INT_MIN for signed and UINT_MAX for unsigned. sf64_*, matching the
// LLVM/C cast convention, saturates to the representable endpoint in
// the input's direction. Both behaviours are defensible but they
// disagree on sentinel identity, so we only bit-compare results that
// fall in the *representable* range (the finite in-range cases exercise
// rounding logic, which is the whole point of the oracle).
template <typename IntT, IntT (*Fn)(double)>
static uint64_t run_f64_to_int(const char* op, double tmin, double tmax_plus_one) {
    // `-exact` makes Berkeley raise INEXACT on lossy int conversions (IEEE
    // §5.4.1 default). Without it, softfloat's `-notexact` default silences
    // INEXACT on all integer rounding, which doesn't match sf64's IEEE-
    // conformant behavior.
    Proc p;
    if (!p.open(mk_cmd(op, {"-rminMag", "-exact"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t skipped_oor = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        const bool in_range = !std::isnan(a) && std::isfinite(a) && a >= tmin && a < tmax_plus_one;
        if (!in_range) {
            ++skipped_oor;
            ++n;
            continue;
        }
        IntT z = (IntT)v[1];
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        IntT got = Fn(a);
        if (got != z) {
            char detail[160];
            std::snprintf(detail, sizeof(detail),
                          "a=0x%016" PRIx64 " got=0x%016" PRIx64 " expect=0x%016" PRIx64, v[0],
                          (uint64_t)(uint64_t)got, (uint64_t)(uint64_t)z);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail),
                              "a=0x%016" PRIx64 " flags got=0x%02x expected=0x%02x", v[0],
                              got_flags, expected_flags);
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    std::fprintf(stderr,
                 "    [%s] %" PRIu64 " / %" PRIu64 " vectors in-range checked (%" PRIu64
                 " skipped OOR)\n",
                 op, n - skipped_oor, n, skipped_oor);
    return n;
}

// Narrow int → f64. TestFloat 3e has no generator for i8/i16/u8/u16 so we
// clamp the TestFloat-emitted `i32_to_f64` / `ui32_to_f64` inputs into the
// narrow range and cross-check against *both* (a) the oracle's f64 output
// narrowed the same way, and (b) a C++ static_cast round-trip. For inputs
// outside the narrow range we just skip (narrowed input is a different
// vector already covered elsewhere, so no information is lost).
template <typename NarrowT, typename WideT, double (*Fn)(NarrowT)>
static uint64_t run_narrow_int_to_f64(const char* label, const char* gen_op, WideT narrow_min,
                                      WideT narrow_max) {
    Proc p;
    if (!p.open(mk_cmd(gen_op, {})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t skipped = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[2];
        if (!parse_hex_n(line, 2, v))
            fail(label, n, line, "parse");
        WideT wide = (WideT)v[0];
        if (wide < narrow_min || wide > narrow_max) {
            ++skipped;
            continue;
        }
        NarrowT narrow = (NarrowT)wide;
        if (kFlagsActive) {
            sf64_fe_clear(0x1Fu);
        }
        double got = Fn(narrow);
        // Narrow int → f64 is exact by construction: every i8/i16/u8/u16
        // value fits inside the 53-bit f64 significand with zero rounding
        // loss, so no exception flag may be raised.
        if (kFlagsActive && sf64_fe_getall() != 0u) {
            char detail[192];
            std::snprintf(detail, sizeof(detail),
                          "narrow=0x%llx got=0x%016" PRIx64 " spurious fl=0x%02x",
                          (long long)(int64_t)wide, bits(got), sf64_fe_getall());
            fail(label, n, line, detail);
        }
        // Oracle 1: TestFloat's i32/u32→f64 result (always exact for narrow
        // range since |narrow| < 2^31, so no rounding).
        double oracle = from_bits(v[1]);
        // Oracle 2: plain static_cast round-trip (host FP exact for exact-
        // representable integers < 2^53).
        double cast_oracle = static_cast<double>(narrow);
        if (!nan_equiv(got, oracle) || !nan_equiv(got, cast_oracle)) {
            char detail[192];
            std::snprintf(detail, sizeof(detail),
                          "narrow=0x%llx got=0x%016" PRIx64 " tf_oracle=0x%016" PRIx64
                          " cast_oracle=0x%016" PRIx64,
                          (long long)(int64_t)wide, bits(got), v[1], bits(cast_oracle));
            fail(label, n, line, detail);
        }
        ++n;
    }
    (void)skipped;
    return n;
}

// Narrow f64 → int. TestFloat's `f64_to_i32` / `f64_to_ui32` emits f64
// inputs. We accept only those whose truncated value lands inside the
// narrow destination range and cross-check both oracles: (a) TestFloat's
// int32 result truncated to NarrowT must match, (b) a C++ static_cast
// round-trip from the same f64 to NarrowT must match. Out-of-range inputs
// and NaN/inf are skipped (matches the policy for wide f64→int).
template <typename NarrowT, typename WideT, NarrowT (*Fn)(double), WideT (*WideFn)(double)>
static uint64_t run_f64_to_narrow_int(const char* label, const char* gen_op, double tmin,
                                      double tmax_plus_one) {
    Proc p;
    if (!p.open(mk_cmd(gen_op, {"-rminMag"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t skipped = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[2];
        if (!parse_hex_n(line, 2, v))
            fail(label, n, line, "parse");
        double a = from_bits(v[0]);
        const bool in_range = !std::isnan(a) && std::isfinite(a) && a >= tmin && a < tmax_plus_one;
        if (!in_range) {
            ++skipped;
            continue;
        }
        if (kFlagsActive) {
            sf64_fe_clear(0x1Fu);
        }
        NarrowT got = Fn(a);
        const unsigned got_flags = kFlagsActive ? sf64_fe_getall() : 0u;
        // IEEE §7.1: integer conversion raises INEXACT iff round-trip
        // through double loses bits. In-range here means no OVERFLOW /
        // UNDERFLOW, and the NaN/Inf gate above means no INVALID. Only
        // INEXACT can fire, and only when `(double)got != a`.
        if (kFlagsActive) {
            const bool lossy = (static_cast<double>(got) != a);
            const unsigned expected = lossy ? SF64_FE_INEXACT : 0u;
            if (got_flags != expected) {
                char detail[192];
                std::snprintf(detail, sizeof(detail),
                              "a=0x%016" PRIx64 " got=0x%llx got_fl=0x%02x expected_fl=0x%02x",
                              v[0], (long long)(int64_t)got, got_flags, expected);
                fail(label, n, line, detail);
            }
        }
        // TestFloat oracle: use the wide sf64_to_i32/u32 path applied to
        // the same input, then truncate. WideFn is used rather than the
        // stored v[1] because TestFloat emits sentinels for OOR, and the
        // narrow path is the sf64_* canonical truncation by definition
        // (not the Berkeley SSE sentinel).
        if (kFlagsActive) {
            sf64_fe_clear(0x1Fu); // don't let WideFn's flags leak into next iter
        }
        WideT wide = WideFn(a);
        NarrowT tf_oracle = static_cast<NarrowT>(wide);
        NarrowT cast_oracle = static_cast<NarrowT>(a);
        if (got != tf_oracle || got != cast_oracle) {
            char detail[192];
            std::snprintf(detail, sizeof(detail),
                          "a=0x%016" PRIx64 " got=0x%llx tf_oracle=0x%llx cast_oracle=0x%llx", v[0],
                          (long long)(int64_t)got, (long long)(int64_t)tf_oracle,
                          (long long)(int64_t)cast_oracle);
            fail(label, n, line, detail);
        }
        ++n;
    }
    (void)skipped;
    return n;
}

// f32 → f64.
static uint64_t run_f32_to_f64() {
    const char* op = "f32_to_f64";
    Proc p;
    if (!p.open(mk_cmd(op, {})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        float a = f32_from_bits((uint32_t)v[0]);
        double z = from_bits(v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        double got = sf64_from_f32(a);
        if (!nan_equiv(got, z)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          bits(got), v[1]);
            fail(op, n, line, detail);
        }
        // f32→f64 is always exact (mantissa fits, exp widens), so the only
        // flag Berkeley ever raises is INVALID on sNaN input — sf64 raises
        // INVALID on sNaN per IEEE 754 §6.2 / §7.2, so the comparison is
        // direct.
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[160];
                std::snprintf(detail, sizeof(detail), "a=0x%08x flags got=0x%02x expected=0x%02x",
                              (uint32_t)v[0], got_flags, expected_flags);
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// f64 → f32.
//
// Bit-exact comparison at round-to-nearest-even. Any mismatch aborts.
static uint64_t run_f64_to_f32() {
    const char* op = "f64_to_f32";
    Proc p;
    if (!p.open(mk_cmd(op, {"-rnear_even", "-tininessbefore"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t mismatches = 0;
    uint64_t flag_mismatches = 0;
    char first_detail[256] = {0};
    char first_flag_detail[256] = {0};
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        float z = f32_from_bits((uint32_t)v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        float got = sf64_to_f32(a);
        if (!nan_equiv_f32(got, z)) {
            if (mismatches < 5) {
                std::fprintf(stderr,
                             "  [%s] mismatch line=%" PRIu64 " a=0x%016" PRIx64
                             " got=0x%08x expect=0x%08x\n",
                             op, n, v[0], bits_f32(got), (uint32_t)v[1]);
            }
            if (mismatches == 0) {
                std::snprintf(first_detail, sizeof(first_detail),
                              "line=%" PRIu64 " a=0x%016" PRIx64 " got=0x%08x expect=0x%08x", n,
                              v[0], bits_f32(got), (uint32_t)v[1]);
            }
            ++mismatches;
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                if (flag_mismatches == 0) {
                    std::snprintf(first_flag_detail, sizeof(first_flag_detail),
                                  "line=%" PRIu64 " a=0x%016" PRIx64
                                  " flags got=0x%02x expected=0x%02x",
                                  n, v[0], got_flags, expected_flags);
                }
                ++flag_mismatches;
            }
        }
        ++n;
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL[%s] %" PRIu64 " / %" PRIu64 " mismatches; first: %s\n", op,
                     mismatches, n, first_detail);
        std::abort();
    }
    if (flag_mismatches) {
        std::fprintf(stderr, "FAIL[%s] %" PRIu64 " / %" PRIu64 " flag mismatches; first: %s\n", op,
                     flag_mismatches, n, first_flag_detail);
        std::abort();
    }
    return n;
}

// ---- per-mode runners (sf64_*_r surface) -------------------------------
//
// For each (op, mode) pair TestFloat generates a fresh corpus under the
// matching -r<flag> and we call sf64_*_r(mode, ...). Non-arithmetic ops
// (compare) don't depend on rounding mode and are not mode-looped.

static uint64_t run_f64_binop_rmode(const char* op, f64_binop_r fn, sf64_rounding_mode m,
                                    const char* mode_label, const ModeRow& row) {
    Proc p;
    const std::vector<const char*> flags = {"-tininessbefore", row.tf_flag};
    if (!p.open(mk_cmd(op, flags))) {
        std::fprintf(stderr, "FAIL[%s-%s]: cannot spawn testfloat_gen\n", op, mode_label);
        std::abort();
    }
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[4];
        if (!parse_hex_n(line, 4, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        double z = from_bits(v[2]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[3]);
        sf64_fe_clear(0x1Fu);
        double got = fn(m, a, b);
        if (!nan_equiv(got, z)) {
            char detail[160];
            std::snprintf(detail, sizeof(detail),
                          "mode=%s got=0x%016" PRIx64 " expect=0x%016" PRIx64, mode_label,
                          bits(got), v[2]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[192];
                std::snprintf(detail, sizeof(detail),
                              "mode=%s flags got=0x%02x expected=0x%02x result=0x%016" PRIx64,
                              mode_label, got_flags, expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

static uint64_t run_f64_unop_rmode(const char* op, f64_unop_r fn, sf64_rounding_mode m,
                                   const char* mode_label, const ModeRow& row) {
    Proc p;
    const std::vector<const char*> flags = {"-tininessbefore", row.tf_flag};
    if (!p.open(mk_cmd(op, flags)))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double z = from_bits(v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        double got = fn(m, a);
        if (!nan_equiv(got, z)) {
            char detail[160];
            std::snprintf(detail, sizeof(detail),
                          "mode=%s got=0x%016" PRIx64 " expect=0x%016" PRIx64, mode_label,
                          bits(got), v[1]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[192];
                std::snprintf(detail, sizeof(detail),
                              "mode=%s flags got=0x%02x expected=0x%02x result=0x%016" PRIx64,
                              mode_label, got_flags, expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

static uint64_t run_f64_mulAdd_rmode(uint64_t n_cases, sf64_rounding_mode m, const char* mode_label,
                                     const ModeRow& row) {
    const char* op = "f64_mulAdd";
    char ncount[32];
    std::snprintf(ncount, sizeof(ncount), "%" PRIu64, n_cases);
    Proc p;
    if (!p.open(mk_cmd(op, {"-n", ncount, "-tininessbefore", row.tf_flag})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[5];
        if (!parse_hex_n(line, 5, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        double b = from_bits(v[1]);
        double c = from_bits(v[2]);
        double z = from_bits(v[3]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[4]);
        sf64_fe_clear(0x1Fu);
        double got = sf64_fma_r(m, a, b, c);
        if (!nan_equiv(got, z)) {
            char detail[160];
            std::snprintf(detail, sizeof(detail),
                          "mode=%s got=0x%016" PRIx64 " expect=0x%016" PRIx64, mode_label,
                          bits(got), v[3]);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[192];
                std::snprintf(detail, sizeof(detail),
                              "mode=%s flags got=0x%02x expected=0x%02x result=0x%016" PRIx64,
                              mode_label, got_flags, expected_flags, bits(got));
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    return n;
}

// f64 -> int under mode m. `-exact` preserved so TestFloat raises INEXACT
// on lossy int conversions (IEEE §5.4.1). We thread the mode through
// both sides — Berkeley's oracle rounds per `row.tf_flag` and we call
// `sf64_to_i*_r(m, ...)`. In-range check is done AFTER rounding under
// mode `m`: a value that rounds to INT_MAX + 1 under RNE but to
// INT_MAX under RTZ is in-range for RTZ only. Berkeley sends a
// saturation sentinel (INT_MIN on x86-SSE) for truly out-of-range
// vectors; we only bit-compare rows where the rounded value is
// representable and skip sentinel rows (the existing RNE leg
// documents this convention).
template <typename IntT, IntT (*Fn)(sf64_rounding_mode, double)>
static uint64_t run_f64_to_int_rmode(const char* op, double tmin, double tmax_plus_one,
                                     sf64_rounding_mode m, const char* mode_label,
                                     const ModeRow& row) {
    Proc p;
    if (!p.open(mk_cmd(op, {row.tf_flag, "-exact"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t skipped_oor = 0;
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        // Pre-filter obvious junk: NaN/inf, far-out-of-range. The tight
        // in-range check comes post-rounding below.
        const bool base_in_range =
            !std::isnan(a) && std::isfinite(a) && a >= tmin && a < tmax_plus_one;
        if (!base_in_range) {
            ++skipped_oor;
            ++n;
            continue;
        }
        // Post-rounding range: reject anything whose oracle result
        // equals Berkeley's saturation sentinel (INT_MIN for signed,
        // UINT_MAX for unsigned). Those vectors exercise the saturation
        // semantics where sf64_* and softfloat disagree by design.
        const IntT z = (IntT)v[1];
        const bool saturated_sentinel =
            (std::is_signed<IntT>::value)
                ? (z == std::numeric_limits<IntT>::min() && a > 0.0)
                : (z == std::numeric_limits<IntT>::max() && a > static_cast<double>(z));
        if (saturated_sentinel) {
            ++skipped_oor;
            ++n;
            continue;
        }
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        IntT got = Fn(m, a);
        if (got != z) {
            char detail[192];
            std::snprintf(detail, sizeof(detail),
                          "mode=%s a=0x%016" PRIx64 " got=0x%016" PRIx64 " expect=0x%016" PRIx64,
                          mode_label, v[0], (uint64_t)(uint64_t)got, (uint64_t)(uint64_t)z);
            fail(op, n, line, detail);
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                char detail[192];
                std::snprintf(detail, sizeof(detail),
                              "mode=%s a=0x%016" PRIx64 " flags got=0x%02x expected=0x%02x",
                              mode_label, v[0], got_flags, expected_flags);
                fail(op, n, line, detail);
            }
        }
        ++n;
    }
    std::fprintf(stderr,
                 "    [%s-%s] %" PRIu64 " / %" PRIu64 " vectors in-range checked (%" PRIu64
                 " skipped OOR)\n",
                 op, mode_label, n - skipped_oor, n, skipped_oor);
    return n;
}

// f64 -> f32 under mode m. Runs the runner under the requested testfloat
// flag and compares against sf64_to_f32_r(m, x).
static uint64_t run_f64_to_f32_rmode(sf64_rounding_mode m, const char* mode_label,
                                     const ModeRow& row) {
    const char* op = "f64_to_f32";
    Proc p;
    if (!p.open(mk_cmd(op, {row.tf_flag, "-tininessbefore"})))
        std::abort();
    char line[256];
    uint64_t n = 0;
    uint64_t mismatches = 0;
    uint64_t flag_mismatches = 0;
    char first_detail[256] = {0};
    char first_flag_detail[256] = {0};
    while (std::fgets(line, sizeof(line), p.fp)) {
        uint64_t v[3];
        if (!parse_hex_n(line, 3, v))
            fail(op, n, line, "parse");
        double a = from_bits(v[0]);
        float z = f32_from_bits((uint32_t)v[1]);
        const unsigned expected_flags = sf_flags_from_softfloat(v[2]);
        sf64_fe_clear(0x1Fu);
        float got = sf64_to_f32_r(m, a);
        if (!nan_equiv_f32(got, z)) {
            if (mismatches < 5) {
                std::fprintf(stderr,
                             "  [%s-%s] mismatch line=%" PRIu64 " a=0x%016" PRIx64
                             " got=0x%08x expect=0x%08x\n",
                             op, mode_label, n, v[0], bits_f32(got), (uint32_t)v[1]);
            }
            if (mismatches == 0) {
                std::snprintf(first_detail, sizeof(first_detail),
                              "mode=%s line=%" PRIu64 " a=0x%016" PRIx64
                              " got=0x%08x expect=0x%08x",
                              mode_label, n, v[0], bits_f32(got), (uint32_t)v[1]);
            }
            ++mismatches;
        }
        if (kFlagsActive) {
            const unsigned got_flags = sf64_fe_getall();
            if (got_flags != expected_flags) {
                if (flag_mismatches == 0) {
                    std::snprintf(first_flag_detail, sizeof(first_flag_detail),
                                  "mode=%s line=%" PRIu64 " a=0x%016" PRIx64
                                  " flags got=0x%02x expected=0x%02x",
                                  mode_label, n, v[0], got_flags, expected_flags);
                }
                ++flag_mismatches;
            }
        }
        ++n;
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL[%s-%s] %" PRIu64 " / %" PRIu64 " mismatches; first: %s\n", op,
                     mode_label, mismatches, n, first_detail);
        std::abort();
    }
    if (flag_mismatches) {
        std::fprintf(stderr, "FAIL[%s-%s] %" PRIu64 " / %" PRIu64 " flag mismatches; first: %s\n",
                     op, mode_label, flag_mismatches, n, first_flag_detail);
        std::abort();
    }
    return n;
}

// ---- dispatch table ----------------------------------------------------

int main() {
    Stats s;
    auto run = [&](const char* name, uint64_t n) {
        std::printf("  %-32s %" PRIu64 " vectors\n", name, n);
        s.total += n;
    };

    std::printf("TestFloat oracle: soft-fp64 bit-exact verification\n");
    std::printf("gen: %s\n", SF64_TESTFLOAT_GEN_PATH);

    // ---- arithmetic (default -level 1) ----------------------------------
    //
    // `-tininessbefore` aligns Berkeley's underflow-detection convention with
    // soft-fp64's (tiny-before-rounding — the MIPS/RISC-V/RV64G/POWER default).
    // Without this flag Berkeley detects tininess after rounding and emits a
    // different UNDERFLOW flag pattern for vectors rounding up from subnormal.
    run("f64_add", run_f64_binop("f64_add", sf64_add, {"-tininessbefore"}));
    run("f64_sub", run_f64_binop("f64_sub", sf64_sub, {"-tininessbefore"}));
    run("f64_mul", run_f64_binop("f64_mul", sf64_mul, {"-tininessbefore"}));
    run("f64_div", run_f64_binop("f64_div", sf64_div, {"-tininessbefore"}));
    // IEEE-754 `remainder` (round-to-nearest-even quotient) — lands via
    // sf64_remainder (fmod + tie-break). Rounding mode does not apply.
    //
    // Under explicit mode `sf64_remainder` lives in src/sleef/ and uses
    // the TLS-mode SF64_FE_RAISE macro for its INVALID raises (the
    // sleef path does not have an `_ex` variant — out of scope for the
    // arithmetic / sqrt / fma / convert ABI extension). The bit-exact
    // value check still runs; the flag check is suppressed via the
    // `_no_flags` runner so the result corpus stays exercised.
#if SF64_TEST_FENV_MODE == 2
    run("f64_rem (no-flag-check)",
        run_f64_binop_no_flags("f64_rem", sf64_remainder, {"-tininessbefore"}));
#else
    run("f64_rem", run_f64_binop("f64_rem", sf64_remainder, {"-tininessbefore"}));
#endif
    run("f64_sqrt", run_f64_unop("f64_sqrt", sf64_sqrt, {"-tininessbefore"}));

    // mulAdd has a hardcoded minimum of 6,133,248 vectors at level 1. The
    // runtime is ~15-30s depending on machine. Pass -n explicitly so we
    // document the chosen sample size.
    run("f64_mulAdd", run_f64_mulAdd(6133248));

    // ---- per-mode sf64_*_r surface --------------------------------------
    //
    // Re-run add/sub/mul/div/sqrt/fma/f64_to_i*/f64_to_f32 under each of
    // the four non-RNE rounding modes (RTZ, RUP, RDN, RNA) plus a
    // redundant RNE pass that exercises the explicit-mode entry points.
    // TestFloat regenerates its oracle per-mode via -rnear_even /
    // -rminMag / -rmax / -rmin / -rnear_maxMag; the runner dispatches
    // each vector through the matching sf64_*_r(mode, ...) call.
    //
    // `-tininessbefore` is preserved. `f64_mulAdd` at 6.1M vectors is
    // the slow leg here — roughly 15-30s per mode — so the total per-
    // mode pass stays bounded at ~3 minutes on Apple Silicon. See the
    // per-mode bit-exact MPFR sweep in `tests/mpfr/test_mpfr_diff.cpp`
    // for an independent oracle covering the same surface.
    //
    // mulAdd vector count per mode: 6,133,248 (matches TestFloat's
    // hard-coded level-1 minimum — the generator refuses smaller -n).
    // Total mulAdd coverage across all modes: 5 * 6M = 30M vectors,
    // ~90s added runtime on Apple Silicon.
    for (const ModeRow& row : kModes) {
        char label[64];

        std::snprintf(label, sizeof(label), "f64_add  [%s]", row.label);
        run(label, run_f64_binop_rmode("f64_add", sf64_add_r, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_sub  [%s]", row.label);
        run(label, run_f64_binop_rmode("f64_sub", sf64_sub_r, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_mul  [%s]", row.label);
        run(label, run_f64_binop_rmode("f64_mul", sf64_mul_r, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_div  [%s]", row.label);
        run(label, run_f64_binop_rmode("f64_div", sf64_div_r, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_sqrt [%s]", row.label);
        run(label, run_f64_unop_rmode("f64_sqrt", sf64_sqrt_r, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_mulAdd [%s]", row.label);
        run(label, run_f64_mulAdd_rmode(6133248, row.mode, row.label, row));

        std::snprintf(label, sizeof(label), "f64_to_i32  [%s]", row.label);
        run(label, (run_f64_to_int_rmode<int32_t, sf64_to_i32_r>(
                       "f64_to_i32", -2147483648.0, 2147483648.0, row.mode, row.label, row)));
        std::snprintf(label, sizeof(label), "f64_to_i64  [%s]", row.label);
        run(label, (run_f64_to_int_rmode<int64_t, sf64_to_i64_r>(
                       "f64_to_i64", -9223372036854775808.0, 9223372036854775808.0, row.mode,
                       row.label, row)));
        std::snprintf(label, sizeof(label), "f64_to_ui32 [%s]", row.label);
        run(label, (run_f64_to_int_rmode<uint32_t, sf64_to_u32_r>("f64_to_ui32", 0.0, 4294967296.0,
                                                                  row.mode, row.label, row)));
        std::snprintf(label, sizeof(label), "f64_to_ui64 [%s]", row.label);
        run(label, (run_f64_to_int_rmode<uint64_t, sf64_to_u64_r>(
                       "f64_to_ui64", 0.0, 18446744073709551616.0, row.mode, row.label, row)));

        std::snprintf(label, sizeof(label), "f64_to_f32  [%s]", row.label);
        run(label, run_f64_to_f32_rmode(row.mode, row.label, row));
    }

    // ---- compare --------------------------------------------------------
    //
    // FCmpInst predicate values (see soft_f64.h): pred 0..15, matching
    // LLVM IR's FCmpInst::Predicate. We cover every predicate:
    //
    //   0  FCMP_FALSE       constant false (deterministic sample)
    //   1  FCMP_OEQ         vs f64_eq / f64_eq_signaling
    //   2  FCMP_OGT         derived from f64_le
    //   3  FCMP_OGE         derived from f64_lt
    //   4  FCMP_OLT         vs f64_lt / f64_lt_quiet
    //   5  FCMP_OLE         vs f64_le / f64_le_quiet
    //   6  FCMP_ONE         derived from f64_eq
    //   7  FCMP_ORD         derived from f64_eq (src=eq, any oracle bit)
    //   8  FCMP_UNO         derived from f64_eq
    //   9  FCMP_UEQ         derived from f64_eq
    //  10  FCMP_UGT         derived from f64_le
    //  11  FCMP_UGE         derived from f64_lt
    //  12  FCMP_ULT         derived from f64_lt
    //  13  FCMP_ULE         derived from f64_le
    //  14  FCMP_UNE         derived from f64_eq
    //  15  FCMP_TRUE        constant true  (deterministic sample)
    //
    // TestFloat ops eq / le / lt have both signaling and quiet variants;
    // the returned bool is identical — only the exception flag differs,
    // and sf64_fcmp doesn't expose flags, so we map all four onto the
    // same ordered predicate.
    run("f64_eq            (OEQ)", run_f64_cmp("f64_eq", 1 /*OEQ*/));
    run("f64_eq_signaling  (OEQ)", run_f64_cmp("f64_eq_signaling", 1 /*OEQ*/));
    run("f64_lt            (OLT)", run_f64_cmp("f64_lt", 4 /*OLT*/));
    run("f64_lt_quiet      (OLT)", run_f64_cmp("f64_lt_quiet", 4 /*OLT*/));
    run("f64_le            (OLE)", run_f64_cmp("f64_le", 5 /*OLE*/));
    run("f64_le_quiet      (OLE)", run_f64_cmp("f64_le_quiet", 5 /*OLE*/));

    // Derived predicates: reuse the f64_eq / f64_lt / f64_le generators and
    // compute the expected bool via derive_pred(). Covers OGT, OGE, ONE,
    // ORD, UNO, UEQ, UGT, UGE, ULT, ULE, UNE.
    run("fcmp_OGT (derived, src=le)",
        run_f64_cmp_derived("OGT", 2 /*OGT*/, "f64_le", CmpSource::Le));
    run("fcmp_OGE (derived, src=lt)",
        run_f64_cmp_derived("OGE", 3 /*OGE*/, "f64_lt", CmpSource::Lt));
    run("fcmp_ONE (derived, src=eq)",
        run_f64_cmp_derived("ONE", 6 /*ONE*/, "f64_eq", CmpSource::Eq));
    run("fcmp_ORD (derived, src=eq)",
        run_f64_cmp_derived("ORD", 7 /*ORD*/, "f64_eq", CmpSource::Eq));
    run("fcmp_UNO (derived, src=eq)",
        run_f64_cmp_derived("UNO", 8 /*UNO*/, "f64_eq", CmpSource::Eq));
    run("fcmp_UEQ (derived, src=eq)",
        run_f64_cmp_derived("UEQ", 9 /*UEQ*/, "f64_eq", CmpSource::Eq));
    run("fcmp_UGT (derived, src=le)",
        run_f64_cmp_derived("UGT", 10 /*UGT*/, "f64_le", CmpSource::Le));
    run("fcmp_UGE (derived, src=lt)",
        run_f64_cmp_derived("UGE", 11 /*UGE*/, "f64_lt", CmpSource::Lt));
    run("fcmp_ULT (derived, src=lt)",
        run_f64_cmp_derived("ULT", 12 /*ULT*/, "f64_lt", CmpSource::Lt));
    run("fcmp_ULE (derived, src=le)",
        run_f64_cmp_derived("ULE", 13 /*ULE*/, "f64_le", CmpSource::Le));
    run("fcmp_UNE (derived, src=eq)",
        run_f64_cmp_derived("UNE", 14 /*UNE*/, "f64_eq", CmpSource::Eq));

    // Constant predicates: no oracle, deterministic seed sweep.
    run("fcmp_FALSE (constant)", run_f64_cmp_const(0 /*FALSE*/, 0));
    run("fcmp_TRUE  (constant)", run_f64_cmp_const(15 /*TRUE*/, 1));

    // ---- convert --------------------------------------------------------
    run("i32_to_f64", run_int_to_f64<int32_t>("i32_to_f64", sf64_from_i32));
    run("i64_to_f64", run_int_to_f64<int64_t>("i64_to_f64", sf64_from_i64));
    run("ui32_to_f64", run_int_to_f64<uint32_t>("ui32_to_f64", sf64_from_u32));
    run("ui64_to_f64", run_int_to_f64<uint64_t>("ui64_to_f64", sf64_from_u64));

    //
    // The third/fourth arguments define the "representable range" [min, max+1)
    // for each destination integer. Casts outside that range saturate in
    // sf64_* but return softfloat's sentinel — excluded from the oracle.
    // (2^31, 2^63, 2^32, 2^64 are exact in double.)
    run("f64_to_i32",
        (run_f64_to_int<int32_t, sf64_to_i32>("f64_to_i32", -2147483648.0, 2147483648.0)));
    run("f64_to_i64", (run_f64_to_int<int64_t, sf64_to_i64>("f64_to_i64", -9223372036854775808.0,
                                                            9223372036854775808.0)));
    run("f64_to_ui32", (run_f64_to_int<uint32_t, sf64_to_u32>("f64_to_ui32", 0.0, 4294967296.0)));
    run("f64_to_ui64",
        (run_f64_to_int<uint64_t, sf64_to_u64>("f64_to_ui64", 0.0, 18446744073709551616.0)));

    // ---- narrow-int conversions ----------------------------------------
    //
    // TestFloat 3e has no dedicated i8/u8/i16/u16 generator; we stream the
    // i32/u32 oracle, clamp inputs to the narrow range, and cross-check
    // sf64's narrow conversion against both the TestFloat f64 result and a
    // plain C++ static_cast.
    run("i8_to_f64  (via i32, clamped)", (run_narrow_int_to_f64<int8_t, int32_t, sf64_from_i8>(
                                             "i8_to_f64", "i32_to_f64", -128, 127)));
    run("i16_to_f64 (via i32, clamped)", (run_narrow_int_to_f64<int16_t, int32_t, sf64_from_i16>(
                                             "i16_to_f64", "i32_to_f64", -32768, 32767)));
    run("u8_to_f64  (via ui32, clamped)", (run_narrow_int_to_f64<uint8_t, uint32_t, sf64_from_u8>(
                                              "u8_to_f64", "ui32_to_f64", 0u, 255u)));
    run("u16_to_f64 (via ui32, clamped)", (run_narrow_int_to_f64<uint16_t, uint32_t, sf64_from_u16>(
                                              "u16_to_f64", "ui32_to_f64", 0u, 65535u)));

    run("f64_to_i8  (via f64_to_i32, clamped)",
        (run_f64_to_narrow_int<int8_t, int32_t, sf64_to_i8, sf64_to_i32>("f64_to_i8", "f64_to_i32",
                                                                         -128.0, 128.0)));
    run("f64_to_i16 (via f64_to_i32, clamped)",
        (run_f64_to_narrow_int<int16_t, int32_t, sf64_to_i16, sf64_to_i32>(
            "f64_to_i16", "f64_to_i32", -32768.0, 32768.0)));
    run("f64_to_u8  (via f64_to_ui32, clamped)",
        (run_f64_to_narrow_int<uint8_t, uint32_t, sf64_to_u8, sf64_to_u32>(
            "f64_to_u8", "f64_to_ui32", 0.0, 256.0)));
    run("f64_to_u16 (via f64_to_ui32, clamped)",
        (run_f64_to_narrow_int<uint16_t, uint32_t, sf64_to_u16, sf64_to_u32>(
            "f64_to_u16", "f64_to_ui32", 0.0, 65536.0)));

    run("f32_to_f64", run_f32_to_f64());
    run("f64_to_f32", run_f64_to_f32());

    std::printf("test_testfloat: OK (%" PRIu64 " total vectors)\n", s.total);
    return 0;
}
