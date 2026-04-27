#pragma once

/**
 * @file soft_f64.h
 * @brief Public `sf64_*` ABI for a pure-software IEEE-754 binary64 library.
 *
 * soft-fp64 implements `double`-precision arithmetic, comparison, conversion,
 * classification, rounding, and a full transcendental surface in pure integer
 * code. Every entry point is `extern "C"` with the vendor-neutral `sf64_`
 * prefix. Consumers whose frontends emit calls under a different name
 * (e.g. AdaptiveCpp's MSL emitter calling `__acpp_sscp_soft_f64_add`) should
 * add a thin forwarding shim in their frontend rather than rebuilding this
 * library.
 *
 * Every function below takes/returns plain `double` at the C ABI level. The
 * implementation bit-casts to `uint64_t` internally and must not rely on host
 * FPU arithmetic; the same function runs bit-identically on hosts whose FPU
 * flushes subnormals, lacks an FMA unit, or reroutes `double` through `float`.
 *
 * @section precision Precision
 *
 * - Arithmetic (add/sub/mul/div/fma/rem), convert (int <-> f64, f32 <-> f64),
 *   sqrt, rounding, classification, and sign-magnitude helpers are
 *   **bit-exact** vs. IEEE-754-2008 round-to-nearest-even.
 * - Transcendentals (SLEEF 3.6 purec-scalar port) carry an ULP bound
 *   documented per function. The numbers cited are the **worst-case ULP
 *   measured by `tests/test_transcendental_1ulp.cpp` against the system libm
 *   oracle** — not aspirational u10/u35 tier labels. See each function's
 *   docstring for the measured bound.
 *
 * @section non_goals Non-goals
 *
 * - Signalling-NaN payload preservation (we quiet sNaN on entry; INVALID is
 *   still raised through the @ref fenv surface when available).
 * - Complex-number math.
 * - `fp128` / `fp16` (separate project if ever needed).
 *
 * @section ieee IEEE-754 conformance
 *
 * Arithmetic and convert paths are strictly conformant to IEEE-754-2008 for
 * round-affected ops under all five rounding attributes. The default
 * `sf64_*` surface is round-to-nearest-ties-to-even (RNE); @ref rounding
 * describes the additive `sf64_*_r(mode, …)` surface for the other four
 * modes (RTZ / RUP / RDN / RNA).
 *
 * IEEE-754 exception flags (`INVALID`, `DIVBYZERO`, `OVERFLOW`,
 * `UNDERFLOW`, `INEXACT`) are raised through the @ref fenv surface when
 * `soft_fp64` is built with `SOFT_FP64_FENV=tls` (default on hosted
 * builds). When built with `SOFT_FP64_FENV=disabled`, all `sf64_fe_*`
 * entries become no-ops and the corresponding raise-sites are compiled
 * out for zero runtime cost.
 *
 * sNaN inputs are **quieted** on entry — the quiet bit is forced on, and
 * the signalling payload is not preserved. `INVALID` is raised on sNaN
 * entry when fenv is enabled.
 *
 * @section abi ABI stability
 *
 * v1.0 freezes the `sf64_*` symbol set and calling convention. Additive
 * changes (new symbols) are v1.x-minor. Any breaking change (signature change,
 * symbol removal, semantic change of a documented guarantee) requires a major
 * version bump.
 *
 * SPDX-License-Identifier: MIT
 */

#include "defines.h"
#include "rounding_mode.h"

#include <cstdint>

// The library is compiled with `-fvisibility=hidden`; every declaration
// in this header is part of the shipped ABI and must escape the archive.
// `#pragma GCC visibility push(default)` is equivalent to tagging every
// declaration with `SF64_EXPORT` individually; the pop at the bottom of
// this file restores the prior setting. Consumers that are not themselves
// built with `-fvisibility=hidden` see a no-op.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name Arithmetic
 * @brief Bit-exact IEEE-754 binary64 arithmetic (RNE, no flags, no traps).
 *
 * All arithmetic operations are bit-exact vs. IEEE-754-2008 round-to-nearest-
 * even. Subnormal inputs and outputs are preserved (no FTZ). NaN inputs
 * propagate as a canonical quiet NaN; sNaN is quieted on entry (payload not
 * preserved). Signed zero semantics follow IEEE-754: `add(+0, -0) = +0`,
 * `mul(x, 0)` preserves the combined sign, etc.
 * @{
 */

/** @brief IEEE-754 binary64 addition (RNE). @param a addend @param b addend
 *  @return `a + b`, bit-exact IEEE-754. Any NaN input → canonical quiet NaN.
 *  `(+inf) + (-inf) → NaN`. `(+0) + (-0) → +0`. */
double sf64_add(double a, double b);

/** @brief IEEE-754 binary64 subtraction (RNE). @param a minuend @param b subtrahend
 *  @return `a - b`, bit-exact IEEE-754. Any NaN input → canonical quiet NaN.
 *  `(+inf) - (+inf) → NaN`. `x - x → +0` (RNE). */
double sf64_sub(double a, double b);

/** @brief IEEE-754 binary64 multiplication (RNE). @param a factor @param b factor
 *  @return `a * b`, bit-exact IEEE-754. Any NaN input → canonical quiet NaN.
 *  `0 * inf → NaN`. Signed-zero rules: sign of result is XOR of operand signs. */
double sf64_mul(double a, double b);

/** @brief IEEE-754 binary64 division (RNE). @param a dividend @param b divisor
 *  @return `a / b`, bit-exact IEEE-754. `0/0 → NaN`, `inf/inf → NaN`.
 *  `x / 0 → ±inf` (sign = XOR of operand signs) for finite non-zero `x`.
 *  `x / ±inf → ±0` for finite `x`. Any NaN input → canonical quiet NaN. */
double sf64_div(double a, double b);

/** @brief Truncated-quotient remainder (`fmod` semantics).
 *  @param a dividend @param b divisor
 *  @return `a - trunc(a/b) * b`. Sign of result = sign of `a`. Exact (no
 *  rounding error) for finite inputs. `rem(±inf, y) → NaN`. `rem(x, 0) → NaN`.
 *  `rem(x, ±inf) = x` for finite `x`. `rem(±0, y) → ±0` for finite non-zero `y`.
 *  Any NaN input → canonical quiet NaN. See also @ref sf64_fmod (identical
 *  semantics; kept for naming-convention compatibility) and @ref sf64_remainder
 *  (RNE-quotient variant). */
double sf64_rem(double a, double b);

/** @brief Negation by sign-bit flip. @param a operand
 *  @return `-a`, bit-exact. Flips the sign bit of any input — NaN, inf, zero,
 *  subnormal. NaN payload is preserved (quiet bit unchanged). `neg(-0) = +0`. */
double sf64_neg(double a);

/** @} */

/**
 * @name Compare
 * @brief Ordered / unordered relational predicates and NaN-preserving min/max.
 * @{
 */

/**
 * @brief All-predicates floating-point compare (LLVM FCmpInst encoding).
 *
 * `pred` is an integer in `[0, 15]` matching LLVM IR's `FCmpInst::Predicate`
 * (stable public LLVM ABI since 3.x):
 *
 * | pred | name      | semantics                          |
 * |-----:|-----------|------------------------------------|
 * |   0  | FCMP_FALSE| always false                       |
 * |   1  | FCMP_OEQ  | ordered and equal                  |
 * |   2  | FCMP_OGT  | ordered and greater than           |
 * |   3  | FCMP_OGE  | ordered and greater or equal       |
 * |   4  | FCMP_OLT  | ordered and less than              |
 * |   5  | FCMP_OLE  | ordered and less or equal          |
 * |   6  | FCMP_ONE  | ordered and not equal              |
 * |   7  | FCMP_ORD  | ordered (neither operand is NaN)   |
 * |   8  | FCMP_UNO  | unordered (either operand is NaN)  |
 * |   9  | FCMP_UEQ  | unordered or equal                 |
 * |  10  | FCMP_UGT  | unordered or greater than          |
 * |  11  | FCMP_UGE  | unordered or greater or equal      |
 * |  12  | FCMP_ULT  | unordered or less than             |
 * |  13  | FCMP_ULE  | unordered or less or equal         |
 * |  14  | FCMP_UNE  | unordered or not equal             |
 * |  15  | FCMP_TRUE | always true                        |
 *
 * @param a left operand @param b right operand @param pred predicate in `[0,15]`
 * @return `0` (false) or `1` (true). Out-of-range `pred` returns `0`.
 *         Bit-exact IEEE-754 ordered compare (with `-0 == +0`, NaNs unordered).
 *         sNaN is quieted without raising a flag. */
int sf64_fcmp(double a, double b, int pred);

/** @brief NaN-**preserving** minimum (IEEE 754-2008 `minimum`).
 *  @param a first operand
 *  @param b second operand
 *  @return the lesser of `a`, `b`; if **either** input is NaN, returns a
 *  canonical quiet NaN (propagating NaN-ness). `-0` is treated as less than
 *  `+0` (signed-zero tie-break prefers `-0`). For plain (NaN-flushing)
 *  semantics see @ref sf64_fmin. */
double sf64_fmin_precise(double a, double b);

/** @brief NaN-**preserving** maximum (IEEE 754-2008 `maximum`).
 *  @param a first operand
 *  @param b second operand
 *  @return the greater of `a`, `b`; if **either** input is NaN, returns a
 *  canonical quiet NaN. `+0` is treated as greater than `-0`. For plain
 *  (NaN-flushing) semantics see @ref sf64_fmax. */
double sf64_fmax_precise(double a, double b);

/** @} */

/**
 * @name Convert
 * @brief Bit-exact conversions between integer, `float` (binary32), and `double` (binary64).
 *
 * All `from_*` widenings are bit-exact. Narrowing `to_*` integer conversions
 * follow C99 `(int_type)double` (truncation toward zero); out-of-range inputs
 * have implementation-defined result (the library returns the wrapped or
 * saturated bit pattern deterministically, matching TestFloat's reference).
 * `sf64_from_f32` / `sf64_to_f32` are **subnormal-preserving** on both sides —
 * they use `__builtin_bit_cast` internally so host fp32 FTZ (e.g. Apple6+ MSL
 * §6.20) does not collapse subnormal payloads.
 * @{
 */

/** @brief Widen f32 → f64, subnormal-preserving. @param x f32 input
 *  @return bit-exact f64 value with the same numeric value (widening is
 *  always exact). `NaN` → quiet NaN with payload preserved in high bits.
 *  `±inf` → `±inf`. Subnormal f32 → exact normal f64 (not FTZ). */
double sf64_from_f32(float x);
/** @brief Narrow f64 → f32 (RNE), subnormal-preserving on output.
 *  @param x f64 input
 *  @return nearest representable f32 (round-to-nearest-even). Overflow → `±inf`.
 *  Underflow to subnormal → subnormal f32 (not flushed). NaN → quiet NaN with
 *  high payload bits preserved. */
float sf64_to_f32(double x);

/** @brief Exact widening `int8_t → double`. @param x @return exact f64. */
double sf64_from_i8(int8_t x);
/** @brief Exact widening `int16_t → double`. @param x @return exact f64. */
double sf64_from_i16(int16_t x);
/** @brief Exact widening `int32_t → double`. @param x @return exact f64. */
double sf64_from_i32(int32_t x);
/** @brief `int64_t → double` (RNE). @param x @return nearest f64; exact for
 *  `|x| < 2^53`, rounds to nearest even otherwise. */
double sf64_from_i64(int64_t x);
/** @brief Exact widening `uint8_t → double`. @param x @return exact f64. */
double sf64_from_u8(uint8_t x);
/** @brief Exact widening `uint16_t → double`. @param x @return exact f64. */
double sf64_from_u16(uint16_t x);
/** @brief Exact widening `uint32_t → double`. @param x @return exact f64. */
double sf64_from_u32(uint32_t x);
/** @brief `uint64_t → double` (RNE). @param x @return nearest f64; exact for
 *  `x < 2^53`. */
double sf64_from_u64(uint64_t x);

/** @brief C99-style truncation `double → int8_t`. @param x
 *  @return `(int8_t)trunc(x)` for in-range `x`. NaN returns `0` (soft-fp64
 *  chose this over SoftFloat's `INT*_MAX` so `sf64_to_iN(NaN)` and
 *  `sf64_to_uN(NaN)` are both zero — deterministic and platform-independent).
 *  Out-of-range finite inputs wrap to the C99 truncation result, matching
 *  the TestFloat reference. */
int8_t sf64_to_i8(double x);
/** @brief C99-style truncation `double → int16_t`. See @ref sf64_to_i8. */
int16_t sf64_to_i16(double x);
/** @brief C99-style truncation `double → int32_t`. See @ref sf64_to_i8. */
int32_t sf64_to_i32(double x);
/** @brief C99-style truncation `double → int64_t`. See @ref sf64_to_i8.
 *  Exact for `|trunc(x)| ≤ 2^63 - 1`. */
int64_t sf64_to_i64(double x);
/** @brief C99-style truncation `double → uint8_t`. See @ref sf64_to_i8. */
uint8_t sf64_to_u8(double x);
/** @brief C99-style truncation `double → uint16_t`. See @ref sf64_to_i8. */
uint16_t sf64_to_u16(double x);
/** @brief C99-style truncation `double → uint32_t`. See @ref sf64_to_i8. */
uint32_t sf64_to_u32(double x);
/** @brief C99-style truncation `double → uint64_t`. See @ref sf64_to_i8.
 *  Exact for `trunc(x) ≤ 2^64 - 1`. */
uint64_t sf64_to_u64(double x);

/** @} */

/**
 * @name Sqrt / FMA
 * @brief Bit-exact IEEE-754 square root, reciprocal square root, and fused-multiply-add.
 * @{
 */

/** @brief IEEE-754 square root (RNE). @param x
 *  @return bit-exact `sqrt(x)`. `sqrt(-0) = -0`. `sqrt(+inf) = +inf`.
 *  `sqrt(x) = NaN` for `x < 0` (incl. `-inf`). NaN → canonical quiet NaN. */
double sf64_sqrt(double x);

/** @brief Reciprocal square root `1/sqrt(x)` (correctly-rounded RNE).
 *  @param x input
 *  @return bit-exact `1/sqrt(x)`. `rsqrt(+0) = +inf`,
 *  `rsqrt(-0) = -inf`, `rsqrt(+inf) = +0`, `rsqrt(x<0) = NaN`. */
double sf64_rsqrt(double x);

/** @brief Fused multiply-add `a*b + c` with a single rounding step (RNE).
 *  @param a first multiplicand
 *  @param b second multiplicand
 *  @param c addend
 *  @return bit-exact IEEE-754 `fma(a,b,c)` — i.e. `a*b + c` computed at
 *  infinite precision then rounded once to nearest even. `fma(0, inf, c)` and
 *  `fma(inf, 0, c)` → NaN. Any NaN input → canonical quiet NaN. */
double sf64_fma(double a, double b, double c);

/** @} */

/**
 * @name Rounding & exponent extraction
 * @brief Integer-valued rounding modes, fractional-part extraction, binary-exponent access.
 * @{
 */

/** @brief Round toward −∞. @param x @return largest integer ≤ `x`.
 *  `floor(±0) = ±0`. `floor(±inf) = ±inf`. NaN → canonical quiet NaN. Bit-exact. */
double sf64_floor(double x);

/** @brief Round toward +∞. @param x @return smallest integer ≥ `x`.
 *  `ceil(±0) = ±0`. `ceil(±inf) = ±inf`. NaN → canonical quiet NaN. Bit-exact. */
double sf64_ceil(double x);

/** @brief Round toward zero (truncate fractional part). @param x
 *  @return integer part of `x`, sign preserved. `trunc(±0) = ±0`.
 *  `trunc(±inf) = ±inf`. NaN → canonical quiet NaN. Bit-exact. */
double sf64_trunc(double x);

/** @brief Round-half-away-from-zero. @param x @return `x` rounded to nearest
 *  integer, ties go away from zero (C99 `round` semantics, **not** RNE).
 *  `round(±0.5) = ±1`. `round(±0) = ±0`. `round(±inf) = ±inf`. Bit-exact. */
double sf64_round(double x);

/** @brief Round-half-to-even (banker's rounding, IEEE-754 `roundToIntegralTiesToEven`).
 *  @param x @return `x` rounded to nearest integer with half-even tie-break.
 *  `rint(0.5) = 0`, `rint(1.5) = 2`, `rint(2.5) = 2`. `rint(±inf) = ±inf`.
 *  NaN → canonical quiet NaN. Bit-exact. */
double sf64_rint(double x);

/** @brief Fractional part `x - floor(x)`, clamped to `[0, 1)` (GLSL `fract`).
 *  @param x @return `x - floor(x)`. Always `+0` when the mathematical result
 *  is zero (no `-0`). `fract(±inf) = NaN`. NaN → canonical quiet NaN.
 *  Exact for finite inputs (the subtraction is IEEE-exact by Sterbenz). */
double sf64_fract(double x);

/** @brief Split `x` into integer and fractional parts with a shared sign.
 *  @param x input @param iptr out: integer part (sign-preserving; may be `-0`)
 *  @return fractional part `x - *iptr`; sign matches `x` (so
 *  `modf(-1.5) → (-0.5, -1.0)`). `modf(±inf, *) → *iptr = ±inf`, returns `±0`.
 *  `modf(NaN, *) → *iptr = NaN`, returns NaN. `iptr` may be non-null; a null
 *  pointer is not tolerated (UB). Bit-exact. */
double sf64_modf(double x, double* iptr);

/** @brief Scale by a power of two: `x * 2^n`.
 *  @param x input
 *  @param n exponent
 *  @return `x * 2^n` computed by direct exponent manipulation (no arithmetic).
 *  Overflow → `±inf`; underflow → correctly-rounded subnormal or `±0`. `n` is
 *  clamped to `[-2100, 2100]` internally (outside this range the result is
 *  already inf/0 and no precision is lost). `ldexp(±0, n) = ±0`,
 *  `ldexp(±inf, n) = ±inf`, `ldexp(NaN, n) = NaN`. Bit-exact. */
double sf64_ldexp(double x, int n);

/** @brief Decompose `x` into mantissa in `[0.5, 1)` and integer exponent.
 *  @param x input @param exp out: integer exponent (or `0` for non-finite)
 *  @return mantissa with `|mantissa| ∈ [0.5, 1)` such that `x = mantissa * 2^*exp`.
 *  `frexp(±0) → (±0, *exp=0)`. `frexp(±inf) → (±inf, *exp=0)`.
 *  `frexp(NaN) → (NaN, *exp=0)`. `exp` may be non-null; a null pointer is not
 *  tolerated (UB). Subnormal inputs are renormalized (exponent reflects
 *  true binary log). Bit-exact. */
double sf64_frexp(double x, int* exp);

/** @brief Integer binary exponent (C99 `ilogb`). @param x
 *  @return `floor(log2(|x|))` as `int`. `ilogb(0) = INT_MIN` (`FP_ILOGB0`).
 *  `ilogb(±inf) = INT_MAX`. `ilogb(NaN) = INT_MAX` (`FP_ILOGBNAN`).
 *  Subnormal inputs report their true unbiased exponent (e.g. `denorm_min` → `-1074`). */
int sf64_ilogb(double x);

/** @brief Floating-point binary exponent (C99 `logb`). @param x
 *  @return same unbiased exponent as @ref sf64_ilogb but as `double`.
 *  `logb(±0) = -inf`. `logb(±inf) = +inf`. `logb(NaN) = NaN` (quieted).
 *  No error flags raised. */
double sf64_logb(double x);

/** @} */

/**
 * @name Classify & sign-magnitude
 * @brief IEEE-754 classification predicates and pure-bit sign/magnitude helpers.
 * @{
 */

/** @brief Classify: is `x` any NaN (quiet or signalling)?
 *  @param x @return `1` if NaN, else `0`. Pure bit op — no FP arithmetic. */
int sf64_isnan(double x);
/** @brief Classify: is `x` `±inf`? @param x @return `1` if infinite, else `0`. */
int sf64_isinf(double x);
/** @brief Classify: is `x` finite (not NaN, not inf)?
 *  @param x @return `1` if finite, else `0`. Subnormals count as finite. */
int sf64_isfinite(double x);
/** @brief Classify: is `x` normal (finite, non-zero, non-subnormal)?
 *  @param x @return `1` if normal, else `0`. `±0` and subnormals return `0`. */
int sf64_isnormal(double x);
/** @brief Classify: sign bit of `x`. @param x
 *  @return `1` if sign bit set (including `-0`, `-inf`, negative NaN), else `0`. */
int sf64_signbit(double x);

/** @brief Absolute value by sign-bit clear. @param x
 *  @return `|x|`, bit-exact. `fabs(-0) = +0`. `fabs(-NaN)` clears sign bit but
 *  preserves NaN payload (and quiet bit). */
double sf64_fabs(double x);

/** @brief Copy sign from `y` onto magnitude of `x` (C99 `copysign`).
 *  @param x magnitude source @param y sign source
 *  @return `|x|` with sign bit of `y`. Bit-exact. Works for all inputs
 *  including NaN (payload of `x` preserved). */
double sf64_copysign(double x, double y);

/** @brief **NaN-flushing** minimum (C99 `fmin`).
 *  @param a first operand
 *  @param b second operand
 *  @return If exactly one operand is NaN, returns the other (non-NaN) operand.
 *  If both are NaN, returns a canonical quiet NaN. Otherwise the lesser of
 *  `a`, `b`, with signed-zero tie-break preferring `-0`. Contrast with
 *  @ref sf64_fmin_precise (IEEE 754-2008 NaN-propagating `minimum`). */
double sf64_fmin(double a, double b);

/** @brief **NaN-flushing** maximum (C99 `fmax`).
 *  @param a first operand
 *  @param b second operand
 *  @return If exactly one operand is NaN, returns the other. Both NaN → NaN.
 *  Otherwise the greater of `a`, `b`; `+0` preferred over `-0` on tie.
 *  Contrast with @ref sf64_fmax_precise (IEEE 754-2008 NaN-propagating `maximum`). */
double sf64_fmax(double a, double b);

/** @brief Positive difference `max(a - b, +0)` (C99 `fdim`).
 *  @param a first operand
 *  @param b second operand
 *  @return `a - b` if `a > b`, else `+0`. Any NaN input → canonical quiet NaN. */
double sf64_fdim(double a, double b);

/** @brief The operand with larger magnitude (C99 / IEEE-754 `maxmag`).
 *  @param a first operand
 *  @param b second operand
 *  @return `a` if `|a| > |b|`, `b` if `|b| > |a|`, otherwise falls back to
 *  @ref sf64_fmax (signed-zero aware, NaN-flushing). */
double sf64_maxmag(double a, double b);

/** @brief The operand with smaller magnitude (C99 / IEEE-754 `minmag`).
 *  @param a first operand
 *  @param b second operand
 *  @return `a` if `|a| < |b|`, `b` if `|b| < |a|`, otherwise falls back to
 *  @ref sf64_fmin (signed-zero aware, NaN-flushing). */
double sf64_minmag(double a, double b);

/** @brief The representable value adjacent to `x` in the direction of `y`
 *  (C99 `nextafter`).
 *  @param x start
 *  @param y direction target
 *  @return neighbor of `x` toward `y`. `x == y` (treating `±0` equal) → returns
 *  `y` (preserving sign of `y`). `nextafter(±0, non-zero y)` → `±denorm_min`
 *  with sign matching direction of `y`. Any NaN input → canonical quiet NaN.
 *  Crosses zero, overflows to `±inf`, and underflows to subnormal/zero as
 *  required by IEEE-754. Bit-exact. */
double sf64_nextafter(double x, double y);

/** @brief Euclidean norm `sqrt(a² + b²)` using a scaled formula to avoid
 *  spurious overflow/underflow (C99 `hypot`).
 *  @param a first operand
 *  @param b second operand
 *  @return `sqrt(a² + b²)` correctly rounded to the extent of the underlying
 *  `sqrt` (≤1 ULP). **`±inf` beats NaN**: if either operand is infinite, the
 *  result is `+inf` even if the other is NaN (C99 F.10.4.3). Otherwise any
 *  NaN input → canonical quiet NaN. `hypot(±0, ±0) = +0`. Returns `±inf` only
 *  on true mathematical overflow, never from intermediate `a²`. */
double sf64_hypot(double a, double b);

/** @} */

/**
 * @name Transcendentals (SLEEF 3.6 purec-scalar port)
 * @brief Integer-only polynomial evaluation — no host FPU, no `<cmath>`.
 *
 * Every function below is computed with zero host FPU arithmetic; every `+`,
 * `-`, `*`, `/`, `fma`, `sqrt`, `floor`, `ldexp` on a `double` value is a call
 * to the corresponding `sf64_*` primitive. Polynomial constants are static
 * `constexpr` tables (no runtime-computed coefficients).
 *
 * @note ULP bounds cited per function are measured worst-case against a
 * 200-bit MPFR oracle in `tests/test_transcendental_1ulp.cpp` plus
 * `tests/mpfr/test_mpfr_diff.cpp`.
 * @{
 */

/**
 * @name Trigonometric
 * @{
 */

/** @brief Sine. @param x radians @return `sin(x)`.
 *  @details Measured worst-case **≤4 ULP** vs. host libm on `[1e-6, 100]`.
 *  Cody-Waite argument reduction for `|x| < 1e14`; Payne-Hanek (via SLEEF's
 *  `rempitabdp`) for `|x| ≥ 1e14`. `sin(±0) = ±0`. `sin(±inf) = NaN`.
 *  NaN → NaN. No IEEE exception flags. */
double sf64_sin(double x);

/** @brief Cosine. @param x radians @return `cos(x)`.
 *  @details Measured worst-case **≤4 ULP** vs. host libm on `[1e-6, 100]`.
 *  `cos(±0) = 1`. `cos(±inf) = NaN`. NaN → NaN. */
double sf64_cos(double x);

/** @brief Tangent. @param x radians @return `tan(x)`.
 *  @details Measured worst-case **≤8 ULP** vs. host libm on `[1e-6, 1.5]`.
 *  `tan(±0) = ±0`. `tan(±inf) = NaN`. NaN → NaN. At odd multiples of π/2
 *  the mathematical value is unbounded; result approaches `±inf` as precision
 *  allows (no trap). */
double sf64_tan(double x);

/** @brief Simultaneous sine and cosine. @param x radians
 *  @param s out: `sin(x)` @param c out: `cos(x)`
 *  @details Same precision guarantees as @ref sf64_sin / @ref sf64_cos
 *  evaluated separately (shared argument reduction). Null pointers are UB. */
void sf64_sincos(double x, double* s, double* c);

/** @brief Inverse sine. @param x @return `asin(x)` in radians, in `[-π/2, π/2]`.
 *  @details Measured worst-case **≤4 ULP** vs. host libm on `[1e-6, 0.99]`.
 *  Domain: `|x| ≤ 1`. `asin(x) = NaN` for `|x| > 1`. `asin(±0) = ±0`. NaN → NaN. */
double sf64_asin(double x);

/** @brief Inverse cosine. @param x @return `acos(x)` in radians, in `[0, π]`.
 *  @details Measured worst-case **≤4 ULP** vs. host libm on `[1e-6, 0.99]`.
 *  Domain: `|x| ≤ 1`. `acos(x) = NaN` for `|x| > 1`. `acos(1) = +0`. NaN → NaN. */
double sf64_acos(double x);

/** @brief Inverse tangent. @param x @return `atan(x)` in radians, in `(-π/2, π/2)`.
 *  @details Measured worst-case **≤4 ULP** vs. host libm on `[1e-6, 1e6]`.
 *  `atan(±0) = ±0`. `atan(±inf) = ±π/2`. NaN → NaN. */
double sf64_atan(double x);

/** @brief Two-argument inverse tangent (quadrant-correct).
 *  @param y ordinate @param x abscissa @return `atan2(y, x)` in radians, in `[-π, π]`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) vs. host libm.
 *  Full IEEE special-case table implemented: `atan2(±0, +0) = ±0`,
 *  `atan2(±0, -0) = ±π`, `atan2(±y, ±inf)`, `atan2(±inf, ±inf)` = quadrant
 *  multiples of π/4 with correct sign. NaN in either argument → NaN. */
double sf64_atan2(double y, double x);

/** @brief π-scaled sine: `sin(π·x)`. @param x @return `sin(π x)`.
 *  @details Measured worst-case **≤4 ULP**. `sinpi(0) = +0`, `sinpi(integer) = ±0`.
 *  `sinpi(±inf) = NaN`. NaN → NaN. */
double sf64_sinpi(double x);

/** @brief π-scaled cosine: `cos(π·x)`. @param x @return `cos(π x)`.
 *  @details Measured worst-case **≤4 ULP**. `cospi(integer) = ±1`.
 *  `cospi(±inf) = NaN`. NaN → NaN. */
double sf64_cospi(double x);

/** @brief π-scaled tangent: `tan(π·x)`. @param x @return `tan(π x)`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier). Singularities at
 *  half-integer `x` approach `±inf`. `tanpi(±inf) = NaN`. NaN → NaN. */
double sf64_tanpi(double x);

/** @brief `asin(x) / π`. @param x @return `asin(x)/π` in `[-0.5, 0.5]`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier) vs. `asin(x)/π` from
 *  host libm on `[1e-6, 0.99]`. Domain `|x| ≤ 1`; outside → NaN. NaN → NaN. */
double sf64_asinpi(double x);

/** @brief `acos(x) / π`. @param x @return `acos(x)/π` in `[0, 1]`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier). Domain `|x| ≤ 1`;
 *  outside → NaN. NaN → NaN. */
double sf64_acospi(double x);

/** @brief `atan(x) / π`. @param x @return `atan(x)/π` in `(-0.5, 0.5)`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier). NaN → NaN. */
double sf64_atanpi(double x);

/** @brief `atan2(y, x) / π`.
 *  @param y numerator
 *  @param x denominator
 *  @return `atan2(y,x)/π` in `[-1, 1]`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier). Same special-case
 *  table as @ref sf64_atan2, scaled by `1/π`. NaN → NaN. */
double sf64_atan2pi(double y, double x);

/** @} */

/**
 * @name Hyperbolic
 * @{
 */

/** @brief Hyperbolic sine. @param x @return `sinh(x)`.
 *  @details Worst-case **≤8 ULP** (u35 tier) on `|x| ∈ [1e-4, 20]`
 *  (symmetric sweep vs MPFR). `sinh(±0) = ±0`, `sinh(±inf) = ±inf`.
 *  Overflow → `±inf`. NaN → NaN. */
double sf64_sinh(double x);

/** @brief Hyperbolic cosine. @param x @return `cosh(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-4, 20]`.
 *  `cosh(±0) = 1`, `cosh(±inf) = +inf`. Overflow → `+inf`. NaN → NaN. */
double sf64_cosh(double x);

/** @brief Hyperbolic tangent. @param x @return `tanh(x)` in `(-1, 1)`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier) on `[1e-4, 20]`.
 *  `tanh(±0) = ±0`, `tanh(±inf) = ±1`. NaN → NaN. */
double sf64_tanh(double x);

/** @brief Inverse hyperbolic sine. @param x @return `asinh(x)`.
 *  @details Measured worst-case **≤8 ULP** (u35 tier) on `[1e-4, 1e6]`.
 *  `asinh(±0) = ±0`, `asinh(±inf) = ±inf`. NaN → NaN. No domain restriction. */
double sf64_asinh(double x);

/** @brief Inverse hyperbolic cosine. @param x @return `acosh(x)` in `[0, ∞)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1.01, 1e6]`.
 *  Domain: `x ≥ 1`. `acosh(x) = NaN` for `x < 1`. `acosh(1) = +0`.
 *  `acosh(+inf) = +inf`. NaN → NaN. */
double sf64_acosh(double x);

/** @brief Inverse hyperbolic tangent. @param x @return `atanh(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-4, 0.99]`.
 *  Domain: `|x| ≤ 1`. `atanh(±1) = ±inf`. `atanh(x) = NaN` for `|x| > 1`.
 *  `atanh(±0) = ±0`. NaN → NaN. */
double sf64_atanh(double x);

/** @} */

/**
 * @name Exponential / Logarithm
 * @{
 */

/** @brief Natural exponential `e^x`. @param x @return `exp(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-6, 700]`.
 *  `exp(±0) = 1`, `exp(+inf) = +inf`, `exp(-inf) = +0`. Overflow (`x` large)
 *  → `+inf`. Underflow → subnormal or `+0`. NaN → NaN. */
double sf64_exp(double x);

/** @brief Base-2 exponential `2^x`. @param x @return `exp2(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-6, 1000]`.
 *  Same boundary behavior as @ref sf64_exp. */
double sf64_exp2(double x);

/** @brief Base-10 exponential `10^x`. @param x @return `exp10(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on the same sweep band.
 *  Same boundary behavior as @ref sf64_exp. */
double sf64_exp10(double x);

/** @brief `e^x - 1`, accurate near zero (C99 `expm1`). @param x @return `exp(x) - 1`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-3, 700]`.
 *  `expm1(±0) = ±0` (sign-preserving). `expm1(+inf) = +inf`.
 *  `expm1(-inf) = -1`. NaN → NaN. */
double sf64_expm1(double x);

/** @brief Natural logarithm. @param x @return `ln(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-100, 1e100]`.
 *  Domain: `x > 0`. `log(+0) = -inf`, `log(-0) = -inf`, `log(x) = NaN` for `x < 0`.
 *  `log(+inf) = +inf`. `log(1) = +0`. NaN → NaN. */
double sf64_log(double x);

/** @brief Base-2 logarithm. @param x @return `log2(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier). Same domain as @ref sf64_log. */
double sf64_log2(double x);

/** @brief Base-10 logarithm. @param x @return `log10(x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier). Same domain as @ref sf64_log. */
double sf64_log10(double x);

/** @brief `ln(1 + x)`, accurate near zero (C99 `log1p`). @param x @return `log(1+x)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-10, 1e10]`.
 *  Domain: `x ≥ -1`. `log1p(-1) = -inf`, `log1p(x) = NaN` for `x < -1`.
 *  `log1p(±0) = ±0`. `log1p(+inf) = +inf`. NaN → NaN. */
double sf64_log1p(double x);

/** @} */

/**
 * @name Power / Root
 * @{
 */

/** @brief General power `x^y` (IEEE `pow`). @param x base @param y exponent
 *  @details Classified under the **u35 tier** (≤8 ULP) — gated uniformly
 *  at ≤8 ULP by `tests/mpfr/test_mpfr_diff.cpp` across three overlapping
 *  bounded windows covering the validated domain:
 *    - `x ∈ [1e-6, 1e6],    |y| ≤ 50`     (moderate)
 *    - `x ∈ [1e-100, 1e100], |y| ≤ 5`     (x wide, y modest)
 *    - `x ∈ [1e-6, 1e3],    |y| ≤ 100`    (x modest, y wide)
 *  Outside these windows — specifically the "near-unit base × huge
 *  exponent" corner (`x ∈ [0.5, 2], |y| ≳ 200`) — ULP drifts to ~40 because
 *  `logk_dd` evaluates its tail polynomial on `x².hi` as a plain double,
 *  which caps the log DD at ~2^-56 relative and magnifies through
 *  `y · log(x)`. A full DD-Horner rewrite of the log minimax is pencilled
 *  in for v1.2 (see TODO.md). Consumers needing ≤1 ULP on that corner
 *  should compose from @ref sf64_log and @ref sf64_exp directly with
 *  their own DD arithmetic.
 *  Full IEEE special-case table:
 *    - `pow(x, ±0) = 1` for any `x` (including NaN).
 *    - `pow(±1, y) = 1` (including `y = NaN`).
 *    - `pow(x, y) = NaN` for `x < 0` and non-integer finite `y`.
 *    - `pow(±0, y<0)` = `±inf` (odd integer `y`) or `+inf` (else).
 *    - `pow(±0, y>0)` = `±0` (odd integer `y`) or `+0` (else).
 *    - `pow(±inf, y)` and `pow(x, ±inf)` per IEEE 754-2008 §9.2.1.
 *  NaN input (other than the `x=±1` or `y=±0` exceptions above) → NaN. */
double sf64_pow(double x, double y);

/** @brief Positive-base power (IEEE 754-2019 §9.2.1 `powr`).
 *  @param x base (must be `≥ 0`; negative base → qNaN + INVALID)
 *  @param y exponent
 *  @return `x^y` with strict §9.2.1 domain semantics.
 *  @details Stricter than @ref sf64_pow — every degenerate case returns
 *  qNaN, not 1. Exceptional cases per IEEE 754-2019 §9.2.1:
 *    - `powr(NaN, y)` = `powr(x, NaN)` = qNaN (quiet propagation).
 *    - `powr(x<0, y)` = qNaN + INVALID.
 *    - `powr(±0, ±0)` = qNaN + INVALID.
 *    - `powr(+inf, ±0)` = qNaN + INVALID.
 *    - `powr(1, ±inf)` = qNaN + INVALID.
 *    - `powr(±0, y<0)` = `powr(±0, -inf)` = +inf + DIVBYZERO.
 *    - `powr(±0, y>0)` = +0.
 *    - `powr(+inf, y>0)` = +inf, `powr(+inf, y<0)` = +0.
 *    - `powr(x>1, +inf)` = +inf, `powr(x>1, -inf)` = +0.
 *    - `powr(0<x<1, +inf)` = +0, `powr(0<x<1, -inf)` = +inf.
 *  `-0` is treated as a zero (not as "< 0"). Precision on the
 *  non-degenerate interior matches @ref sf64_pow (U35 ≤ 8 ULP).
 *  Boundary conformance is gated bit-exact by
 *  `tests/test_powr_ieee754.cpp`. */
double sf64_powr(double x, double y);

/** @brief Integer-exponent power (IEEE `pown`).
 *  @param x base
 *  @param n integer exponent
 *  @return `x^n`.
 *  @details Measured worst-case **≤8 ULP** (piggybacks on @ref sf64_pow internally).
 *  `pown(x, 0) = 1` for any finite `x` (including `±0`). `pown(±inf, n)` per IEEE.
 *  Since `n` is integer, the sign of `0^n` for negative `n` is unambiguous
 *  (`pown(-0, -1) = -inf`). */
double sf64_pown(double x, int n);

/** @brief Integer `n`-th root (IEEE `rootn`). @param x base @param n integer root
 *  @return `x^(1/n)`.
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[0.01, 1e10]` with
 *  `n ∈ {2,3,4,5,7,11}`. `rootn(x, 0) = NaN`. `rootn(x<0, n)` = real-valued
 *  result for odd `n`, NaN for even `n`. `rootn(±0, n)` per IEEE. */
double sf64_rootn(double x, int n);

/** @brief Real cube root. @param x @return `cbrt(x)` (sign-preserving).
 *  @details Measured worst-case **≤4 ULP** (u10 tier) on `[1e-300, 1e300]`,
 *  including subnormals. `cbrt(-27) = -3`. `cbrt(±0) = ±0`.
 *  `cbrt(±inf) = ±inf`. NaN → NaN. Never returns NaN for non-NaN real input. */
double sf64_cbrt(double x);

/** @} */

/**
 * @name Special / misc
 * @{
 */

/** @brief Error function `erf(x) = 2/√π · ∫₀ˣ e^{-t²} dt`. @param x @return `erf(x)` in `[-1, 1]`.
 *  @details Measured worst-case **≤256 ULP** on `[-5, 5]` (Taylor/Chebyshev
 *  stitching). `erf(±0) = ±0`, `erf(±inf) = ±1`. NaN → NaN.
 *  @note **Experimental** — tightening to ≤4 ULP is deferred pending
 *  polynomial-table refinement. */
double sf64_erf(double x);

/** @brief Complementary error function `erfc(x) = 1 - erf(x)`. @param x @return `erfc(x)` in `[0,
 * 2]`.
 *  @details Worst-case **≤1024 ULP** on `[-5, 27]` against 200-bit MPFR.
 *  The deep-tail exp argument is carried in double-double
 *  (`erfc_cheb` → `expk_dd`), so the relative drift in the [15, 27] region
 *  is ≤ 8 ULP despite the absolute result sitting near IEEE double's
 *  underflow floor.
 *  `erfc(-inf) = 2`, `erfc(+inf) = +0`. NaN → NaN. */
double sf64_erfc(double x);

/** @brief True gamma function `Γ(x)`. @param x @return `tgamma(x)`.
 *  @details Worst-case **≤1024 ULP** on `[0.5, 170]` against 200-bit MPFR.
 *  `tgamma_pos` builds the Lanczos lg body in double-double and feeds it
 *  into `expk_dd`, which keeps the near-overflow bucket (`x ∈ [20, 170]`)
 *  inside GAMMA tier at ~0.9 k ULP worst-case.
 *  `tgamma(positive integer n) = (n-1)!` exactly where representable.
 *  `tgamma(0) = ±inf` (sign follows `±0`). `tgamma(negative integer) = NaN`.
 *  `tgamma(-inf) = NaN`, `tgamma(+inf) = +inf`. NaN → NaN. */
double sf64_tgamma(double x);

/** @brief Natural log of `|Γ(x)|`. @param x @return `log(|gamma(x)|)`.
 *  @details Worst-case **≤1024 ULP** vs 200-bit MPFR over the full
 *  positive shippable range `(0.5, 1e4]`:
 *  * `(0.5, 3]` — zero-centered DD Taylor pivots at `x = 1` (window
 *    `|x-1| ≤ 0.25`) and `x = 2` (window `|x-2| ≤ 0.5`), 22-term series
 *    in DD Horner from DLMF §5.7.3; outside the windows the Lanczos
 *    body handles the regime.
 *  * `[3, 1e4]` — Lanczos g=7 in DD.
 *  `lgamma(1) = +0`, `lgamma(2) = +0`.
 *  `lgamma(non-positive integer) = +inf`. `lgamma(±inf) = +inf`. NaN → NaN. */
double sf64_lgamma(double x);

/** @brief Reentrant `lgamma` that also reports the sign of `Γ(x)`.
 *  @param x input
 *  @param sign out: `+1` if `Γ(x) > 0`, `-1` if `Γ(x) < 0`, `0` on NaN/poles
 *  @return `log(|gamma(x)|)`; see @ref sf64_lgamma for bound and edge cases. */
double sf64_lgamma_r(double x, int* sign);

/** @brief Truncated-quotient remainder `x - trunc(x/y) * y` (C99 `fmod`).
 *  @param x dividend @param y divisor @return `fmod(x, y)`, sign of `x`; exact.
 *  @details Exact (no rounding error) for finite inputs. `fmod(±inf, y) = NaN`,
 *  `fmod(x, 0) = NaN`, `fmod(x, ±inf) = x` (finite `x`), `fmod(±0, y) = ±0`
 *  (non-zero finite `y`). Any NaN input → NaN. Same semantics as @ref sf64_rem. */
double sf64_fmod(double x, double y);

/** @brief IEEE-754 `remainder` — quotient rounded to **nearest even**.
 *  @param x dividend @param y divisor
 *  @return `x - n·y` where `n = round-half-to-even(x/y)`. Exact; result is in
 *  `[-|y|/2, +|y|/2]` and `remainder(x, y) = 0` when `x` is an exact multiple
 *  of `y`. On ties (`|r| == |y|/2`), the even quotient is chosen.
 *  `remainder(±inf, y) = NaN`, `remainder(x, 0) = NaN`, `remainder(x, ±inf) = x`
 *  (finite `x`). Any NaN input → NaN. Contrast with @ref sf64_fmod (truncated
 *  quotient, sign of `x`). */
double sf64_remainder(double x, double y);

/** @} */

/** @} */ // Transcendentals

/**
 * @name rounding Non-RNE rounding variants (`_r` surface)
 * @brief Explicit-mode versions of every round-affected op. Default `sf64_*`
 *        entry points are RNE; the `_r` suffix takes an @ref sf64_rounding_mode
 *        and covers all five IEEE-754 rounding attributes.
 *
 * Ops whose result does not depend on the rounding attribute —
 * @ref sf64_neg, @ref sf64_fabs, @ref sf64_copysign, the compare predicates,
 * @ref sf64_ldexp, @ref sf64_frexp, classification, and `fmod`/`remainder` —
 * have no `_r` variant because their semantics are either exact or defined
 * independently of the rounding attribute.
 *
 * `floor`/`ceil`/`trunc`/`round` are likewise mode-fixed by definition and
 * have no `_r` form; @ref sf64_rint has @ref sf64_rint_r because `rint` is
 * the one user-facing rounding op whose result is mode-dependent.
 *
 * Bit-exactness guarantee: every `_r` entry point is bit-exact vs. IEEE-754
 * for the requested mode, validated against MPFR 200-bit and Berkeley
 * TestFloat 3e across all five modes.
 * @{
 */

/** @brief Addition with explicit rounding mode. @param mode see @ref sf64_rounding_mode.
 *  @param a addend @param b addend
 *  @return `a + b`, bit-exact IEEE-754 under `mode`. Matches @ref sf64_add
 *  for `SF64_RNE`. NaN / inf / zero handling is identical across modes. */
double sf64_add_r(sf64_rounding_mode mode, double a, double b);
/** @brief Subtraction with explicit rounding mode. @see sf64_add_r */
double sf64_sub_r(sf64_rounding_mode mode, double a, double b);
/** @brief Multiplication with explicit rounding mode. @see sf64_add_r */
double sf64_mul_r(sf64_rounding_mode mode, double a, double b);
/** @brief Division with explicit rounding mode. @see sf64_add_r */
double sf64_div_r(sf64_rounding_mode mode, double a, double b);
/** @brief Square root with explicit rounding mode. @see sf64_sqrt */
double sf64_sqrt_r(sf64_rounding_mode mode, double x);
/** @brief Fused multiply-add with explicit rounding mode. Single rounding step.
 *  @see sf64_fma */
double sf64_fma_r(sf64_rounding_mode mode, double a, double b, double c);

/** @brief Narrow f64 → f32 with explicit rounding mode. @see sf64_to_f32 */
float sf64_to_f32_r(sf64_rounding_mode mode, double x);

/** @brief `double → int8_t` with explicit rounding mode (mode-aware rounding
 *  of the intermediate before truncation-to-integer). `SF64_RTZ` matches
 *  @ref sf64_to_i8. Saturation/NaN behavior identical across modes. */
int8_t sf64_to_i8_r(sf64_rounding_mode mode, double x);
/** @brief `double → int16_t` with explicit rounding mode. @see sf64_to_i8_r */
int16_t sf64_to_i16_r(sf64_rounding_mode mode, double x);
/** @brief `double → int32_t` with explicit rounding mode. @see sf64_to_i8_r */
int32_t sf64_to_i32_r(sf64_rounding_mode mode, double x);
/** @brief `double → int64_t` with explicit rounding mode. @see sf64_to_i8_r */
int64_t sf64_to_i64_r(sf64_rounding_mode mode, double x);
/** @brief `double → uint8_t` with explicit rounding mode. @see sf64_to_i8_r */
uint8_t sf64_to_u8_r(sf64_rounding_mode mode, double x);
/** @brief `double → uint16_t` with explicit rounding mode. @see sf64_to_i8_r */
uint16_t sf64_to_u16_r(sf64_rounding_mode mode, double x);
/** @brief `double → uint32_t` with explicit rounding mode. @see sf64_to_i8_r */
uint32_t sf64_to_u32_r(sf64_rounding_mode mode, double x);
/** @brief `double → uint64_t` with explicit rounding mode. @see sf64_to_i8_r */
uint64_t sf64_to_u64_r(sf64_rounding_mode mode, double x);

/** @brief Round-to-integer with explicit rounding mode (IEEE-754
 *  `roundToIntegralExact`). `SF64_RNE` matches @ref sf64_rint;
 *  `SF64_RTZ` matches @ref sf64_trunc; `SF64_RUP` matches @ref sf64_ceil;
 *  `SF64_RDN` matches @ref sf64_floor; `SF64_RNA` matches @ref sf64_round. */
double sf64_rint_r(sf64_rounding_mode mode, double x);

/** @} */ // rounding

/**
 * @name fenv IEEE-754 exception flags (thread-local)
 * @brief Sticky flag accumulators matching IEEE-754 §7.
 *
 * Flag storage is per-thread when the library is built with
 * `SOFT_FP64_FENV=tls` (the default). When built with
 * `SOFT_FP64_FENV=disabled`, the raise-sites are compiled out and all
 * `sf64_fe_*` entries become zero-cost no-ops / no-op queries (getall
 * returns 0). Bit assignments match `<fenv.h>` conventions so consumers
 * can bridge to glibc fenv without a lookup table.
 * @{
 */

/** @brief Exception-flag bit positions. Bitwise-OR to combine. */
typedef enum sf64_fe_flag {
    SF64_FE_INVALID = 1u << 0,   /**< invalid operation (NaN from non-NaN, etc.) */
    SF64_FE_DIVBYZERO = 1u << 1, /**< division by zero (finite non-zero / 0)      */
    SF64_FE_OVERFLOW = 1u << 2,  /**< result too large to represent              */
    SF64_FE_UNDERFLOW = 1u << 3, /**< subnormal-or-smaller result w/ inexact      */
    SF64_FE_INEXACT = 1u << 4,   /**< result not exactly representable            */
} sf64_fe_flag;

/** @brief Read all currently-sticky flags.
 *  @return bitwise-OR of @ref sf64_fe_flag values for the calling thread.
 *  Returns 0 under `SOFT_FP64_FENV=disabled`. */
unsigned sf64_fe_getall(void);

/** @brief Test whether any flag in `mask` is currently set.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values
 *  @return `1` if `getall() & mask` is non-zero, else `0`. Returns 0 under
 *  `SOFT_FP64_FENV=disabled`. */
int sf64_fe_test(unsigned mask);

/** @brief Set the given flags sticky for the calling thread.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values. No-op under
 *  `SOFT_FP64_FENV=disabled`. */
void sf64_fe_raise(unsigned mask);

/** @brief Clear the given flags for the calling thread.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values. No-op under
 *  `SOFT_FP64_FENV=disabled`. */
void sf64_fe_clear(unsigned mask);

/** @brief Opaque flag-state snapshot for save/restore. Size and alignment
 *  are ABI-stable for the 1.x line. Under `SOFT_FP64_FENV=disabled` the
 *  structure still exists (for caller-side ABI compatibility) but
 *  save/restore are no-ops. */
typedef struct sf64_fe_state_t {
    unsigned flags;
} sf64_fe_state_t;

/** @brief Snapshot the current thread-local flag state into `out`.
 *  @param out destination; must be non-null. */
void sf64_fe_save(sf64_fe_state_t* out);

/** @brief Restore a previously-saved flag state.
 *  @param in source; must be non-null. Replaces (does not OR) the current
 *  thread-local flag state. */
void sf64_fe_restore(const sf64_fe_state_t* in);

/** @} */ // fenv

/**
 * @name fenv_ex IEEE-754 exception flags (caller-supplied state)
 * @brief Parallel ABI matching the @ref fenv surface but reading and
 *        writing a caller-provided @ref sf64_fe_state_t instead of
 *        thread-local storage.
 *
 * GPU and SIMT consumers (Apple Metal, WebGPU/WGSL, OpenCL device code)
 * can't use `thread_local`, so the TLS-backed default surface is
 * unreachable in those targets. The `_ex` surface accepts a pointer to a
 * stack-local @ref sf64_fe_state_t and the parallel arithmetic / sqrt /
 * fma / convert ABI (`sf64_*_ex`) raises into that struct directly.
 *
 * Mode availability:
 *  - `SOFT_FP64_FENV=tls` (default): both surfaces are emitted. The
 *    default surface still uses TLS; the `_ex` surface uses the caller's
 *    pointer (a null pointer drops flags).
 *  - `SOFT_FP64_FENV=explicit`: only the `_ex` surface carries state.
 *    The TLS surface compiles to no-op stubs (returns 0 / no-op writes)
 *    so consumers that reference both surfaces still link, but the TLS
 *    flag bag does not exist on the device.
 *  - `SOFT_FP64_FENV=disabled`: both surfaces compile to no-ops, every
 *    raise site in the library is DCE'd, and the `_ex` symbols are not
 *    emitted into the archive.
 *
 * Bit layout of @ref sf64_fe_state_t::flags is the same as the TLS
 * surface — reuses the `SF64_FE_*` enum values.
 * @{
 */

/** @brief Read all currently-sticky flags from the caller's state.
 *  @param state caller-supplied state; null returns 0. */
unsigned sf64_fe_getall_ex(const sf64_fe_state_t* state);

/** @brief Test whether any flag in `mask` is set in the caller's state.
 *  @param state caller-supplied state; null returns 0.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values. */
int sf64_fe_test_ex(const sf64_fe_state_t* state, unsigned mask);

/** @brief Set the given flags sticky in the caller's state.
 *  @param state caller-supplied state; null is a no-op.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values. */
void sf64_fe_raise_ex(sf64_fe_state_t* state, unsigned mask);

/** @brief Clear the given flags from the caller's state.
 *  @param state caller-supplied state; null is a no-op.
 *  @param mask bitwise-OR of @ref sf64_fe_flag values. */
void sf64_fe_clear_ex(sf64_fe_state_t* state, unsigned mask);

/** @brief Snapshot the caller's state into `out`. Both arguments may be
 *  null (no-op when `out` is null; reads zero when `state` is null). */
void sf64_fe_save_ex(const sf64_fe_state_t* state, sf64_fe_state_t* out);

/** @brief Replace the caller's state from `in`. Either argument null is a
 *  no-op (no host fp64 fallback — the call simply does nothing). */
void sf64_fe_restore_ex(sf64_fe_state_t* state, const sf64_fe_state_t* in);

/**
 * @name Caller-state arithmetic / sqrt / fma / convert (`sf64_*_ex`)
 * @brief Bit-identical to their TLS-backed counterparts; the only
 *        difference is that exception flags raised by the operation are
 *        OR'd into `*state` instead of into the thread-local sticky bag.
 *        A null `state` pointer drops flags (zero overhead).
 * @{
 */

double sf64_add_ex(double a, double b, sf64_fe_state_t* state);
double sf64_sub_ex(double a, double b, sf64_fe_state_t* state);
double sf64_mul_ex(double a, double b, sf64_fe_state_t* state);
double sf64_div_ex(double a, double b, sf64_fe_state_t* state);
double sf64_sqrt_ex(double x, sf64_fe_state_t* state);
double sf64_fma_ex(double a, double b, double c, sf64_fe_state_t* state);

/** @brief Mode-parametrised explicit-state addition. Combines the `_r`
 *  rounding-mode dispatch with the `_ex` caller-state ABI. */
double sf64_add_r_ex(sf64_rounding_mode mode, double a, double b, sf64_fe_state_t* state);
double sf64_sub_r_ex(sf64_rounding_mode mode, double a, double b, sf64_fe_state_t* state);
double sf64_mul_r_ex(sf64_rounding_mode mode, double a, double b, sf64_fe_state_t* state);
double sf64_div_r_ex(sf64_rounding_mode mode, double a, double b, sf64_fe_state_t* state);
double sf64_sqrt_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
double sf64_fma_r_ex(sf64_rounding_mode mode, double a, double b, double c, sf64_fe_state_t* state);

/** @brief Widen f32 → f64 (caller-state). @see sf64_from_f32 */
double sf64_from_f32_ex(float x, sf64_fe_state_t* state);
/** @brief Narrow f64 → f32 (RNE, caller-state). @see sf64_to_f32 */
float sf64_to_f32_ex(double x, sf64_fe_state_t* state);
/** @brief Narrow f64 → f32 with explicit rounding mode (caller-state). */
float sf64_to_f32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);

double sf64_from_i8_ex(int8_t x, sf64_fe_state_t* state);
double sf64_from_i16_ex(int16_t x, sf64_fe_state_t* state);
double sf64_from_i32_ex(int32_t x, sf64_fe_state_t* state);
double sf64_from_i64_ex(int64_t x, sf64_fe_state_t* state);
double sf64_from_u8_ex(uint8_t x, sf64_fe_state_t* state);
double sf64_from_u16_ex(uint16_t x, sf64_fe_state_t* state);
double sf64_from_u32_ex(uint32_t x, sf64_fe_state_t* state);
double sf64_from_u64_ex(uint64_t x, sf64_fe_state_t* state);

int8_t sf64_to_i8_ex(double x, sf64_fe_state_t* state);
int16_t sf64_to_i16_ex(double x, sf64_fe_state_t* state);
int32_t sf64_to_i32_ex(double x, sf64_fe_state_t* state);
int64_t sf64_to_i64_ex(double x, sf64_fe_state_t* state);
uint8_t sf64_to_u8_ex(double x, sf64_fe_state_t* state);
uint16_t sf64_to_u16_ex(double x, sf64_fe_state_t* state);
uint32_t sf64_to_u32_ex(double x, sf64_fe_state_t* state);
uint64_t sf64_to_u64_ex(double x, sf64_fe_state_t* state);

int8_t sf64_to_i8_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
int16_t sf64_to_i16_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
int32_t sf64_to_i32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
int64_t sf64_to_i64_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
uint8_t sf64_to_u8_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
uint16_t sf64_to_u16_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
uint32_t sf64_to_u32_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);
uint64_t sf64_to_u64_r_ex(sf64_rounding_mode mode, double x, sf64_fe_state_t* state);

/** @} */
/** @} */ // fenv_ex

#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC visibility pop
#endif
