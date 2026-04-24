#pragma once
//
// Call-site rewrite macros for the SLEEF TUs.
//
// Included immediately after `sleef_internal.h` in every SLEEF `.cpp` file
// so that `sf64_add(a, b)` / `sf64_mul(a, b)` / `sf64_fma(a, b, c)` /
// `sf64_sqrt(x)` / `ddadd2_dd_dd(a, b)` / etc. textually resolve to the
// RNE-specialized, fe-threaded inline helpers in `sleef_common.h` and
// `internal_arith.h` — **while the user-visible call syntax stays
// identical to the pre-1.1 form**.
//
// Each macro references a local identifier `fe` which must be in scope at
// the call site. Public SLEEF entry functions declare
// `soft_fp64::internal::sf64_internal_fe_acc fe;` at the top and flush
// once before returning; every helper function threaded through the call
// graph takes `soft_fp64::sleef::sf64_internal_fe_acc& fe` as an
// explicit parameter.
//
// C++ preprocessor self-reference: when the macro body contains the same
// identifier (e.g. `add_` refers to `::soft_fp64::sleef::add_`), the
// replacement is not re-expanded, which is exactly what we want — the
// namespace-qualified call resolves to the real function.
//
// Keep this header textually small — each macro is a single-line rewrite.
//
// SPDX-License-Identifier: MIT

#include "sleef_common.h"

// ---- arithmetic / sqrt / fma -------------------------------------------
// Variadic so any comma hidden inside {...} or <...> in the original
// arguments survives preprocessing intact.
#define sf64_add(...) ::soft_fp64::sleef::add_(__VA_ARGS__, fe)
#define sf64_sub(...) ::soft_fp64::sleef::sub_(__VA_ARGS__, fe)
#define sf64_mul(...) ::soft_fp64::sleef::mul_(__VA_ARGS__, fe)
#define sf64_div(...) ::soft_fp64::sleef::div_(__VA_ARGS__, fe)
#define sf64_fma(...) ::soft_fp64::sleef::fma_(__VA_ARGS__, fe)
#define sf64_sqrt(...) ::soft_fp64::sleef::sqrt_(__VA_ARGS__, fe)
// sf64_neg doesn't touch fe but is part of the rewrite surface for
// consistency; resolves to a bit flip.
#define sf64_neg(x) ::soft_fp64::internal::sf64_internal_neg((x))

// ---- DD primitives ------------------------------------------------------
// Each DD primitive listed in sleef_common.h takes `fe` as its final
// parameter. The macros below append `fe` to the caller's argument list.
// Variadic so a `DD{a, b}` literal (whose internal comma the preprocessor
// would otherwise split) passes through as a single argument. Self-
// reference on the identifier prevents re-expansion of the function name.
#define ddadd_dd_d_d(...) ddadd_dd_d_d(__VA_ARGS__, fe)
#define ddadd_dd_dd_d(...) ddadd_dd_dd_d(__VA_ARGS__, fe)
#define ddadd_dd_d_dd(...) ddadd_dd_d_dd(__VA_ARGS__, fe)
#define ddadd_dd_dd_dd(...) ddadd_dd_dd_dd(__VA_ARGS__, fe)
#define ddmul_d_dd_dd(...) ddmul_d_dd_dd(__VA_ARGS__, fe)
#define ddadd2_dd_dd(...) ddadd2_dd_dd(__VA_ARGS__, fe)
#define ddadd2_dd_d_d(...) ddadd2_dd_d_d(__VA_ARGS__, fe)
#define ddadd2_dd_dd_d(...) ddadd2_dd_dd_d(__VA_ARGS__, fe)
#define ddadd2_dd_d_dd(...) ddadd2_dd_d_dd(__VA_ARGS__, fe)
#define ddmul_dd_d_d(...) ddmul_dd_d_d(__VA_ARGS__, fe)
#define ddmul_dd_dd_d(...) ddmul_dd_dd_d(__VA_ARGS__, fe)
#define ddmul_dd_dd_dd(...) ddmul_dd_dd_dd(__VA_ARGS__, fe)
#define ddsqu_dd_dd(...) ddsqu_dd_dd(__VA_ARGS__, fe)
#define ddrec_dd_d(...) ddrec_dd_d(__VA_ARGS__, fe)
#define ddrec_dd_dd(...) ddrec_dd_dd(__VA_ARGS__, fe)
#define dddiv_dd_dd_dd(...) dddiv_dd_dd_dd(__VA_ARGS__, fe)
#define ddnormalize_dd_dd(...) ddnormalize_dd_dd(__VA_ARGS__, fe)
#define dd_to_d(...) dd_to_d(__VA_ARGS__, fe)
#define ddscale_dd_dd_d(...) ddscale_dd_dd_d(__VA_ARGS__, fe)
// ddneg_dd_dd doesn't need fe (bit flip).

// ---- polynomial evaluation ---------------------------------------------
#define poly2(...) ::soft_fp64::sleef::poly2(__VA_ARGS__, fe)
#define poly3(...) ::soft_fp64::sleef::poly3(__VA_ARGS__, fe)
#define poly4(...) ::soft_fp64::sleef::poly4(__VA_ARGS__, fe)
#define poly_array(...) ::soft_fp64::sleef::poly_array(__VA_ARGS__, fe)
#define mla(...) ::soft_fp64::sleef::mla(__VA_ARGS__, fe)
#define mlapn(...) ::soft_fp64::sleef::mlapn(__VA_ARGS__, fe)

// ---- integer predicates (sleef_internal::detail) -----------------------
// `is_int` / `is_odd_int` now take fe. Macros keep the old syntax.
#define is_int(...) ::soft_fp64::sleef::detail::is_int(__VA_ARGS__, fe)
#define is_odd_int(...) ::soft_fp64::sleef::detail::is_odd_int(__VA_ARGS__, fe)

// ---- cross-TU SLEEF cores ----------------------------------------------
// The core signatures take `fe` explicitly — the macros below are only
// for consumer TUs, not the defining TUs. If a defining TU (sleef_exp_log
// / sleef_inv_hyp_pow) wants to emit the definition without macro
// substitution, it must `#undef` the macro name locally before the
// definition. Consumer TUs just call the core as if it had its pre-1.1
// single-argument shape and the macro appends `fe`.
//
// (Left undefined here — call sites pass `fe` explicitly so the
// macro-vs-definition collision simply doesn't arise.)
