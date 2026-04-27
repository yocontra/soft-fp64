// Public sf64_fe_* surface (TLS-backed) plus the parallel sf64_fe_*_ex
// surface (caller-state-backed). Storage and call sites:
//
//   SOFT_FP64_FENV_MODE:
//     0 — disabled: every public entry below is a no-op (or returns 0).
//                   Both surfaces are still emitted so adapters that
//                   reference either ABI link cleanly; CMake's
//                   `disabled` cell uses a separate sentinel that hides
//                   them from `nm -g` only when the entire fenv ABI is
//                   intentionally absent (the install-smoke gate already
//                   permits the symbols on the safe sf64_* prefix).
//     1 — tls: thread_local sticky accumulator. The default `sf64_fe_*`
//              surface reads/writes that bag. The `sf64_fe_*_ex` surface
//              services callers that supply their own state pointer.
//     2 — explicit: thread_local storage is omitted (so consumers like
//              Metal / WebGPU GPU kernels can link). The TLS surface
//              compiles to no-op stubs; only `sf64_fe_*_ex` carries
//              flag state — and only when the caller passes a non-null
//              `sf64_fe_state_t*`.
//
// Cross-TU raise sites in arithmetic / sqrt_fma / convert / sleef thread
// either the TLS-backed or caller-state-backed accumulator (see
// src/internal_fenv.h) — both end up here at flush time.
//
// SPDX-License-Identifier: MIT

#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

namespace soft_fp64::internal {
#if SOFT_FP64_FENV_MODE == 1
// Visibility hidden must match the extern declaration in internal_fenv.h
// byte-for-byte, otherwise clang on Linux ELF may emit a `_ZTH...` TLS init
// wrapper reference at cross-TU access sites that has no matching definition
// (the constant-init `= 0u` body suppresses wrapper emission here, but the
// reference is still emitted at the access site if visibility doesn't match).
// PIE link then fails with "PC32 against undefined hidden symbol _ZTH...".
[[gnu::visibility("hidden"),
  gnu::tls_model("initial-exec")]] thread_local unsigned sf64_internal_fe_flags = 0u;
#endif
} // namespace soft_fp64::internal

// ---------------------------------------------------------------------------
// Default (TLS) surface — present in `tls` mode; under `disabled` and
// `explicit` it compiles to no-op shims so adapters that mix and match
// the two surfaces still link.
// ---------------------------------------------------------------------------

extern "C" unsigned sf64_fe_getall(void) {
#if SOFT_FP64_FENV_MODE == 1
    return soft_fp64::internal::sf64_internal_fe_flags;
#else
    return 0u;
#endif
}

extern "C" int sf64_fe_test(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    return (soft_fp64::internal::sf64_internal_fe_flags & mask) != 0 ? 1 : 0;
#else
    (void)mask;
    return 0;
#endif
}

extern "C" void sf64_fe_raise(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags |= mask;
#else
    (void)mask;
#endif
}

extern "C" void sf64_fe_clear(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags &= ~mask;
#else
    (void)mask;
#endif
}

extern "C" void sf64_fe_save(sf64_fe_state_t* out) {
    if (out == nullptr)
        return;
#if SOFT_FP64_FENV_MODE == 1
    out->flags = soft_fp64::internal::sf64_internal_fe_flags;
#else
    out->flags = 0u;
#endif
}

extern "C" void sf64_fe_restore(const sf64_fe_state_t* in) {
    if (in == nullptr)
        return;
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags = in->flags;
#else
    (void)in;
#endif
}

// ---------------------------------------------------------------------------
// Caller-state surface (`_ex`). Present under `tls` and `explicit`; the
// `disabled` build hides it (and the underlying raise sites are no-ops
// regardless).
// ---------------------------------------------------------------------------

#if SOFT_FP64_FENV_MODE == 1 || SOFT_FP64_FENV_MODE == 2

extern "C" unsigned sf64_fe_getall_ex(const sf64_fe_state_t* state) {
    return state != nullptr ? state->flags : 0u;
}

extern "C" int sf64_fe_test_ex(const sf64_fe_state_t* state, unsigned mask) {
    if (state == nullptr)
        return 0;
    return (state->flags & mask) != 0 ? 1 : 0;
}

extern "C" void sf64_fe_raise_ex(sf64_fe_state_t* state, unsigned mask) {
    if (state == nullptr)
        return;
    state->flags |= mask;
}

extern "C" void sf64_fe_clear_ex(sf64_fe_state_t* state, unsigned mask) {
    if (state == nullptr)
        return;
    state->flags &= ~mask;
}

extern "C" void sf64_fe_save_ex(const sf64_fe_state_t* state, sf64_fe_state_t* out) {
    if (out == nullptr)
        return;
    out->flags = state != nullptr ? state->flags : 0u;
}

extern "C" void sf64_fe_restore_ex(sf64_fe_state_t* state, const sf64_fe_state_t* in) {
    if (state == nullptr || in == nullptr)
        return;
    state->flags = in->flags;
}

#endif // SOFT_FP64_FENV_MODE == 1 || SOFT_FP64_FENV_MODE == 2
