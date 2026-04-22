# Verifies that the staged AdaptiveCpp Metal forwarder archive exposes
# the advertised __acpp_sscp_* symbol surface. Invoked by ctest via
# `cmake -P`. Expects -DNM=<path> -DARCHIVE=<path>.
#
# SPDX-License-Identifier: MIT

if(NOT NM)
    message(FATAL_ERROR "NM not provided")
endif()
if(NOT ARCHIVE)
    message(FATAL_ERROR "ARCHIVE not provided")
endif()
if(NOT EXISTS "${ARCHIVE}")
    message(FATAL_ERROR "ARCHIVE does not exist: ${ARCHIVE}")
endif()

execute_process(
    COMMAND "${NM}" --defined-only "${ARCHIVE}"
    OUTPUT_VARIABLE NM_OUT
    RESULT_VARIABLE NM_RC
    ERROR_VARIABLE NM_ERR)

if(NOT NM_RC EQUAL 0)
    message(FATAL_ERROR "nm failed on ${ARCHIVE}:\n${NM_ERR}")
endif()

# Representative subset of the forwarder surface. Covers:
#  - core arithmetic primitives (acpp_metal_primitives.cpp)
#  - IEEE min/max, conversions, signbit
#  - the optional __acpp_sscp_<name>_f64 math surface
#    (acpp_metal_math.cpp)
set(REQUIRED_SYMBOLS
    __acpp_sscp_soft_f64_add
    __acpp_sscp_soft_f64_sub
    __acpp_sscp_soft_f64_mul
    __acpp_sscp_soft_f64_div
    __acpp_sscp_soft_f64_rem
    __acpp_sscp_soft_f64_neg
    __acpp_sscp_soft_f64_fmin_precise
    __acpp_sscp_soft_f64_fmax_precise
    __acpp_sscp_soft_f64_from_f32
    __acpp_sscp_sin_f64
    __acpp_sscp_cos_f64
    __acpp_sscp_exp_f64
    __acpp_sscp_log_f64
    __acpp_sscp_sqrt_f64
    __acpp_sscp_pow_f64
    __acpp_sscp_atan_f64)

set(MISSING "")
foreach(sym ${REQUIRED_SYMBOLS})
    string(FIND "${NM_OUT}" "${sym}" pos)
    if(pos EQUAL -1)
        list(APPEND MISSING "${sym}")
    endif()
endforeach()

if(MISSING)
    message("--- nm --defined-only output (truncated):")
    string(SUBSTRING "${NM_OUT}" 0 4096 NM_HEAD)
    message("${NM_HEAD}")
    message("---")
    string(REPLACE ";" "\n  " MISSING_STR "${MISSING}")
    message(FATAL_ERROR
        "acpp_metal link smoke FAILED: ${ARCHIVE} is missing required "
        "forwarder symbols:\n  ${MISSING_STR}")
endif()

list(LENGTH REQUIRED_SYMBOLS N_REQUIRED)
message(STATUS "acpp_metal link smoke: all ${N_REQUIRED} checked forwarder "
               "symbols present in ${ARCHIVE}")
