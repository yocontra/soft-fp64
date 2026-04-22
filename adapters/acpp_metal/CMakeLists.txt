# AdaptiveCpp Metal SSCP external soft-fp64 adapter.
#
# AdaptiveCpp's Metal libkernel globs *.cpp from a single directory
# (non-recursive) pointed at by -DACPP_METAL_EXTERNAL_FP64_DIR, and
# compiles them into the libkernel bitcode. There is no static-archive
# link step: every translation unit needed to resolve sf64_* must be
# present in that directory, and every #include path those TUs use must
# resolve inside it.
#
# Our core source tree has two include-path shapes that would break in a
# flat staged dir:
#   1. src/*.cpp include "internal.h" (sibling) and "soft_fp64/soft_f64.h"
#      (public header, found via -I include/).
#   2. src/sleef/*.cpp include "sleef_internal.h" (sibling), which in turn
#      includes "../../include/soft_fp64/soft_f64.h" / ".../defines.h"
#      — a relative path tied to the repo layout.
#
# This target stages a flat directory with:
#   staged/
#     soft_fp64/                (copy of include/soft_fp64/, so quoted
#                                #include "soft_fp64/foo.h" resolves
#                                relative to the .cpp's directory)
#     internal.h                (from src/)
#     sleef_common.h            (rewritten: ../../include/... -> ./...)
#     sleef_internal.h          (rewritten: ../../include/... -> ./...)
#     arithmetic.cpp, classify.cpp, compare.cpp, convert.cpp,
#     rounding.cpp, sqrt_fma.cpp,
#     sleef_exp_log.cpp, sleef_inv_hyp_pow.cpp,
#     sleef_stubs.cpp, sleef_trig.cpp,
#     acpp_metal_primitives.cpp, acpp_metal_math.cpp
#
# Consumer usage:
#   cmake -S /path/to/AdaptiveCpp -B /path/to/acpp-build \
#         -DACPP_METAL_EXTERNAL_FP64_DIR=<soft-fp64 build>/adapters/acpp_metal/staged
#   cmake --build /path/to/acpp-build --target libkernel-metal-bitcode
#
# The staging approach trades a one-time copy for zero intrusion on the
# core layout and a clean non-overlapping symbol surface. The staged
# archive itself is not a link artefact — no libsoft_fp64_acpp_metal.a
# is emitted, so install-smoke's ABI grep (sf64_*-only) is unaffected.

set(SF64_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/staged")
file(MAKE_DIRECTORY "${SF64_STAGE_DIR}")
file(MAKE_DIRECTORY "${SF64_STAGE_DIR}/soft_fp64")

set(SF64_CORE_CPP
    arithmetic.cpp
    classify.cpp
    compare.cpp
    convert.cpp
    rounding.cpp
    sqrt_fma.cpp)

set(SF64_SLEEF_CPP
    sleef_exp_log.cpp
    sleef_inv_hyp_pow.cpp
    sleef_stubs.cpp
    sleef_trig.cpp)

set(SF64_STAGED_OUTPUTS)

# Public headers: copy the full include/soft_fp64/ tree verbatim.
file(GLOB SF64_PUBLIC_HEADERS
     "${CMAKE_SOURCE_DIR}/include/soft_fp64/*.h")
foreach(hdr ${SF64_PUBLIC_HEADERS})
    get_filename_component(name "${hdr}" NAME)
    set(dst "${SF64_STAGE_DIR}/soft_fp64/${name}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${hdr}" "${dst}"
        DEPENDS "${hdr}"
        COMMENT "Staging soft_fp64/${name}")
    list(APPEND SF64_STAGED_OUTPUTS "${dst}")
endforeach()

# src/internal.h — copy verbatim. It uses "soft_fp64/defines.h" which
# will resolve via the staged soft_fp64/ subdir next to it.
add_custom_command(
    OUTPUT "${SF64_STAGE_DIR}/internal.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/internal.h"
            "${SF64_STAGE_DIR}/internal.h"
    DEPENDS "${CMAKE_SOURCE_DIR}/src/internal.h"
    COMMENT "Staging internal.h")
list(APPEND SF64_STAGED_OUTPUTS "${SF64_STAGE_DIR}/internal.h")

# src/*.cpp — copy verbatim. Their includes are "internal.h" (sibling)
# and "soft_fp64/soft_f64.h" (next to .cpp via staged soft_fp64/).
foreach(src_cpp ${SF64_CORE_CPP})
    set(dst "${SF64_STAGE_DIR}/${src_cpp}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/src/${src_cpp}"
                "${dst}"
        DEPENDS "${CMAKE_SOURCE_DIR}/src/${src_cpp}"
        COMMENT "Staging src/${src_cpp}")
    list(APPEND SF64_STAGED_OUTPUTS "${dst}")
endforeach()

# src/sleef/sleef_common.h and sleef_internal.h — rewrite the
# ../../include/soft_fp64/ include prefix to plain soft_fp64/ so they
# resolve next to the staged .cpp files. sleef_internal.h also includes
# sleef_common.h (sibling), which stays unchanged.
foreach(hdr sleef_common.h sleef_internal.h)
    set(dst "${SF64_STAGE_DIR}/${hdr}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_SOURCE_DIR}/src/sleef/${hdr}
                                 -DOUTPUT=${dst}
                                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rewrite_sleef_include.cmake
        DEPENDS "${CMAKE_SOURCE_DIR}/src/sleef/${hdr}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rewrite_sleef_include.cmake"
        COMMENT "Staging sleef/${hdr} with include-path rewrite")
    list(APPEND SF64_STAGED_OUTPUTS "${dst}")
endforeach()

# src/sleef/*.cpp — copy verbatim; they only include sleef_internal.h
# (sibling in the staged dir).
foreach(sleef_cpp ${SF64_SLEEF_CPP})
    set(dst "${SF64_STAGE_DIR}/${sleef_cpp}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/src/sleef/${sleef_cpp}"
                "${dst}"
        DEPENDS "${CMAKE_SOURCE_DIR}/src/sleef/${sleef_cpp}"
        COMMENT "Staging sleef/${sleef_cpp}")
    list(APPEND SF64_STAGED_OUTPUTS "${dst}")
endforeach()

# The two forwarder TUs themselves.
foreach(fwd acpp_metal_primitives.cpp acpp_metal_math.cpp)
    set(dst "${SF64_STAGE_DIR}/${fwd}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/${fwd}"
                "${dst}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${fwd}"
        COMMENT "Staging ${fwd}")
    list(APPEND SF64_STAGED_OUTPUTS "${dst}")
endforeach()

add_custom_target(soft_fp64_acpp_metal_stage ALL
    DEPENDS ${SF64_STAGED_OUTPUTS}
    COMMENT "Staged AdaptiveCpp Metal fp64 sources at ${SF64_STAGE_DIR}")

set(SOFT_FP64_ACPP_METAL_STAGED_DIR "${SF64_STAGE_DIR}" CACHE PATH
    "Absolute path to the staged fp64 source directory to pass as ACPP_METAL_EXTERNAL_FP64_DIR when configuring AdaptiveCpp." FORCE)

message(STATUS "soft-fp64 ACPP Metal adapter: stage dir = ${SF64_STAGE_DIR}")

# ----------------------------------------------------------------------------
# Link-smoke target: compiles the two forwarder TUs against the main
# soft_fp64 library and produces a non-installed static archive. Its
# only job is to prove that every sf64_* entry point the forwarders
# reference actually resolves, and that every advertised
# __acpp_sscp_*_f64 / __acpp_sscp_soft_f64_* symbol is defined.
#
# The archive is deliberately NOT exported, NOT installed, and does NOT
# feed the install-smoke ABI grep — its forwarder symbols would (by
# design) violate the sf64_*-only rule for the installed archive.
# ----------------------------------------------------------------------------
add_library(soft_fp64_acpp_metal_forwarders STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/acpp_metal_primitives.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/acpp_metal_math.cpp")
target_link_libraries(soft_fp64_acpp_metal_forwarders
    PRIVATE soft_fp64::soft_fp64)
target_compile_options(soft_fp64_acpp_metal_forwarders PRIVATE
    -ffp-contract=off -fno-math-errno)
set_target_properties(soft_fp64_acpp_metal_forwarders PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# ctest-visible link smoke. Runs nm against the freshly built archive
# and asserts a representative set of the advertised symbol surface is
# present. Failure here means either the forwarders failed to compile,
# or a symbol listed in adapters/acpp_metal/README.md's coverage table
# silently dropped off the surface.
find_program(NM_EXECUTABLE NAMES nm llvm-nm)
if(NM_EXECUTABLE AND SOFT_FP64_BUILD_TESTS)
    add_test(NAME test_acpp_metal_adapter_link
        COMMAND ${CMAKE_COMMAND}
            -DNM=${NM_EXECUTABLE}
            -DARCHIVE=$<TARGET_FILE:soft_fp64_acpp_metal_forwarders>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_forwarder_symbols.cmake)
endif()
