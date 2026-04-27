cmake_minimum_required(VERSION 3.20)
project(soft_fp64
    VERSION 2.0.0
    DESCRIPTION "Portable bit-exact IEEE-754 binary64 soft-float library"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED PROJECT_IS_TOP_LEVEL)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(PROJECT_IS_TOP_LEVEL TRUE)
    else()
        set(PROJECT_IS_TOP_LEVEL FALSE)
    endif()
endif()

# Off-by-default for add_subdirectory / FetchContent consumers.
option(SOFT_FP64_BUILD_TESTS "Build host unit tests"      ${PROJECT_IS_TOP_LEVEL})
option(SOFT_FP64_WERROR      "Treat warnings as errors"   ${PROJECT_IS_TOP_LEVEL})

add_library(soft_fp64 STATIC)
add_library(soft_fp64::soft_fp64 ALIAS soft_fp64)

target_include_directories(soft_fp64
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Arithmetic + classify + convert + rounding + sqrt/fma, then the
# SLEEF-derived transcendentals (see src/sleef/NOTICE). Partitioned by
# family: trig / exp-log / inv+hyp+pow / long-tail stubs.
target_sources(soft_fp64 PRIVATE
    src/arithmetic.cpp
    src/compare.cpp
    src/convert.cpp
    src/sqrt_fma.cpp
    src/rounding.cpp
    src/classify.cpp
    src/fenv.cpp
    src/sleef/sleef_exp_log.cpp
    src/sleef/sleef_trig.cpp
    src/sleef/sleef_inv_hyp_pow.cpp
    src/sleef/sleef_stubs.cpp
)

# IEEE-754 exception-flag storage. `tls` uses a thread_local accumulator;
# `disabled` compiles every raise / query to a no-op (zero-cost); `explicit`
# emits the parallel `sf64_*_ex` caller-state ABI for GPU / SIMT consumers
# (Apple Metal, WebGPU, OpenCL device code) where `thread_local` is not
# available — and the legacy TLS surface stays present as no-op stubs so
# adapters that reference both can link.
set(SOFT_FP64_FENV "tls" CACHE STRING
    "Exception-flag storage: tls | disabled | explicit")
set_property(CACHE SOFT_FP64_FENV PROPERTY STRINGS tls disabled explicit)
if(SOFT_FP64_FENV STREQUAL "tls")
    target_compile_definitions(soft_fp64 PRIVATE SOFT_FP64_FENV_MODE=1)
elseif(SOFT_FP64_FENV STREQUAL "disabled")
    target_compile_definitions(soft_fp64 PRIVATE SOFT_FP64_FENV_MODE=0)
elseif(SOFT_FP64_FENV STREQUAL "explicit")
    target_compile_definitions(soft_fp64 PRIVATE SOFT_FP64_FENV_MODE=2)
else()
    message(FATAL_ERROR "SOFT_FP64_FENV must be tls | disabled | explicit")
endif()

# Soft-fp64 TUs must not have `a*b+c` fused into llvm.fma.f64 at -O3 —
# fma depends on add/mul and we'd create a link-time cycle. Disable
# contraction globally for this target.
target_compile_options(soft_fp64 PRIVATE
    -ffp-contract=off
    -fno-math-errno
    -fno-trapping-math
    -Wall
    -Wextra
    -Wpedantic)

# ABI surface: hide every symbol by default; only `SF64_EXPORT`-decorated
# entry points in include/soft_fp64/soft_f64.h escape the archive. The
# install-smoke CI job runs `nm -g` and fails if non-`sf64_*` symbols
# appear in the global table.
set_target_properties(soft_fp64 PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

if(SOFT_FP64_WERROR)
    target_compile_options(soft_fp64 PRIVATE -Werror)
endif()

if(SOFT_FP64_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# libFuzzer targets — opt-in, requires clang.
option(SOFT_FP64_BUILD_FUZZ "Build libFuzzer targets (requires clang)" OFF)
if(SOFT_FP64_BUILD_FUZZ AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/CMakeLists.txt")
    add_subdirectory(fuzz)
endif()

# Microbenchmarks — opt-in.
option(SOFT_FP64_BUILD_BENCH "Build microbenchmark harness" OFF)
if(SOFT_FP64_BUILD_BENCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt")
    add_subdirectory(bench)
endif()

# -----------------------------------------------------------------------------
# Install, export set, find_package + pkg-config integration.
#
# Consumers can use either of:
#   find_package(soft_fp64 CONFIG REQUIRED)          # CMake >= 3.20
#   pkg-config --cflags --libs soft_fp64             # plain Makefiles, etc.
#
# Install is skipped automatically when the project is pulled in as a
# subdirectory (add_subdirectory / FetchContent), so embedding soft_fp64 in
# another build tree doesn't pollute the host project's install tree.
# -----------------------------------------------------------------------------
option(SOFT_FP64_INSTALL "Generate install + export rules for soft_fp64"
       ${PROJECT_IS_TOP_LEVEL})

if(SOFT_FP64_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/soft_fp64"
        CACHE STRING "Install path for soft_fp64 CMake package files")

    install(TARGETS soft_fp64
        EXPORT soft_fp64Targets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(DIRECTORY include/soft_fp64
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

    # Ship the `sf64_*` source bodies + the SLEEF-derived transcendental
    # sources alongside the installed headers. Consumers that need to
    # compile soft-fp64 into a non-host target (the AdaptiveCpp libkernel
    # bitcode build, downstream GPU adapters) read these via the
    # `soft_fp64_SOURCE_DIR` / `soft_fp64_SLEEF_SOURCE_DIR` variables
    # exposed by `soft_fp64Config.cmake`. Hidden-visibility internal
    # headers (`internal*.h`) ride along; they are not promoted to the
    # public include set.
    install(DIRECTORY src/
        DESTINATION "${CMAKE_INSTALL_DATADIR}/soft_fp64/src"
        FILES_MATCHING
            PATTERN "*.cpp"
            PATTERN "*.h"
            PATTERN "*.hpp")

    install(EXPORT soft_fp64Targets
        FILE soft_fp64Targets.cmake
        NAMESPACE soft_fp64::
        DESTINATION ${SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR})

    # Source-tree paths surfaced to consumers that need to compile the
    # `sf64_*` bodies themselves (e.g. AdaptiveCpp's libkernel bitcode
    # build, which globs `src/*.cpp` + `src/sleef/*.cpp` directly into a
    # SYCL/SSCP target rather than linking the host static library).
    # Resolved at install time relative to the install prefix; the source
    # tree is shipped under `${CMAKE_INSTALL_DATADIR}/soft_fp64/src`.
    set(SOFT_FP64_SOURCE_RELDIR        "${CMAKE_INSTALL_DATADIR}/soft_fp64/src")
    set(SOFT_FP64_SLEEF_SOURCE_RELDIR  "${CMAKE_INSTALL_DATADIR}/soft_fp64/src/sleef")
    set(SOFT_FP64_INCLUDE_RELDIR       "${CMAKE_INSTALL_INCLUDEDIR}")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/soft_fp64Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64Config.cmake"
        INSTALL_DESTINATION "${SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR}"
        PATH_VARS SOFT_FP64_SOURCE_RELDIR SOFT_FP64_SLEEF_SOURCE_RELDIR
                  SOFT_FP64_INCLUDE_RELDIR)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64ConfigVersion.cmake"
        DESTINATION "${SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR}")

    # pkg-config descriptor for non-CMake consumers.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/soft_fp64.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64.pc"
        @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

    # License + third-party attribution files alongside the headers.
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}")
endif()
