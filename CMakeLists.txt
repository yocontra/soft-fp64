cmake_minimum_required(VERSION 3.20)
project(soft_fp64
    VERSION 1.0.0
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
    src/sleef/sleef_exp_log.cpp
    src/sleef/sleef_trig.cpp
    src/sleef/sleef_inv_hyp_pow.cpp
    src/sleef/sleef_stubs.cpp
)

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

# AdaptiveCpp Metal SSCP external-fp64 adapter — opt-in. When enabled,
# stages a flat directory of soft-fp64 sources + forwarders ready to be
# passed as -DACPP_METAL_EXTERNAL_FP64_DIR when configuring AdaptiveCpp.
# See adapters/acpp_metal/README.md for the full integration contract.
# Does not produce a conventional archive; install-smoke's sf64_*-only
# ABI grep is unaffected.
option(SOFT_FP64_BUILD_ACPP_METAL_ADAPTER
       "Stage sources for the AdaptiveCpp Metal SSCP external-fp64 adapter" OFF)
if(SOFT_FP64_BUILD_ACPP_METAL_ADAPTER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/adapters/acpp_metal/CMakeLists.txt")
    add_subdirectory(adapters/acpp_metal)
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

    install(EXPORT soft_fp64Targets
        FILE soft_fp64Targets.cmake
        NAMESPACE soft_fp64::
        DESTINATION ${SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR})

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/soft_fp64Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/soft_fp64Config.cmake"
        INSTALL_DESTINATION "${SOFT_FP64_CMAKE_CONFIG_INSTALL_DIR}")

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
