# Smoke test — references every ABI symbol group.
add_executable(test_scaffold test_scaffold.cpp)
target_link_libraries(test_scaffold PRIVATE soft_fp64::soft_fp64)
target_compile_options(test_scaffold PRIVATE -ffp-contract=off -fno-math-errno)
add_test(NAME test_scaffold COMMAND test_scaffold)

# Arithmetic / compare / convert / sqrt / rounding / classify unit tests.
set(SF64_CORE_TESTS
    test_arithmetic_exact
    test_compare_all_predicates
    test_convert_subnormal
    test_convert_widths
    test_sqrt_fma_exact
    test_rounding_edges
    test_classify
    test_hypot_scaling
    test_reentrancy
)

find_package(Threads REQUIRED)

foreach(_test IN LISTS SF64_CORE_TESTS)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cpp")
    if(EXISTS ${_src})
        add_executable(${_test} ${_test}.cpp)
        target_link_libraries(${_test} PRIVATE soft_fp64::soft_fp64)
        target_compile_options(${_test} PRIVATE -ffp-contract=off -fno-math-errno)
        if(${_test} STREQUAL "test_reentrancy")
            target_link_libraries(${_test} PRIVATE Threads::Threads)
        endif()
        add_test(NAME ${_test} COMMAND ${_test})
    else()
        message(STATUS "Core test pending: ${_test}.cpp")
    endif()
endforeach()

# Transcendental 1-ULP diff tests.
set(SF64_TRANSCENDENTAL_TESTS
    test_transcendental_1ulp
)

foreach(_test IN LISTS SF64_TRANSCENDENTAL_TESTS)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cpp")
    if(EXISTS ${_src})
        add_executable(${_test} ${_test}.cpp)
        target_link_libraries(${_test} PRIVATE soft_fp64::soft_fp64)
        target_compile_options(${_test} PRIVATE -ffp-contract=off -fno-math-errno)
        add_test(NAME ${_test} COMMAND ${_test})
    else()
        message(STATUS "Transcendental test pending: ${_test}.cpp")
    endif()
endforeach()

# Exhaustive / oracle tests. Each lives in its own subdirectory and is
# lazily added if present (keeps the top-level configure path light when
# oracle deps are missing).
foreach(_oracle_dir testfloat mpfr)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_oracle_dir}/CMakeLists.txt")
        add_subdirectory(${_oracle_dir})
    else()
        message(STATUS "Oracle ${_oracle_dir} pending (tests/${_oracle_dir}/CMakeLists.txt missing)")
    endif()
endforeach()

# Experimental (report-only) sweeps for sf64_* input regions that are NOT
# part of the v1.0 correctness claim. See tests/experimental/CMakeLists.txt.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experimental/CMakeLists.txt")
    add_subdirectory(experimental)
endif()

# MPFR-backed coverage test — gracefully skipped if MPFR is not installed.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_coverage_mpfr.cpp")
    set(_pg_mpfr_found FALSE)
    set(_pg_mpfr_include_dirs "")
    set(_pg_mpfr_libraries "")

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PG_MPFR_PC QUIET mpfr)
        if(PG_MPFR_PC_FOUND)
            set(_pg_mpfr_found TRUE)
            set(_pg_mpfr_include_dirs ${PG_MPFR_PC_INCLUDE_DIRS})
            set(_pg_mpfr_libraries ${PG_MPFR_PC_LINK_LIBRARIES})
            if(NOT _pg_mpfr_libraries)
                set(_pg_mpfr_libraries ${PG_MPFR_PC_LIBRARIES})
            endif()
            pkg_check_modules(PG_GMP_PC QUIET gmp)
            if(PG_GMP_PC_FOUND)
                list(APPEND _pg_mpfr_include_dirs ${PG_GMP_PC_INCLUDE_DIRS})
            endif()
        endif()
    endif()

    if(NOT _pg_mpfr_found)
        find_library(PG_MPFR_LIB mpfr)
        find_path(PG_MPFR_INCLUDE mpfr.h)
        if(PG_MPFR_LIB AND PG_MPFR_INCLUDE)
            set(_pg_mpfr_found TRUE)
            set(_pg_mpfr_include_dirs ${PG_MPFR_INCLUDE})
            set(_pg_mpfr_libraries ${PG_MPFR_LIB})
            find_library(PG_GMP_LIB gmp)
            find_path(PG_GMP_INCLUDE gmp.h)
            if(PG_GMP_LIB)
                list(APPEND _pg_mpfr_libraries ${PG_GMP_LIB})
            endif()
            if(PG_GMP_INCLUDE)
                list(APPEND _pg_mpfr_include_dirs ${PG_GMP_INCLUDE})
            endif()
        endif()
    endif()

    if(_pg_mpfr_found)
        add_executable(test_coverage_mpfr test_coverage_mpfr.cpp)
        target_include_directories(test_coverage_mpfr PRIVATE ${_pg_mpfr_include_dirs})
        target_link_libraries(test_coverage_mpfr
            PRIVATE
                soft_fp64::soft_fp64
                ${_pg_mpfr_libraries}
        )
        target_compile_options(test_coverage_mpfr PRIVATE -ffp-contract=off -fno-math-errno)
        add_test(NAME test_coverage_mpfr COMMAND test_coverage_mpfr)
        set_tests_properties(test_coverage_mpfr PROPERTIES TIMEOUT 60)
        message(STATUS "test_coverage_mpfr registered")
    else()
        message(STATUS "MPFR not found — skipping test_coverage_mpfr "
                       "(install with `brew install mpfr` on macOS).")
    endif()
endif()
