cmake_minimum_required(VERSION 3.15)

project(ndzip)

option(NDZIP_BUILD_TEST "Build unit tests" OFF)
option(NDZIP_BUILD_BENCHMARK "Build benchmarks against other algorithms" OFF)
option(NDZIP_WITH_MT "Enable parallel CPU implementation through OpenMP if available " ON)
option(NDZIP_WITH_HIPSYCL "Enable GPU implementation through hipSYCL if available" ON)
option(NDZIP_WITH_CUDA "Enable GPU implementation through CUDA if available" ON)
option(NDZIP_WITH_3RDPARTY_BENCHMARKS "Build third-party libraries for benchmarking" ON)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(SplitConfiguration)

find_package(Boost REQUIRED COMPONENTS thread program_options)

if (NDZIP_WITH_MT)
    find_package(OpenMP)
    set(NDZIP_USE_OPENMP "${OpenMP_FOUND}")
endif ()
if (NDZIP_WITH_HIPSYCL)
    find_package(hipSYCL CONFIG)
    set(NDZIP_USE_HIPSYCL "${hipSYCL_FOUND}")
endif ()
if (NDZIP_WITH_CUDA)
    find_package(CUDAToolkit)
    set(NDZIP_USE_CUDA "${CUDAToolkit_FOUND}")
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
    endif ()
endif ()

if (NDZIP_BUILD_TEST)
    find_package(Catch2)
endif()

if (NDZIP_BUILD_BENCHMARK AND NDZIP_WITH_3RDPARTY_BENCHMARKS)
    find_package(ZLIB QUIET)
    find_package(PkgConfig)
    pkg_check_modules(LZMA liblzma>=5.0.0 QUIET IMPORTED_TARGET)
    pkg_check_modules(LZ4 liblz4>=1.0.0 QUIET IMPORTED_TARGET)
    pkg_check_modules(ZSTD libzstd>=1.0.0 QUIET IMPORTED_TARGET)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/fpzip/CMakeLists.txt")
        set(FPZIP_FOUND TRUE)
    else ()
        message(WARNING "Building without fpzip benchmark, perform git submodule step to enable")
    endif ()

    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/contrib/cudpp/CMakeLists.txt")
        message(WARNING "Building without cudpp benchmark, perform git submodule step to enable")
    elseif (CMAKE_CUDA_COMPILER_ID STREQUAL Clang)
        message(WARNING "Building without cudpp which does not currently support Clang")
    else ()
        set(CUDPP_FOUND TRUE)
    endif ()

    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/contrib/nvcomp/CMakeLists.txt")
        message(WARNING "Building without NVCOMP benchmark, perform git submodule step to enable")
    elseif (CMAKE_CUDA_COMPILER_ID STREQUAL Clang)
        message(WARNING "Building without NVCOMP which does not currently support Clang")
    else ()
        set(NVCOMP_FOUND TRUE)
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/zfp/CMakeLists.txt")
        set(ZFP_FOUND TRUE)
    else ()
        message(WARNING "Building without ZFP benchmark, perform git submodule step to enable")
    endif ()
endif ()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(NDZIP_COMPILE_FLAGS -Wall -Wextra -Wno-attributes -Wimplicit-fallthrough)
set(NDZIP_CXX_FLAGS ${NDZIP_COMPILE_FLAGS} -Werror=return-type -Werror=init-self -Werror=undef)

if (NDZIP_USE_HIPSYCL)
    # Aggressive inlining avoids GPU call stack allocation == global memory access bottleneck
    set(NDZIP_HIPSYCL_FLAGS ${NDZIP_CXX_FLAGS} -mllvm -inline-threshold=2000 -Xarch_device -DNDEBUG=1)
endif()

if (NDZIP_USE_CUDA)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
        set(NDZIP_CUDA_FLAGS ${NDZIP_COMPILE_FLAGS}
            --extended-lambda  # Allow [] __device__ () lambda syntax
            --expt-relaxed-constexpr # Allow calling constexpr __host__ in __device__ constexprs
            # All diagnostics disabled here were false-positives
            -Xcudafe --display_error_number
            -Xcudafe --diag_suppress=2473 # always_inline is ignored on non-inline functions
            -Xcudafe --diag_suppress=186 # pointless comparison of unsigned integer with zero
            -Xcudafe --diag_suppress=940 # missing return statement at end of non-void function
        )
    else ()
        set(NDZIP_CUDA_FLAGS ${NDZIP_CXX_FLAGS} -mllvm -inline-threshold=2000 -Xarch_device -DNDEBUG=1)
    endif ()
endif ()

set(NDZIP_PROFILE_CONFIGURATIONS
    VARIABLE DATA_TYPE VALUES float double
    VARIABLE DIMENSIONS VALUES 1 2 3
)

add_library(ndzip SHARED
    include/ndzip/ndzip.hh
    include/ndzip/cpu_encoder.hh
    include/ndzip/cpu_encoder.hh
    include/ndzip/sycl_encoder.hh
    src/ndzip/common.hh
    src/ndzip/common.cc
    src/ndzip/cpu_encoder.inl
)
target_split_configured_sources(ndzip PRIVATE
    GENERATE cpu_encoder.cc FROM src/ndzip/cpu_encoder.inl
    ${NDZIP_PROFILE_CONFIGURATIONS}
)

target_include_directories(ndzip PUBLIC include)
target_compile_definitions(ndzip PUBLIC
    -DNDZIP_HIPSYCL_SUPPORT=$<BOOL:${NDZIP_USE_HIPSYCL}>
    -DNDZIP_CUDA_SUPPORT=$<BOOL:${NDZIP_USE_CUDA}>
    -DNDZIP_OPENMP_SUPPORT=$<BOOL:${OpenMP_FOUND}>
)
target_compile_options(ndzip PRIVATE ${NDZIP_CXX_FLAGS})

if (NDZIP_USE_OPENMP)
    target_link_libraries(ndzip PRIVATE OpenMP::OpenMP_CXX Boost::thread)
endif ()

if (NDZIP_USE_HIPSYCL)
    add_library(ndzip-sycl SHARED
        include/ndzip/sycl.hh
        src/ndzip/sycl_bits.hh
        src/ndzip/sycl_encoder.inl
    )
    target_split_configured_sources(ndzip-sycl PRIVATE
        GENERATE sycl_encoder.cc FROM src/ndzip/sycl_encoder.inl
        ${NDZIP_PROFILE_CONFIGURATIONS}
    )
    target_include_directories(ndzip-sycl PUBLIC include)
    target_compile_options(ndzip-sycl PRIVATE ${NDZIP_HIPSYCL_FLAGS})
    add_sycl_to_target(TARGET ndzip-sycl)
    target_link_libraries(ndzip-sycl PUBLIC ndzip)
endif ()

if (NDZIP_USE_CUDA)
    add_library(ndzip-cuda SHARED
        include/ndzip/cuda.hh
        src/ndzip/cuda_bits.cuh
        src/ndzip/cuda_encoder.inl
    )
    target_split_configured_sources(ndzip-cuda PRIVATE
        GENERATE cuda_encoder.cu FROM src/ndzip/cuda_encoder.inl
        ${NDZIP_PROFILE_CONFIGURATIONS}
    )
    target_include_directories(ndzip-cuda PUBLIC include)
    # target_compile_options and the COMPILE_OPTIONS property will de-duplicate flags which breaks
    # -Xcudafe <arg>. Use (deprecated) COMPILE_FLAGS instead.
    list(JOIN NDZIP_CUDA_FLAGS " " NDZIP_CUDA_FLAGS_STRING)
    set_target_properties(ndzip-cuda PROPERTIES COMPILE_FLAGS "${NDZIP_CUDA_FLAGS_STRING}")
    target_link_libraries(ndzip-cuda PUBLIC ndzip CUDA::cudart)
endif ()

add_library(io STATIC
    src/io/io.cc
    src/io/io.hh
)

target_include_directories(io INTERFACE src)

add_executable(compress
    src/compress/compress.cc
)

target_compile_options(compress PRIVATE ${NDZIP_CXX_FLAGS})
target_link_libraries(compress PRIVATE ndzip io Boost::program_options)
if (NDZIP_USE_HIPSYCL)
    target_link_libraries(compress PRIVATE ndzip-sycl)
endif ()
if (NDZIP_USE_CUDA)
    target_link_libraries(compress PRIVATE ndzip-cuda)
endif ()


if (NDZIP_BUILD_TEST AND Catch2_FOUND)
    add_executable(encoder_test
        src/test/test_utils.hh
        src/test/test_main.cc
        src/test/encoder_generic_test.cc
        src/test/encoder_profile_test.inl)

    target_split_configured_sources(encoder_test PRIVATE
        GENERATE encoder_profile_test.cc FROM src/test/encoder_profile_test.inl
        ${NDZIP_PROFILE_CONFIGURATIONS}
        )

    if (NDZIP_USE_CUDA AND NOT NDZIP_USE_HIPSYCL)
        # TODO hack: SYCL / CUDA versions of encoder_test must be compiled separately
        #   to allow the CUDA part to optionally use NVCC instead of Clang. This only
        #   works with NVCC if SYCL is disabled.
        get_target_property(ENCODER_TEST_SOURCES encoder_test SOURCES)
        foreach (source in ${ENCODER_TEST_SOURCES})
            if (source MATCHES ".*/encoder_profile_test.*\\.cc")
                set_source_files_properties("${source}" PROPERTIES
                    LANGUAGE CUDA
                    COMPILE_FLAGS "${NDZIP_CUDA_FLAGS_STRING}")
            endif ()
        endforeach ()
    endif()

    target_include_directories(encoder_test PRIVATE src)
    target_link_libraries(encoder_test PRIVATE ndzip Catch2::Catch2 Boost::thread)

    if (NDZIP_USE_HIPSYCL)
        target_compile_options(encoder_test PRIVATE ${NDZIP_HIPSYCL_FLAGS})
        add_sycl_to_target(TARGET encoder_test)
        target_link_libraries(encoder_test PRIVATE ndzip-sycl)
    endif ()
    if (NDZIP_USE_CUDA)
        target_link_libraries(encoder_test PRIVATE ndzip-cuda)
    endif ()

    if (NDZIP_USE_HIPSYCL)
        add_executable(sycl_bits_test
            src/test/test_utils.hh
            src/test/test_main.cc
            src/test/sycl_bits_test.cc)
        target_include_directories(sycl_bits_test PRIVATE src include)
        target_compile_options(sycl_bits_test PRIVATE ${NDZIP_HIPSYCL_FLAGS})
        target_link_libraries(sycl_bits_test PRIVATE Catch2::Catch2)
        add_sycl_to_target(TARGET sycl_bits_test)

        add_executable(sycl_ubench
                src/sycl_ubench/ubench.hh
                src/sycl_ubench/encoder_ubench.inl
                src/sycl_ubench/sycl_bits_ubench.cc
                src/sycl_ubench/ubench_main.cc)
        target_split_configured_sources(sycl_ubench PRIVATE
                GENERATE encoder_ubench.cc FROM src/sycl_ubench/encoder_ubench.inl
                ${NDZIP_PROFILE_CONFIGURATIONS}
                )
        target_include_directories(sycl_ubench PRIVATE src include)
        target_compile_options(sycl_ubench PRIVATE ${NDZIP_HIPSYCL_FLAGS})
        target_link_libraries(sycl_ubench PRIVATE ndzip-sycl Catch2::Catch2)
        add_sycl_to_target(TARGET sycl_ubench)
    endif ()

    if (NDZIP_USE_CUDA)
        add_executable(cuda_bits_test
                src/test/test_utils.hh
                src/test/test_main.cc
                src/test/cuda_bits_test.cu)
        target_include_directories(cuda_bits_test PRIVATE src include)
        set_source_files_properties(src/test/cuda_bits_test.cu
                PROPERTIES COMPILE_FLAGS "${NDZIP_CUDA_FLAGS_STRING}")
        target_link_libraries(cuda_bits_test PRIVATE ndzip-cuda Catch2::Catch2)
    endif ()
endif ()


if (NDZIP_BUILD_BENCHMARK AND NDZIP_WITH_3RDPARTY_BENCHMARKS)
    find_library(STD_CPP_FS stdc++fs)

    if (FPZIP_FOUND)
        set(BUILD_UTILITIES OFF CACHE BOOL "Build command-line utilities for fpzip")
        set(BUILD_TESTING OFF CACHE BOOL "Build tests")
        add_subdirectory(contrib/fpzip EXCLUDE_FROM_ALL)
    endif ()

    add_subdirectory(contrib/fpc EXCLUDE_FROM_ALL)
    add_subdirectory(contrib/pfpc EXCLUDE_FROM_ALL)
    add_subdirectory(contrib/spdp EXCLUDE_FROM_ALL)

    if (CUDPP_FOUND)
        set(CUDPP_BUILD_SHARED_LIBS ON CACHE BOOL "On to build shared libraries, off for static libraries.")
        add_subdirectory(contrib/cudpp EXCLUDE_FROM_ALL)
        include_directories(contrib/cudpp/include)
    endif ()

    if (CUDAToolkit_FOUND)
        add_subdirectory(contrib/gfc EXCLUDE_FROM_ALL)
        add_subdirectory(contrib/mpc EXCLUDE_FROM_ALL)
    else ()
        message(WARNING "CUDA not found, omitting some algorithms in benchmark")
    endif ()

    if (NVCOMP_FOUND)
        add_subdirectory(contrib/nvcomp EXCLUDE_FROM_ALL)
    endif ()

    if (ZFP_FOUND)
        add_subdirectory(contrib/zfp EXCLUDE_FROM_ALL)
    endif ()
endif ()

if (NDZIP_BUILD_BENCHMARK)
    add_executable(benchmark src/benchmark/benchmark.cc)
    target_compile_options(benchmark PRIVATE ${NDZIP_CXX_FLAGS})
    target_link_libraries(benchmark PRIVATE ndzip io Boost::program_options)
    if (STD_CPP_FS)
        target_link_libraries(benchmark PRIVATE ${STD_CPP_FS})
    endif ()
    target_compile_definitions(benchmark PRIVATE
        "-DNDZIP_BENCHMARK_HAVE_3RDPARTY=$<BOOL:${NDZIP_WITH_3RDPARTY_BENCHMARKS}>"
        "-DNDZIP_BENCHMARK_HAVE_ZLIB=$<BOOL:${ZLIB_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_LZ4=$<BOOL:${LZ4_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_LZMA=$<BOOL:${LZMA_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_FPZIP=$<BOOL:${FPZIP_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_CUDPP=$<BOOL:${CUDPP_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_NVCOMP=$<BOOL:${NVCOMP_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_ZFP=$<BOOL:${ZFP_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_GFC=$<BOOL:${CUDAToolkit_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_MPC=$<BOOL:${CUDAToolkit_FOUND}>"
        "-DNDZIP_BENCHMARK_HAVE_ZSTD=$<BOOL:${ZSTD_FOUND}>"
        )
    if (NDZIP_USE_OPENMP)
        target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX Boost::thread)
    endif ()
    if (NDZIP_USE_HIPSYCL)
        target_link_libraries(benchmark PRIVATE ndzip-sycl)
    endif ()
    if (NDZIP_USE_CUDA)
        target_link_libraries(benchmark PRIVATE ndzip-cuda)
    endif ()
endif ()

if (NDZIP_BUILD_BENCHMARK AND NDZIP_WITH_3RDPARTY_BENCHMARKS)
    target_link_libraries(benchmark PRIVATE fpc pfpc spdp)
    if (ZLIB_FOUND)
        target_link_libraries(benchmark PRIVATE ZLIB::ZLIB)
    endif ()
    if (LZ4_FOUND)
        target_link_libraries(benchmark PRIVATE PkgConfig::LZ4)
    endif ()
    if (LZMA_FOUND)
        target_link_libraries(benchmark PRIVATE PkgConfig::LZMA)
    endif ()
    if (FPZIP_FOUND)
        target_link_libraries(benchmark PRIVATE fpzip::fpzip)
    endif ()
    if (CUDPP_FOUND)
        target_link_libraries(benchmark PRIVATE cudpp)
    endif ()
    if (CUDAToolkit_FOUND)
        target_link_libraries(benchmark PRIVATE gfc)
        target_link_libraries(benchmark PRIVATE mpc)
    endif ()
    if (ZSTD_FOUND)
        target_link_libraries(benchmark PRIVATE PkgConfig::ZSTD)
    endif ()
    if (NVCOMP_FOUND)
        target_link_libraries(benchmark PRIVATE nvcomp CUDA::cudart)
    endif ()
    if (ZFP_FOUND)
        target_link_libraries(benchmark PRIVATE zfp)
    endif ()
endif ()
