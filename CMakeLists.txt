cmake_minimum_required(VERSION 3.10)

project(ndzip)

option(NDZIP_BUILD_TEST "Build unit tests" ON)
option(NDZIP_BUILD_BENCHMARK "Build benchmarks against other algorithms" ON)
option(NDZIP_WITH_MT "Enable parallel CPU implementation through OpenMP if available " ON)
option(NDZIP_WITH_GPU "Enable GPU implementation through hipSYCL if available" ON)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(SplitConfiguration)

find_package(Boost REQUIRED COMPONENTS thread program_options)

if (NDZIP_WITH_MT)
    find_package(OpenMP)
    set(NDZIP_USE_OPENMP "${OpenMP_FOUND}")
endif ()
if (NDZIP_WITH_GPU)
    find_package(hipSYCL CONFIG)
    set(NDZIP_USE_HIPSYCL "${hipSYCL_FOUND}")
endif ()

if (NDZIP_BUILD_TEST)
    find_package(Catch2)
endif()

if (NDZIP_BUILD_BENCHMARK)
    find_package(ZLIB QUIET)
    find_package(PkgConfig)
    pkg_check_modules(LZMA liblzma>=5.0.0 QUIET IMPORTED_TARGET)
    pkg_check_modules(LZ4 liblz4>=1.0.0 QUIET IMPORTED_TARGET)
    pkg_check_modules(ZSTD libzstd>=1.0.0 QUIET IMPORTED_TARGET)

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/fpzip/CMakeLists.txt")
        set(FPZIP_FOUND TRUE)
    else ()
        message(WARNING "Building without fpzip benchmark, perform git submodule step to enable")
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/cudpp/CMakeLists.txt")
        set(CUDPP_FOUND TRUE)
    else ()
        message(WARNING "Building without cudpp benchmark, perform git submodule step to enable")
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/nvcomp/CMakeLists.txt")
        set(NVCOMP_FOUND TRUE)
    else ()
        message(WARNING "Building without NVCOMP benchmark, perform git submodule step to enable")
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/zfp/CMakeLists.txt")
        set(ZFP_FOUND TRUE)
    else ()
        message(WARNING "Building without ZFP benchmark, perform git submodule step to enable")
    endif ()
endif ()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(NDZIP_COMPILE_FLAGS
    -Wall -Wextra -Werror=return-type -Werror=init-self -Werror=undef -Wno-attributes -Wimplicit-fallthrough
    # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
    # Rationale in https://github.com/ninja-build/ninja/issues/814
    $<$<STREQUAL:${CMAKE_GENERATOR},Ninja>:-fdiagnostics-color=always>)

# Aggressive inlining avoids GPU call stack allocation == global memory access bottleneck
set(NDZIP_GPU_COMPILE_FLAGS -mllvm -inline-threshold=2000 -Xarch_device -DNDEBUG=1)

set(NDZIP_PROFILE_CONFIGURATIONS
    VARIABLE DATA_TYPE VALUES float double
    VARIABLE DIMENSIONS VALUES 1 2 3
)

add_library(ndzip SHARED
    include/ndzip/ndzip.hh
    include/ndzip/array.hh
    include/ndzip/cpu_encoder.hh
    include/ndzip/mt_cpu_encoder.hh
    include/ndzip/gpu_encoder.hh
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
    -DNDZIP_GPU_SUPPORT=$<BOOL:${NDZIP_USE_HIPSYCL}>
    -DNDZIP_OPENMP_SUPPORT=$<BOOL:${OpenMP_FOUND}>
)
target_compile_options(ndzip PRIVATE ${NDZIP_COMPILE_FLAGS})

if (NDZIP_USE_OPENMP)
    target_link_libraries(ndzip PRIVATE OpenMP::OpenMP_CXX Boost::thread)
endif ()

if (NDZIP_USE_HIPSYCL)
    add_library(ndzip-gpu SHARED
        src/ndzip/gpu_bits.hh
        src/ndzip/gpu_encoder.inl
    )
    target_split_configured_sources(ndzip-gpu PRIVATE
        GENERATE gpu_encoder.cc FROM src/ndzip/gpu_encoder.inl
        ${NDZIP_PROFILE_CONFIGURATIONS}
    )
    target_include_directories(ndzip-gpu PUBLIC include)
    target_compile_options(ndzip-gpu PRIVATE ${NDZIP_COMPILE_FLAGS} ${NDZIP_GPU_COMPILE_FLAGS})
    add_sycl_to_target(TARGET ndzip-gpu)
    target_link_libraries(ndzip-gpu PUBLIC ndzip)
endif ()

add_library(io STATIC
    src/io/io.cc
    src/io/io.hh
)

target_include_directories(io INTERFACE src)

add_executable(compress
    src/compress/compress.cc
)

target_compile_options(compress PRIVATE ${NDZIP_COMPILE_FLAGS})
target_link_libraries(compress PRIVATE ndzip io Boost::program_options)
if (NDZIP_USE_HIPSYCL)
    target_link_libraries(compress PRIVATE ndzip-gpu)
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
    target_include_directories(encoder_test PRIVATE src)
    target_compile_options(encoder_test PRIVATE ${NDZIP_COMPILE_FLAGS} ${NDZIP_GPU_COMPILE_FLAGS})
    target_link_libraries(encoder_test PRIVATE ndzip Catch2::Catch2 Boost::thread)
    if (NDZIP_USE_HIPSYCL)
        add_sycl_to_target(TARGET encoder_test)
        target_link_libraries(encoder_test PRIVATE ndzip-gpu)
    endif ()
endif ()

if (NDZIP_BUILD_TEST AND Catch2_FOUND AND NDZIP_USE_HIPSYCL)
    add_executable(gpu_bits_test
            src/test/test_utils.hh
            src/test/test_main.cc
            src/test/gpu_bits_test.cc)
    target_include_directories(gpu_bits_test PRIVATE src include)
    target_compile_options(gpu_bits_test PRIVATE ${NDZIP_COMPILE_FLAGS} -Werror=pass-failed ${NDZIP_GPU_COMPILE_FLAGS})
    target_link_libraries(gpu_bits_test PRIVATE Catch2::Catch2)
    add_sycl_to_target(TARGET gpu_bits_test)

    add_executable(gpu_ubench
        src/gpu_ubench/ubench.hh
        src/gpu_ubench/encoder_ubench.inl
        src/gpu_ubench/gpu_bits_ubench.cc
        src/gpu_ubench/ubench_main.cc)
    target_split_configured_sources(gpu_ubench PRIVATE
        GENERATE encoder_ubench.cc FROM src/gpu_ubench/encoder_ubench.inl
        ${NDZIP_PROFILE_CONFIGURATIONS}
    )
    target_include_directories(gpu_ubench PRIVATE src include)
    target_compile_options(gpu_ubench PRIVATE ${NDZIP_COMPILE_FLAGS} ${NDZIP_GPU_COMPILE_FLAGS})
    target_link_libraries(gpu_ubench PRIVATE ndzip-gpu Catch2::Catch2)
    add_sycl_to_target(TARGET gpu_ubench)
endif ()


if (NDZIP_BUILD_BENCHMARK)
    find_package(CUDAToolkit)
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

    add_executable(benchmark src/benchmark/benchmark.cc)
    target_compile_options(benchmark PRIVATE ${NDZIP_COMPILE_FLAGS})
    target_link_libraries(benchmark PRIVATE ndzip io Boost::program_options fpc pfpc spdp)
    if (STD_CPP_FS)
        target_link_libraries(benchmark PRIVATE ${STD_CPP_FS})
    endif ()
    target_compile_definitions(benchmark PRIVATE
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
        target_link_libraries(benchmark PRIVATE ndzip-gpu)
    endif ()
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
        target_link_libraries(benchmark PRIVATE nvcomp)
    endif ()
    if (ZFP_FOUND)
        target_link_libraries(benchmark PRIVATE zfp)
    endif ()
endif ()
