cmake_minimum_required(VERSION 3.10)

project(hcde)

option(HCDE_BUILD_TEST "Build unit tests" ON)
option(HCDE_BUILD_BENCHMARK "Build benchmarks against other algorithms" ON)
option(HCDE_WITH_MT "Enable parallel CPU implementation through OpenMP if available " ON)
option(HCDE_WITH_GPU "Enable GPU implementation through hipSYCL if available" ON)

find_package(Boost REQUIRED COMPONENTS thread program_options)

if (HCDE_WITH_MT)
    find_package(OpenMP)
    set(HCDE_USE_OPENMP "${OpenMP_FOUND}")
endif ()
if (HCDE_WITH_GPU)
    find_package(hipSYCL CONFIG)
    set(HCDE_USE_HIPSYCL "${hipSYCL_FOUND}")
endif ()

if (HCDE_BUILD_TEST)
    find_package(Catch2)
endif()

if (HCDE_BUILD_BENCHMARK)
    find_package(ZLIB QUIET)
    find_package(PkgConfig)
    pkg_check_modules(LZMA liblzma>=5.0.0 QUIET IMPORTED_TARGET)
    pkg_check_modules(LZ4 liblz4>=1.0.0 QUIET IMPORTED_TARGET)

    if (EXISTS "${CMAKE_SOURCE_DIR}/contrib/fpzip/CMakeLists.txt")
        set(FPZIP_FOUND TRUE)
    else ()
        message(WARNING "Building without fpzip benchmark, perform git submodule step to enable")
    endif ()
endif ()


set(CMAKE_CXX_STANDARD 17)
set(HCDE_COMPILE_FLAGS
    -Wall -Wextra -Werror=return-type -Werror=init-self -Werror=undef -Wno-attributes
    # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
    # Rationale in https://github.com/ninja-build/ninja/issues/814
    $<$<STREQUAL:${CMAKE_GENERATOR},Ninja>:-fdiagnostics-color=always>)

add_library(hcde SHARED
    include/hcde/hcde.hh
    src/hcde/common.hh
    src/hcde/common.cc
    src/hcde/cpu_encoder.inl
    src/hcde/cpu_encoder.cc
)

target_include_directories(hcde PUBLIC include)
target_compile_definitions(hcde PUBLIC
    -DHCDE_GPU_SUPPORT=$<BOOL:hipSYCL_FOUND>
    -DHCDE_OPENMP_SUPPORT=$<BOOL:OpenMP_FOUND>
)
target_compile_options(hcde PRIVATE ${HCDE_COMPILE_FLAGS})

if (HCDE_USE_OPENMP)
    target_link_libraries(hcde PRIVATE OpenMP::OpenMP_CXX Boost::thread)
endif ()

if (HCDE_USE_HIPSYCL)
    add_library(hcde-gpu SHARED
        src/hcde/gpu_encoder.inl
        src/hcde/gpu_encoder.cc
    )
    target_include_directories(hcde-gpu PUBLIC include)
    target_compile_options(hcde-gpu PRIVATE ${HCDE_COMPILE_FLAGS})
    add_sycl_to_target(TARGET hcde-gpu)
    target_link_libraries(hcde-gpu PUBLIC hcde)
endif ()


add_executable(compress
    src/compress/compress.cc
    src/compress/io.cc
    src/compress/io.hh
)
target_compile_options(compress PRIVATE ${HCDE_COMPILE_FLAGS})
target_link_libraries(compress PRIVATE hcde Boost::program_options)
if (HCDE_USE_HIPSYCL)
    target_link_libraries(compress PRIVATE hcde-gpu)
endif ()


if (HCDE_BUILD_TEST AND Catch2_FOUND)
    add_executable(test src/test/test.cc)
    target_include_directories(test PRIVATE src)
    target_compile_options(test PRIVATE ${HCDE_COMPILE_FLAGS})
    target_link_libraries(test PRIVATE hcde Catch2::Catch2)
    if (HCDE_USE_HIPSYCL)
        target_link_libraries(test PRIVATE hcde-gpu)
    endif ()
endif ()


if (HCDE_BUILD_BENCHMARK)
    find_package(CUDAToolkit)

    if (FPZIP_FOUND)
        set(BUILD_UTILITIES OFF CACHE BOOL "Build command-line utilities for fpzip")
        set(BUILD_TESTING OFF CACHE BOOL "Build tests")
        add_subdirectory(contrib/fpzip)
    endif ()

    add_subdirectory(contrib/fpc)
    add_subdirectory(contrib/spdp)

    if (CUDAToolkit_FOUND)
        add_subdirectory(contrib/gfc)
    else ()
        message(WARNING "CUDA not found, omitting some algorithms in benchmark")
    endif ()

    add_executable(benchmark src/benchmark/benchmark.cc)
    target_compile_options(benchmark PRIVATE ${HCDE_COMPILE_FLAGS})
    target_link_libraries(benchmark PRIVATE hcde Boost::program_options fpc spdp)
    target_compile_definitions(benchmark PRIVATE
        "-DHCDE_BENCHMARK_HAVE_ZLIB=$<BOOL:ZLIB_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_LZ4=$<BOOL:LZ4_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_LZMA=$<BOOL:LZMA_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_FPZIP=$<BOOL:FPZIP_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_GFC=$<BOOL:CUDAToolkit_FOUND>"
    )
    if (HCDE_USE_HIPSYCL)
        target_link_libraries(benchmark PRIVATE hcde-gpu)
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
    if (CUDAToolkit_FOUND)
        target_link_libraries(benchmark PRIVATE gfc)
    endif ()
endif ()
