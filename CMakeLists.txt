cmake_minimum_required(VERSION 3.10)

project(hcde)

option(HCDE_BUILD_TEST "Build unit tests" ON)
option(HCDE_BUILD_BENCHMARK "Build benchmarks against other algorithms" ON)


find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(hipSYCL CONFIG)

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
set(HCDE_COMPILE_FLAGS -Wall -Wextra -Werror=return-type -Werror=init-self -Werror=undef)
if(CMAKE_GENERATOR STREQUAL "Ninja")
    # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
    # Rationale in https://github.com/ninja-build/ninja/issues/814
    set(HCDE_COMPILE_FLAGS ${COMPILE_FLAGS} -fdiagnostics-color=always)
endif()


add_library(hcde STATIC
    include/hcde/hcde.hh
    src/hcde/common.hh
    src/hcde/cpu_encoder.inl
    src/hcde/cpu_encoder.cc
    src/hcde/mt_cpu_encoder.inl
    src/hcde/mt_cpu_encoder.cc
)

target_include_directories(hcde PUBLIC include)
target_compile_options(hcde PRIVATE ${HCDE_COMPILE_FLAGS})
target_link_libraries(hcde PRIVATE Threads::Threads)


if (hipSYCL_FOUND)
    target_compile_definitions(hcde PUBLIC -DHCDE_GPU_SUPPORT=1)
    target_sources(hcde PRIVATE
        src/hcde/gpu_encoder.inl
        src/hcde/gpu_encoder.cc
    )
    add_sycl_to_target(TARGET hcde SOURCES src/hcde/gpu_encoder.cc)
else ()
    target_compile_definitions(hcde PUBLIC -DHCDE_GPU_SUPPORT=0)
endif ()


add_executable(compress
    src/compress/compress.cc
    src/compress/io.cc
    src/compress/io.hh
)
target_compile_options(compress PRIVATE ${HCDE_COMPILE_FLAGS})
target_link_libraries(compress PRIVATE hcde Boost::program_options)

if (hipSYCL_FOUND)
    # For some reason, add_sycl_to_target does not link `compress` to OpenMP when using hipCPU
    add_sycl_to_target(TARGET compress)
endif ()


if (HCDE_BUILD_TEST AND Catch2_FOUND)
    add_executable(test src/test/test.cc)
    target_include_directories(test PRIVATE src)
    target_compile_options(test PRIVATE ${HCDE_COMPILE_FLAGS})
    target_link_libraries(test PRIVATE hcde Catch2::Catch2)
endif ()


if (HCDE_BUILD_BENCHMARK)
    if (FPZIP_FOUND)
        set(BUILD_UTILITIES OFF CACHE BOOL "Build command-line utilities for fpzip")
        set(BUILD_TESTING OFF CACHE BOOL "Build tests")
        add_subdirectory(contrib/fpzip)
    endif ()

    add_executable(benchmark src/benchmark/benchmark.cc)
    target_compile_options(benchmark PRIVATE ${HCDE_COMPILE_FLAGS})
    target_link_libraries(benchmark PRIVATE hcde Boost::program_options)
    target_compile_definitions(benchmark PRIVATE
        "-DHCDE_BENCHMARK_HAVE_ZLIB=$<BOOL:ZLIB_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_LZ4=$<BOOL:LZ4_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_LZMA=$<BOOL:LZMA_FOUND>"
        "-DHCDE_BENCHMARK_HAVE_FPZIP=$<BOOL:FPZIP_FOUND>"
    )

    if (hipSYCL_FOUND)
        add_sycl_to_target(TARGET benchmark)
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
endif ()
