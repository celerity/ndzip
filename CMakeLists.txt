cmake_minimum_required(VERSION 3.10)

project(hcde)
set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Catch2)
find_package(hipSYCL CONFIG)

set(COMPILE_FLAGS "-Wall -Wextra -Werror=return-type -Werror=init-self -Werror=undef")
if(CMAKE_GENERATOR STREQUAL "Ninja")
    # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
    # Rationale in https://github.com/ninja-build/ninja/issues/814
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -fdiagnostics-color=always")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} ${COMPILE_FLAGS}")

add_library(hcde STATIC
    include/hcde.hh
    src/hcde/common.hh
    src/hcde/fast_profile.inl
    src/hcde/strong_profile.inl
    src/hcde/cpu_encoder.inl
    src/hcde/cpu_encoder.cc
    src/hcde/mt_cpu_encoder.inl
    src/hcde/mt_cpu_encoder.cc
)

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

target_include_directories(hcde PUBLIC include)
target_link_libraries(hcde PRIVATE ${CMAKE_THREAD_LIBS_INIT})

add_executable(compress
    src/compress/compress.cc
    src/compress/io.cc
    src/compress/io.hh
)
target_link_libraries(compress PRIVATE hcde Boost::program_options)

if (hipSYCL_FOUND)
    # For some reason, add_sycl_to_target does not link `compress` to OpenMP when using hipCPU
    add_sycl_to_target(TARGET compress)
endif ()

if (Catch2_FOUND)
    add_executable(test src/test/test.cc)
    if (hipSYCL_FOUND)
        add_sycl_to_target(TARGET test SOURCES src/test/test.cc)
    endif ()
    target_link_libraries(test PRIVATE hcde Catch2::Catch2)
endif ()
