cmake_minimum_required(VERSION 3.10)

project(hcde)
set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Catch2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror=return-type -Werror=init-self -Werror=undef")

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

set_target_properties(hcde PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(hcde ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(hcde PUBLIC include)
target_link_libraries(hcde PRIVATE ${CMAKE_THREAD_LIBS_INIT})

add_executable(compress src/compress/compress.cc)
target_link_libraries(compress hcde Boost::program_options)

if (Catch2_FOUND)
    add_executable(test src/test/test.cc)
    target_link_libraries(test hcde Catch2::Catch2)
endif ()
