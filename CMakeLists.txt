cmake_minimum_required(VERSION 3.10)

project(hcde)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror=return-type -Werror=init-self")
include_directories(include)

add_library(hcde STATIC
    include/hcde.hh
    src/common.hh
    src/fast_profile.hh
    src/strong_profile.hh
    src/singlethread_cpu.hh
)

set_target_properties(hcde PROPERTIES LINKER_LANGUAGE CXX)

add_executable(compress src/compress.cc)
target_link_libraries(compress hcde Boost::program_options)

add_executable(test test/test.cc)
target_link_libraries(test hcde)
