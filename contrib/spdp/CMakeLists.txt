project(SPDP VERSION 1.1)

add_library(spdp SPDP_11.h SPDP_11.c)
target_include_directories(spdp PUBLIC .)
