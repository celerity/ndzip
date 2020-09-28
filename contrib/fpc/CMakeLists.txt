project(FPC VERSION 1.1)

add_library(fpc fpc.h fpc.c)
set_target_properties(fpc PROPERTIES C_STANDARD 99)
target_include_directories(fpc PUBLIC .)
