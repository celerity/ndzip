project(pFPC VERSION 1.0)

add_library(pfpc pFPC.h pFPC.c)
set_target_properties(pfpc PROPERTIES C_STANDARD 99)
target_include_directories(pfpc PUBLIC .)
