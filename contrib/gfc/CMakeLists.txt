project(GFC VERSION 2.2 LANGUAGES CUDA)

add_library(gfc GFC_22.cu GFC_22.h)
set_target_properties(gfc PROPERTIES CUDA_ARCHITECTURES 35)
set_target_properties(gfc PROPERTIES CUDA_STANDARD 14) # allow register keyword
target_include_directories(gfc PUBLIC .)
