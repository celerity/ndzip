project(MPC VERSION 1.2 LANGUAGES CUDA)

add_library(mpc MPC_12.h MPC_float_12.cu MPC_double_12.cu)
target_include_directories(mpc PUBLIC .)
