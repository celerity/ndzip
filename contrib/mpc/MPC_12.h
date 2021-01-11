#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int MPC_float_compressBound(int insize);

int MPC_float_compressMemory(
        int *output, const int *input, int insize, int dim, uint64_t *kernel_time_us);

int MPC_float_decompressedSize(const int *input, int insize);

int MPC_float_decompressMemory(int *output, const int *input, int insize, uint64_t *kernel_time_us);

int MPC_double_compressBound(int insize);

int MPC_double_compressMemory(
        long *output, const long *input, int insize, int dim, uint64_t *kernel_time_us);

int MPC_double_decompressedSize(const long *input, int insize);

int MPC_double_decompressMemory(
        long *output, const long *input, int insize, uint64_t *kernel_time_us);

extern const char *MPC_Version_String;

#ifdef __cplusplus
}
#endif
