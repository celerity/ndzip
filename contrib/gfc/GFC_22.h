#pragma once

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void GFC_Init();

size_t GFC_CompressBound(size_t in_size);

size_t GFC_Compress_Memory(const void *in_stream, size_t in_size, void *out_stream, int blocks,
    int warpsperblock, int dimensionality, uint64_t *kernel_time_us);

size_t GFC_Decompress_Memory(const void *in_stream, size_t in_size, void *out_stream,
    uint64_t *kernel_time_us);

extern const char *GFC_Version_String;

#ifdef __cplusplus
}
#endif
