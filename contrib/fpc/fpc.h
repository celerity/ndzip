#pragma once

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t FPC_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, long predsizem1);

size_t FPC_Deompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream);

#ifdef __cplusplus
}
#endif
