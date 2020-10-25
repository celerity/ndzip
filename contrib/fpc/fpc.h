#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t FPC_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, long predsizem1);

size_t FPC_Decompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream);

extern const char *FPC_Version_String;

#ifdef __cplusplus
}
#endif
