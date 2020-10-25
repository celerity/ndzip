#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t pFPC_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, int predsizelg2, int threads, int chunksize);

size_t pFPC_Decompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream);

extern const char *pFPC_Version_String;

#ifdef __cplusplus
}
#endif
