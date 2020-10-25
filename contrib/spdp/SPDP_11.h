#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char byte_t;

size_t SPDP_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, int level);

size_t SPDP_Decompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream);

extern const char *SPDP_Version_String;

#ifdef __cplusplus
}
#endif
