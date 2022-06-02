#pragma once

// CUDA 11.6 #defines __noinline__, but GCC 12.1.0 headers use __attribute__(__noinline__). Remove the definition in
// that case. This requires "cuda_workaround.hh" to be included before <memory>.
#ifdef __noinline__
#undef __noinline__
#endif
