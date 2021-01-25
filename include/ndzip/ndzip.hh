#pragma once

#include "array.hh"
#include "cpu_encoder.hh"
#if NDZIP_OPENMP_SUPPORT
#include "mt_cpu_encoder.hh"
#endif
#if NDZIP_GPU_SUPPORT
#include "gpu_encoder.hh"
#endif  // NDZIP_GPU_SUPPORT
