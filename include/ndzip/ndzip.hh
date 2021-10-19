#pragma once

#include "array.hh"
#include "cpu_encoder.hh"
#if NDZIP_OPENMP_SUPPORT
#include "mt_cpu_encoder.hh"
#endif
#if NDZIP_HIPSYCL_SUPPORT
#include "sycl_encoder.hh"
#endif
#if NDZIP_CUDA_SUPPORT
#include "cuda_encoder.hh"
#endif
