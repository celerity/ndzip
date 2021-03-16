/*
MPC code [float] (LnVs BIT LVs ZE): A GPU-based compressor for arrays of 
single-precision floating-point values.  See the following publication for
more information: http://cs.txstate.edu/~mb92/papers/cluster15.pdf.

Copyright (c) 2015-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Annie Yang and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/MPC/.

Publication: This work is described in detail in the following paper.
Annie Yang, Hari Mukka, Farbod Hesaaraki, and Martin Burtscher. MPC: A
Massively Parallel Compression Algorithm for Scientific Data. Proceedings
of the IEEE International Conference on Cluster Computing, pp. 381-389.
September 2015.
*/


#include "MPC_12.h"
#include <cassert>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

#define TPB 1024  /* do not change */

// BLOCKS_PER_SM on the device must be equal to blocksPerSM (see below) on the host
#ifdef __CUDA_ARCH__
#   if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
#      define BLOCKS_PER_SM 1
#   elif __CUDA_ARCH__ <= 860
#      define BLOCKS_PER_SM 2
#   else
#      error Check how many 1024-thread blocks per SM your target supports and update BLOCKS_PER_SM
#   endif
#else
#   define BLOCKS_PER_SM 0 // dummy for host compilation
#endif

#if __CUDA_ARCH__ >= 700 || CUDART_VERSION >= 9000
// Unsynced versions are deprecated
#define __shfl(...) __shfl_sync(0xffffffff, __VA_ARGS__)
#define __shfl_up(...) __shfl_up_sync(0xffffffff, __VA_ARGS__)
#define __ballot(...) __ballot_sync(0xffffffff, __VA_ARGS__)
#endif

static inline __device__
void prefixsum2(int &val, int sbuf[TPB], int &valb, int sbufb[TPB])
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  for (int d = 1; d < 32; d *= 2) {
    int tmp = __shfl_up(val, d);
    if (lane >= d) val += tmp;
  }
  if (lane == 31) sbuf[warp] = val;

  for (int d = 1; d < 32; d *= 2) {
    int tmpb = __shfl_up(valb, d);
    if (lane >= d) valb += tmpb;
  }
  if (lane == 31) sbufb[warp] = valb;

  __syncthreads();
  if (warp == 0) {
    int v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      int tmp = __shfl_up(v, d);
      if (lane >= d) v += tmp;
    }
    sbuf[lane] = v;

    int vb = sbufb[lane];
    for (int d = 1; d < 32; d *= 2) {
      int tmpb = __shfl_up(vb, d);
      if (lane >= d) vb += tmpb;
    }
    sbufb[lane] = vb;
  }

  __syncthreads();
  if (warp > 0) {
    val += sbuf[warp - 1];
    valb += sbufb[warp - 1];
  }
}

static inline __device__
void prefixsum2dim(int &val, int sbuf[TPB], int &valb, int sbufb[TPB], const unsigned char dim)
{
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tix = (warp * dim) + (tid % dim);

  for (int d = dim; d < 32; d *= 2) {
    int tmp = __shfl_up(val, d);
    if (lane >= d) val += tmp;
  }

  for (int d = dim; d < 32; d *= 2) {
    int tmpb = __shfl_up(valb, d);
    if (lane >= d) valb += tmpb;
  }

  if ((lane + dim) > 31) {
    sbuf[tix] = val;
    sbufb[tix] = valb;
  }

  __syncthreads();
  if (warp < dim) {
    const int idx = (lane * dim) + warp;

    int v = sbuf[idx];
    for (int d = 1; d < 32; d *= 2) {
      int tmp = __shfl_up(v, d);
      if (lane >= d) v += tmp;
    }
    sbuf[idx] = v;

    int vb = sbufb[idx];
    for (int d = 1; d < 32; d *= 2) {
      int tmpb = __shfl_up(vb, d);
      if (lane >= d) vb += tmpb;
    }
    sbufb[idx] = vb;
  }

  __syncthreads();
  if (warp > 0) {
    val += sbuf[tix - dim];
    valb += sbufb[tix - dim];
  }
}

/*****************************************************************************
This is the GPU compression kernel, which requires 1024 threads per block and
should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
n: the number of float values to be compressed
original: the input array holding the n floats (has to be cast to an int array)
goffset: a temporary array with m elements where m = number of thread blocks
dim: the dimensionality of the input data (dim must be between 1 and 32)

Output
------
compressed: the output array that holds the compressed data in integer format

The output array needs to provide space for up to 2 + n + (n + 31) / 32 elements.
The second element of the output array specifies how many elements are actually
used.  It should be replaced by the value n before the data is further processed.
*****************************************************************************/

static __global__ __launch_bounds__(TPB, BLOCKS_PER_SM)
void MPCcompress(const int n, int* const original, int* const compressed, volatile int* const goffset, unsigned char dim)
{
  const int tid = threadIdx.x;
  const int tidm1 = tid - 1;
  const int tidmdim = tid - dim;
  const int lane = tid & 31;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 2 + (n + 31) / 32;
  const int chunksm1 = ((n + ((TPB * 2) - 1)) / (TPB * 2)) - 1;

  __shared__ int start, startb, top, topb, sbuf1[TPB], sbuf1b[TPB], sbuf2[TPB], sbuf2b[TPB];

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * (TPB * 2);
    const int idxb = idx + TPB;

    int v1 = 0;
    if (idx < n) {
      v1 = original[idx];
      sbuf1[tid] = v1;
    }

    int v1b = 0;
    if (idxb < n) {
      v1b = original[idxb];
      sbuf1b[tid] = v1b;
    }

    __syncthreads();
    if (tid >= dim) {
      if (idx < n) {
        v1 -= sbuf1[tidmdim];

        if (idxb < n) {
          v1b -= sbuf1b[tidmdim];
        }
      }
    }

    int v2 = 0;
    for (int i = 31; i >= 0; i--) {
      v2 = (v2 << 1) + ((__shfl(v1, i) >> lane) & 1);
    }
    sbuf2[tid] = v2;

    int v2b = 0;
    for (int i = 31; i >= 0; i--) {
      v2b = (v2b << 1) + ((__shfl(v1b, i) >> lane) & 1);
    }
    sbuf2b[tid] = v2b;

    __syncthreads();
    if (tid > 0) {
      v2 -= sbuf2[tidm1];
      v2b -= sbuf2b[tidm1];
    }

    int bitmap = __ballot(v2);
    int bitmapb = __ballot(v2b);

    if (lane == 0) {
      if (idx < n) {
        compressed[2 + idx / 32] = bitmap;

        if (idxb < n) {
          compressed[2 + idxb / 32] = bitmapb;
        }
      }
    }

    int loc = 0;
    if (v2 != 0) loc = 1;

    int locb = 0;
    if (v2b != 0) locb = 1;

    prefixsum2(loc, sbuf1, locb, sbuf1b);

    if (v2 != 0) {
      sbuf2[loc - 1] = v2;
    }

    if (v2b != 0) {
      sbuf2b[locb - 1] = v2b;
    }

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc + locb;
      goffset[bid] = -1;
      if (chunk == chunksm1) {
        compressed[0] = (0x63706d00 - 1) + dim;
        compressed[1] = st + loc + locb;
      }
      top = loc;
      topb = locb;
      start = st;
      startb = st + loc;
    }

    __syncthreads();
    if (tid < top) {
      compressed[start + tid] = sbuf2[tid];
    }
    if (tid < topb) {
      compressed[startb + tid] = sbuf2b[tid];
    }
  }
}

/*****************************************************************************
This is the GPU decompression kernel, which requires 1024 threads per block
and should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
compressed: the input array holding the compressed data in integer format
goffset: a temporary array with m elements where m = number of thread blocks

The second element of the input array must hold the value n, i.e., the number
of floats that the data will generate upon decompression.

Output
------
decompressed: the output array holding the decompressed data in integer format

The output array needs to provide space for n elements has to be cast to an
array of floats before it can be used.
*****************************************************************************/

static __global__ __launch_bounds__(TPB, BLOCKS_PER_SM)
void MPCdecompress(int* const compressed, int* const decompressed, volatile int* const goffset)
{
  const int dim = (compressed[0] & 31) + 1;
  const int n = compressed[1];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 2 + (n + 31) / 32;
  const int nru = (n - 1) | 31;
  const int chunksm1 = ((n + ((TPB * 2) - 1)) / (TPB * 2)) - 1;

  __shared__ int start, startb, top, topb, sbuf1[TPB], sbuf1b[TPB], sbuf2[TPB], sbuf2b[TPB];

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * (TPB * 2);
    const int idxb = idx + TPB;

    int flag = 0;
    if (idx <= nru) {
      flag = (compressed[2 + idx / 32] >> lane) & 1;
    }
    int loc = flag;

    int flagb = 0;
    if (idxb <= nru) {
      flagb = (compressed[2 + idxb / 32] >> lane) & 1;
    }
    int locb = flagb;

    prefixsum2(loc, sbuf1, locb, sbuf1b);

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc + locb;
      goffset[bid] = -1;
      top = loc;
      topb = locb;
      start = st;
      startb = st + loc;
    }

    __syncthreads();
    if (tid < top) {
      sbuf2[tid] = compressed[start + tid];
    }

    if (tid < topb) {
      sbuf2b[tid] = compressed[startb + tid];
    }

    __syncthreads();
    int v2 = 0;
    if (flag != 0) {
      v2 = sbuf2[loc - 1];
    }

    int v2b = 0;
    if (flagb != 0) {
      v2b = sbuf2b[locb - 1];
    }

    prefixsum2(v2, sbuf1, v2b, sbuf1b);

    int v1 = 0;
    for (int i = 0; i < 32; i++) {
      v1 += ((__shfl(v2, i) >> lane) & 1) << i;
    }

    int v1b = 0;
    for (int i = 0; i < 32; i++) {
      v1b += ((__shfl(v2b, i) >> lane) & 1) << i;
    }

    prefixsum2dim(v1, sbuf2, v1b, sbuf2b, dim);

    if (idx < n) {
      decompressed[idx] = v1;
    }

    if (idxb < n) {
      decompressed[idxb] = v1b;
    }
  }
}

static void CudaTest(const char *msg)
{
  cudaError_t e;

  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    abort();
  }
}

static_assert(sizeof(int) == sizeof(float));

int MPC_float_compressBound(int insize)
{
    return insize + 2 + (insize + 31) / 32;
}

int MPC_float_compressMemory(int *output, const int *input, int insize, int dim,
        uint64_t *kernel_time_us)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "There is no CUDA capable device\n");
        abort();
    }
    if (deviceProp.major < 3) {
        fprintf(stderr, "Need at least compute capability 3.0\n");
        abort();
    }
    const int blocksPerSM = deviceProp.maxThreadsPerMultiProcessor / TPB;
    const int blocks = deviceProp.multiProcessorCount * blocksPerSM;

    int outsize = MPC_float_compressBound(insize);
    assert(0 < dim);  assert(dim <= 32);

    int *d_in, *d_out, *d_offs;
    cudaMalloc(&d_in, insize * sizeof(int));  CudaTest("malloc failed");
    cudaMalloc(&d_out, outsize * sizeof(int));  CudaTest("malloc failed");
    cudaMalloc(&d_offs, blocks * sizeof(int));  CudaTest("malloc failed");

    cudaMemcpy(d_in, input, insize * sizeof(int), cudaMemcpyHostToDevice);  CudaTest("memcpy failed");

    cudaEvent_t begin, end;
    if (kernel_time_us) {
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, NULL);
    }

    cudaMemset(d_offs, -1, blocks * sizeof(int));
    MPCcompress<<<blocks, TPB>>>(insize, d_in, d_out, d_offs, dim);
    CudaTest("compression failed");

    if (kernel_time_us) {
        cudaEventRecord(end, NULL);
        float duration_ms;
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&duration_ms, begin, end);
        *kernel_time_us = (uint64_t) (duration_ms * 1000);
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);  CudaTest("memcpy failed");
    outsize = output[1];

    cudaMemcpy(output, d_out, outsize * sizeof(int), cudaMemcpyDeviceToHost);  CudaTest("memcpy failed");
    output[1] = insize;

    cudaFree(d_offs);
    cudaFree(d_out);
    cudaFree(d_in);
    CudaTest("free failed");

    return outsize;
}

int MPC_float_decompressedSize(const int *input, int insize) {
    assert(insize > 0);
    assert((input[0] >> 8) == 0x63706d);
    return input[1];
}

int MPC_float_decompressMemory(int *output, const int *input, int insize, uint64_t *kernel_time_us)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "There is no CUDA capable device\n");
        abort();
    }
    if (deviceProp.major < 3) {
        fprintf(stderr, "Need at least compute capability 3.0\n");
        abort();
    }
    const int blocksPerSM = deviceProp.maxThreadsPerMultiProcessor / TPB;
    const int blocks = deviceProp.multiProcessorCount * blocksPerSM;

    int outsize = MPC_float_decompressedSize(input, insize);

    int *d_in, *d_out, *d_offs;
    cudaMalloc(&d_in, insize * sizeof(int));  CudaTest("malloc failed");
    cudaMalloc(&d_out, outsize * sizeof(int));  CudaTest("malloc failed");
    cudaMalloc(&d_offs, blocks * sizeof(int));  CudaTest("malloc failed");

    cudaMemcpy(d_in, input, insize * sizeof(int), cudaMemcpyHostToDevice);  CudaTest("memcpy failed");

    cudaEvent_t begin, end;
    if (kernel_time_us) {
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, NULL);
    }

    cudaMemset(d_offs, -1, blocks * sizeof(int));
    MPCdecompress<<<blocks, TPB>>>(d_in, d_out, d_offs);
    CudaTest("decompression failed");

    if (kernel_time_us) {
        cudaEventRecord(end, NULL);
        float duration_ms;
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&duration_ms, begin, end);
        *kernel_time_us = (uint64_t) (duration_ms * 1000);
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_out, outsize * sizeof(int), cudaMemcpyDeviceToHost);  CudaTest("memcpy failed");

    cudaFree(d_offs);
    cudaFree(d_out);
    cudaFree(d_in);
    CudaTest("free failed");

    return outsize;
}

const char *MPC_VersionString = "MPC - Massively Parallel Compression 1.2";

#ifdef __cplusplus
}
#endif
