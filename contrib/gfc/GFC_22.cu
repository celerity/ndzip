/*
GFC code: A GPU-based compressor for arrays of double-precision
floating-point values.

Copyright (c) 2011-2020, Texas State University. All rights reserved.

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

Authors: Molly A. O'Neil and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/GFC/.

Publication: This work is described in detail in the following paper.
Molly A. O'Neil and Martin Burtscher. Floating-Point Data Compression at 75
Gb/s on a GPU. Proceedings of the Fourth Workshop on General Purpose Processing
Using GPUs, pp. 7:1-7:7. March 2011.
*/


#include "GFC_22.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

#define ull unsigned long long
#define MAX (128*1024*1024)

#define WARPSIZE 32

__constant__ int dimensionalityd; // dimensionality parameter
__constant__ ull *cbufd; // ptr to uncompressed data
__constant__ unsigned char *dbufd; // ptr to compressed data
__constant__ ull *fbufd; // ptr to decompressed data
__constant__ int *cutd; // ptr to chunk boundaries
__constant__ int *offd; // ptr to chunk offsets after compression

/************************************************************************************/

/*
This is the GPU compression kernel, which should be launched using the block count
and warps/block:
  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();

Inputs
------
dimensionalityd: dimensionality of trace (from cmd line)
cbufd: ptr to the uncompressed data
cutd: ptr to array of chunk boundaries

Output
------
The compressed data, in dbufd 
Compressed chunk offsets for offset table, in offd
*/

__global__ void CompressionKernel()
{
  int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
  ull diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to compress
  start = 0;
  if (warp > 0) start = cutd[warp-1];
  term = cutd[warp];
  off = ((start+1)/2*17);

  // __syncwarp() requires all threads to participate (otherwise we need to derive an active mask)
  assert((term - start) % WARPSIZE == 0);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // calculate delta between value to compress and prediction
    // and negate if negative
    diff = cbufd[i] - prev;
    code = (diff >> 60) & 8;
    if (code != 0) {
      diff = -diff;
    }

    // count leading zeros in positive delta
    bcount = 8 - (__clzll(diff) >> 3);
    if (bcount == 2) bcount = 3; // encode 6 lead-zero bytes as 5

    // prefix sum to determine start positions of non-zero delta bytes
    ibufs[iindex] = bcount;
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-1];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-2];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-4];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-8];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-16];
    __syncwarp();

    // write out non-zero bytes of delta to compressed buffer
    beg = off + (WARPSIZE/2) + ibufs[iindex-1];
    end = beg + bcount;
    for (; beg < end; beg++) {
      dbufd[beg] = diff;
      diff >>= 8;
    }

    if (bcount >= 3) bcount--; // adjust byte count for the dropped encoding
    tmp = ibufs[lastidx];
    code |= bcount;
    ibufs[iindex] = code;
    __syncwarp();

    // write out half-bytes of sign and leading-zero-byte count (every other thread
    // writes its half-byte and neighbor's half-byte)
    if ((lane & 1) != 0) {
      dbufd[off + (lane >> 1)] = ibufs[iindex-1] | (code << 4);
    }
    off += tmp + (WARPSIZE/2);

    // save prediction value from this subchunk (based on provided dimensionality)
    // for use in next subchunk
    prev = cbufd[i + offset];
  }

  // save final value of off, which is total bytes of compressed output for this chunk
  if (lane == 31) offd[warp] = off;
}

/************************************************************************************/

/*
This is the GPU decompression kernel, which should be launched using the block count
and warps/block:
  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();

Inputs
------
dimensionalityd: dimensionality of trace
dbufd: ptr to array of compressed data
cutd: ptr to array of chunk boundaries

Output
------
The decompressed data in fbufd
*/

__global__ void DecompressionKernel()
{
  int offset, code, bcount, off, beg, end, lane, warp, iindex, lastidx, start, term;
  ull diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)];

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to decompress
  start = 0;
  if (warp > 0) start = cutd[warp-1];
  term = cutd[warp];
  off = ((start+1)/2*17);

  // __syncwarp() requires all threads to participate (otherwise we need to derive an active mask)
  assert((term - start) % WARPSIZE == 0);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // read in half-bytes of size and leading-zero count information
    if ((lane & 1) == 0) {
      code = dbufd[off + (lane >> 1)];
      ibufs[iindex] = code;
      ibufs[iindex + 1] = code >> 4;
    }
    off += (WARPSIZE/2);
    __syncwarp();
    code = ibufs[iindex];

    bcount = code & 7;
    if (bcount >= 2) bcount++;

    // calculate start positions of compressed data
    ibufs[iindex] = bcount;
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-1];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-2];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-4];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-8];
    __syncwarp();
    ibufs[iindex] += ibufs[iindex-16];
    __syncwarp();

    // read in compressed data (the non-zero bytes)
    beg = off + ibufs[iindex-1];
    off += ibufs[lastidx];
    end = beg + bcount - 1;
    diff = 0;
    for (; beg <= end; end--) {
      diff <<= 8;
      diff |= dbufd[end];
    }

    // negate delta if sign bit indicates it was negated during compression
    if ((code & 8) != 0) {
      diff = -diff;
    }

    // write out the uncompressed word
    fbufd[i] = prev + diff;
    __syncwarp();

    // save prediction for next subchunk
    prev = fbufd[i + offset];
  }
}

/************************************************************************************/

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


// Burtscher's implementation has been slightly modified to read from / write to memory instead of files.

static size_t stream_read(void *buffer, size_t size, size_t items, const void *stream, size_t bytes, size_t *cursor) {
    assert(*cursor <= bytes);
    size_t remaining_items = (bytes - *cursor) / size;
    size_t read = items < remaining_items ? items : remaining_items;
    memcpy(buffer, (const char*) stream + *cursor, read * size);
    *cursor += read * size;
    return read;
}

static size_t stream_write(const void *buffer, size_t size, size_t items, void *stream, size_t *cursor) {
    memcpy((char*) stream + *cursor, buffer, size * items);
    *cursor += size * items;
    return items;
}


/************************************************************************************/

size_t GFC_CompressBound(size_t in_size)
{
    return 7 + ((in_size + 7) / 8 + 1) / 2 * 17;
}

size_t GFC_Compress_Memory(const void *in_stream, size_t in_size, void *out_stream, int blocks,
    int warpsperblock, int dimensionality, uint64_t *kernel_time_us)
{
  if (sizeof(ull) * MAX < in_size) {
    fprintf(stderr, "GFC: in_size %zu exceeds max %zu\n", in_size, sizeof(ull)*MAX); abort();
  }

  size_t in_cursor = 0;
  size_t out_cursor = 0;

  cudaGetLastError();  // reset error value

  // allocate CPU buffers
  ull *cbuf = (ull *)malloc(sizeof(ull) * MAX); // uncompressed data
  if (cbuf == NULL) {
    fprintf(stderr, "cannot allocate cbuf\n"); abort();
  }
  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data
  if (dbuf == NULL) {
    fprintf(stderr, "cannot allocate dbuf\n"); abort();
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) {
    fprintf(stderr, "cannot allocate cut\n"); abort();
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if (off == NULL) {
    fprintf(stderr, "cannot allocate off\n"); abort();
  }

  // read in trace to cbuf
  int doubles = stream_read(cbuf, 8, MAX, in_stream, in_size, &in_cursor);

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = std::min(curr, doubles);
    if (cut[i] - before > 0) {
      d = cut[i] - before;
    }
    before = cut[i];
  }

  // set the pad values to ensure correct prediction
  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = 0;
    }
  } else {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
    }
  }

  // allocate GPU buffers
  ull *cbufl; // uncompressed data
  char *dbufl; // compressed data
  int *cutl; // chunk boundaries
  int *offl; // offset table
  if (cudaSuccess != cudaMalloc((void **)&cbufl, sizeof(ull) * doubles))
    fprintf(stderr, "could not allocate cbufd\n");
  CudaTest("couldn't allocate cbufd");
  if (cudaSuccess != cudaMalloc((void **)&dbufl, sizeof(char) * ((doubles+1)/2*17)))
    fprintf(stderr, "could not allocate dbufd\n");
  CudaTest("couldn't allocate dbufd");
  if (cudaSuccess != cudaMalloc((void **)&cutl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate cutd\n");
  CudaTest("couldn't allocate cutd");
  if (cudaSuccess != cudaMalloc((void **)&offl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate offd\n");
  CudaTest("couldn't allocate offd");

  // copy buffer starting addresses (pointers) and values to constant memory
  if (cudaSuccess != cudaMemcpyToSymbol(dimensionalityd, &dimensionality, sizeof(int)))
    fprintf(stderr, "copying of dimensionality to device failed\n");
  CudaTest("dimensionality copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cbufd, &cbufl, sizeof(void *)))
    fprintf(stderr, "copying of cbufl to device failed\n");
  CudaTest("cbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dbufd, &dbufl, sizeof(void *)))
    fprintf(stderr, "copying of dbufl to device failed\n");
  CudaTest("dbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cutd, &cutl, sizeof(void *)))
    fprintf(stderr, "copying of cutl to device failed\n");
  CudaTest("cutl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(offd, &offl, sizeof(void *)))
    fprintf(stderr, "copying of offl to device failed\n");
  CudaTest("offl copy to device failed");

  // copy CPU buffer contents to GPU
  if (cudaSuccess != cudaMemcpy(cbufl, cbuf, sizeof(ull) * doubles, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cbuf to device failed\n");
  CudaTest("cbuf copy to device failed");
  if (cudaSuccess != cudaMemcpy(cutl, cut, sizeof(int) * blocks * warpsperblock, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cut to device failed\n");
  CudaTest("cut copy to device failed");

  cudaEvent_t begin, end;
  if (kernel_time_us) {
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin, NULL);
  }

  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();
  CudaTest("compression kernel launch failed");

  if (kernel_time_us) {
    cudaEventRecord(end, NULL);
    float duration_ms;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration_ms, begin, end);
    *kernel_time_us = (uint64_t) (duration_ms * 1000);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
  }

  // transfer offsets back to CPU
  if(cudaSuccess != cudaMemcpy(off, offl, sizeof(int) * blocks * warpsperblock, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of off from device failed\n");
  CudaTest("off copy from device failed");

  // output header
  int num;
  int doublecnt = doubles-padding;
  num = stream_write(&blocks, 1, 1, out_stream, &out_cursor);
  assert(1 == num);
  num = stream_write(&warpsperblock, 1, 1, out_stream, &out_cursor);
  assert(1 == num);
  num = stream_write(&dimensionality, 1, 1, out_stream, &out_cursor);
  assert(1 == num);
  num = stream_write(&doublecnt, 4, 1, out_stream, &out_cursor);
  assert(1 == num);
  (void) num; // silence unused warning
  // output offset table
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int start = 0;
    if(i > 0) start = cut[i-1];
    off[i] -= ((start+1)/2*17);
    num = stream_write(&off[i], 4, 1, out_stream, &out_cursor); // chunk's compressed size in bytes
    assert(1 == num);
  }
  // output compressed data by chunk
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int offset, start = 0;
    if(i > 0) start = cut[i-1];
    offset = ((start+1)/2*17);
    // transfer compressed data back to CPU by chunk
    if (cudaSuccess != cudaMemcpy(dbuf + offset, dbufl + offset, sizeof(char) * off[i], cudaMemcpyDeviceToHost))
      fprintf(stderr, "copying of dbuf from device failed\n");
    CudaTest("dbuf copy from device failed");
    num = stream_write(&dbuf[offset], 1, off[i], out_stream, &out_cursor);
    assert(off[i] == num);
  }
  (void) num; // silence unused warning

  free(cbuf);
  free(dbuf);
  free(cut);
  free(off);

  if (cudaSuccess != cudaFree(cbufl))
    fprintf(stderr, "could not deallocate cbufd\n");
  CudaTest("couldn't deallocate cbufd");
  if (cudaSuccess != cudaFree(dbufl))
    fprintf(stderr, "could not deallocate dbufd\n");
  CudaTest("couldn't deallocate dbufd");
  if (cudaSuccess != cudaFree(cutl))
    fprintf(stderr, "could not deallocate cutd\n");
  CudaTest("couldn't deallocate cutd");
  if (cudaSuccess != cudaFree(offl))
    fprintf(stderr, "could not deallocate offd\n");
  CudaTest("couldn't deallocate offd");

  return out_cursor;
}

/************************************************************************************/

size_t GFC_Decompress_Memory(const void *in_stream, size_t in_size, void *out_stream, uint64_t *kernel_time_us)
{
  size_t in_cursor = 0;
  size_t out_cursor = 0;

  cudaGetLastError();  // reset error value

  int num, doubles;
  int blocks, warpsperblock, dimensionality;

  num = stream_read(&blocks, 1, 1, in_stream, in_size, &in_cursor);
  assert(1 == num);
  blocks &= 255;
  num = stream_read(&warpsperblock, 1, 1, in_stream, in_size, &in_cursor);
  assert(1 == num);
  warpsperblock &= 255;
  num = stream_read(&dimensionality, 1, 1, in_stream, in_size, &in_cursor);
  assert(1 == num);
  dimensionality &= 255;
  num = stream_read(&doubles, 4, 1, in_stream, in_size, &in_cursor);
  assert(1 == num);
  (void) num; // silence unused warning

  // allocate CPU buffers
  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data, divided by chunk
  if (dbuf == NULL) { 
    fprintf(stderr, "cannot allocate dbuf\n"); exit(-1); 
  }
  ull *fbuf = (ull *)malloc(sizeof(ull) * MAX); // decompressed data
  if (fbuf == NULL) { 
    fprintf(stderr, "cannot allocate fbuf\n"); exit(-1);
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) { 
    fprintf(stderr, "cannot allocate cut\n"); exit(-1);
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if(off == NULL) {
    fprintf(stderr, "cannot allocate off\n"); exit(-1);
  }

  // read in offset table
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num = stream_read(&off[i], 4, 1, in_stream, in_size, &in_cursor);
    assert(1 == num);
  }

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock); 
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = std::min(curr, doubles);
  }

  // allocate GPU buffers
  char *dbufl; // compressed data
  ull *fbufl; // uncompressed data
  int *cutl; // chunk boundaries
  if (cudaSuccess != cudaMalloc((void **)&dbufl, sizeof(char) * ((doubles+1)/2*17)))
    fprintf(stderr, "could not allocate dbufd\n");
  CudaTest("couldn't allocate dbufd");
  if (cudaSuccess != cudaMalloc((void **)&fbufl, sizeof(ull) * doubles))
    fprintf(stderr, "could not allocate fbufd\n");
  CudaTest("couldn't allocate fbufd");
  if (cudaSuccess != cudaMalloc((void **)&cutl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate cutd\n");
  CudaTest("couldn't allocate cutd");

  // copy buffer starting addresses (pointers) and values to constant memory
  if (cudaSuccess != cudaMemcpyToSymbol(dimensionalityd, &dimensionality, sizeof(int))) 
    fprintf(stderr, "copying of dimensionality to device failed\n");
  CudaTest("dimensionality copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dbufd, &dbufl, sizeof(void *)))
    fprintf(stderr, "copying of dbufl to device failed\n");
  CudaTest("dbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(fbufd, &fbufl, sizeof(void *)))
    fprintf(stderr, "copying of fbufl to device failed\n");
  CudaTest("fbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cutd, &cutl, sizeof(void *)))
    fprintf(stderr, "copying of cutl to device failed\n");
  CudaTest("cutl copy to device failed");

  // read in input data and divide into chunks
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num, chbeg, start = 0;
    if (i > 0) start = cut[i-1];
    chbeg = ((start+1)/2*17);
    // read in this chunk of data (based on offsets)
    num = stream_read(&dbuf[chbeg], 1, off[i], in_stream, in_size, &in_cursor);
    assert(off[i] == num);
    (void) num; // silence unused warning
    // transfer the chunk to the GPU
    if (cudaSuccess != cudaMemcpy(dbufl + chbeg, dbuf + chbeg, sizeof(char) * off[i], cudaMemcpyHostToDevice))
      fprintf(stderr, "copying of dbuf to device failed\n");
    CudaTest("dbuf copy to device failed");
  }

  // copy CPU cut buffer contents to GPU
  if (cudaSuccess != cudaMemcpy(cutl, cut, sizeof(int) * blocks * warpsperblock, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cut to device failed\n");
  CudaTest("cut copy to device failed");

  cudaEvent_t begin, end;
  if (kernel_time_us) {
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin, NULL);
  }

  DecompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();
  CudaTest("decompression kernel launch failed");

  if (kernel_time_us) {
    cudaEventRecord(end, NULL);
    float duration_ms;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration_ms, begin, end);
    *kernel_time_us = (uint64_t) (duration_ms * 1000);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
  }

  // transfer result back to CPU
  if (cudaSuccess != cudaMemcpy(fbuf, fbufl, sizeof(ull) * doubles, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of fbuf from device failed\n");
  CudaTest("fbuf copy from device failed");

  // output decompressed data
  num = stream_write(fbuf, 8, doubles-padding, out_stream, &out_cursor);
  assert(num == doubles-padding);
  (void) num; // silence unused warning

  free(dbuf);
  free(fbuf);
  free(cut);

  if(cudaSuccess != cudaFree(dbufl))
    fprintf(stderr, "could not deallocate dbufd\n");
  CudaTest("couldn't deallocate dbufd");
  if(cudaSuccess != cudaFree(fbufl))
    fprintf(stderr, "could not deallocate fbufd\n");
  CudaTest("couldn't deallocate dbufd");
  if(cudaSuccess != cudaFree(cutl))
    fprintf(stderr, "could not deallocate cutd\n");
  CudaTest("couldn't deallocate cutd");

  return in_cursor;
}

/************************************************************************************/

static int VerifySystemParameters()
{
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));
#ifndef NDEBUG
  int val = 1;
  assert(1 == *((char *)&val));
#endif

  int current_device = 0, sm_per_multiproc = 0; 
  int max_compute_perf = 0, max_perf_device = 0; 
  int device_count = 0, best_SM_arch = 0; 
  int arch_cores_sm[3] = { 1, 8, 32 }; 
  cudaDeviceProp deviceProp; 

  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }
   
  // Find the best major SM Architecture GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) { 
      best_SM_arch = std::max(best_SM_arch, deviceProp.major);
    }
  }
   
  // Find the best CUDA capable GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device); 
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } 
    else if (deviceProp.major <= 2) { 
      sm_per_multiproc = arch_cores_sm[deviceProp.major]; 
    } 
    else { // Device has SM major > 2 
      sm_per_multiproc = arch_cores_sm[2]; 
    }
      
    int compute_perf = deviceProp.multiProcessorCount * 
                       sm_per_multiproc * deviceProp.clockRate; 
      
    if (compute_perf > max_compute_perf) { 
      // If we find GPU of SM major > 2, search only these 
      if (best_SM_arch > 2) { 
        // If device==best_SM_arch, choose this, or else pass 
        if (deviceProp.major == best_SM_arch) { 
          max_compute_perf = compute_perf; 
          max_perf_device = current_device; 
        } 
      } 
      else { 
        max_compute_perf = compute_perf; 
        max_perf_device = current_device; 
      } 
    } 
  } 
   
  cudaGetDeviceProperties(&deviceProp, max_perf_device); 
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable  device\n");
    exit(-1);
  }
  if (deviceProp.major < 2) {
    fprintf(stderr, "Need at least compute capability 2.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }
  if ((WARPSIZE <= 0) || ((WARPSIZE & (WARPSIZE-1)) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }

  return max_perf_device;
}

/************************************************************************************/

void GFC_Init()
{
    int device;

    device = VerifySystemParameters();
    cudaSetDevice(device);

    cudaFuncSetCacheConfig(CompressionKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(DecompressionKernel, cudaFuncCachePreferL1);
}


const char *GFC_Version_String = "GPU FP Compressor v2.2";

#ifdef __cplusplus
}
#endif
