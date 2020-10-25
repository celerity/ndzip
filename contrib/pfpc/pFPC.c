/*
Copyright (c) 2008 The University of Texas.  All rights reserved.
Author: Dr. Martin Burtscher

For Software License Terms and Conditions, see http://users.ices.utexas.edu/~burtscher/research/pFPC/pFPC_license.pdf.
*/

#include "pFPC.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>


#define ALIGNSIZE (4 * 1024)  /* should be equal to the system's page size */
#define BUFSIZE (8 * 1024 * 1024)  /* has to be a large power of two */
#define MAXTHREADS 128  /* must be less than 256 */

#define HLSHIFT 6
#define HRSHIFT 48
#define DLSHIFT 2
#define DRSHIFT 40

#define ull unsigned long long

static const ull mask[8] =
{0x0000000000000000LL,
 0x00000000000000ffLL,
 0x000000000000ffffLL,
 0x0000000000ffffffLL,
 0x000000ffffffffffLL,
 0x0000ffffffffffffLL,
 0x00ffffffffffffffLL,
 0xffffffffffffffffLL};


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

static volatile int g_predsizem1;
static volatile int g_chunksize;
static volatile int g_threads;
static volatile struct CompStruct {
  ull *ibuf;
  unsigned char *obuf;
  ull *fcm;
  ull *dfcm;
  int end;
} g_comp[MAXTHREADS];
static volatile struct DecompStruct {
  unsigned char *ibuf;
  ull *obuf;
  ull *fcm;
  ull *dfcm;
  int end;
} g_decomp[MAXTHREADS];
static pthread_t thread[MAXTHREADS];



/*
This is the function that should be called by each parallel thread to compress the chunks of data assigned to this thread.

Inputs
------
threads: the number of parallel threads
chunksize: the number of consecutive doubles that make up a chunk
end: the number of doubles to the end of ibuf
ibuf: a pointer to the first double in the input buffer that this thread should compress
obuf: a pointer to the first byte of the output buffer that this thread will write (make sure the buffer is large enough)
predsizem1: the fcm and dfcm predictor size (which must be a power of two) minus one [in number of doubles]
fcm: a pointer to the fcm predictor table
dfcm: a pointer to the dfcm predictor table

Output
------
The compressed data in obuf (the first four bytes specify the total number of bytes)
*/

static void CompressChunk(const int threads, int chunksize, const int end, const ull * restrict const ibuf, unsigned char * restrict const obuf, const int predsizem1, ull * restrict const fcm, ull * restrict const dfcm)
{
  register int i, o, out, hash, dhash, code, bcode, save, offset, increment, full, part;
  register ull val, lastval, stride, pred1, pred2, xor1, xor2;

  full = end / (chunksize * threads);
  part = end - full * chunksize * threads;
  if (part > chunksize) part = chunksize;

  out = full * chunksize + part;
  out = ((out + 1) >> 1) + 4;
  *((ull *)&obuf[out & -8]) = 0;

  hash = 0;
  dhash = 0;
  lastval = 0;
  pred1 = 0;
  pred2 = 0;
  save = 0;

  o = 8;
  offset = 0;
  increment = threads * chunksize;
  while (0 < (end - offset)) {
    if (chunksize > (end - offset)) {
      chunksize = end - offset;
    }
    for (i = 0; i < chunksize; i++) {
      val = ibuf[offset + i];
      xor1 = val ^ pred1;
      fcm[hash] = val;
      hash = ((hash << HLSHIFT) ^ (val >> HRSHIFT)) & predsizem1;
      pred1 = fcm[hash];

      stride = val - lastval;
      xor2 = val ^ (lastval + pred2);
      lastval = val;
      dfcm[dhash] = stride;
      dhash = ((dhash << DLSHIFT) ^ (stride >> DRSHIFT)) & predsizem1;
      pred2 = dfcm[dhash];

      code = 0;
      if (xor1 > xor2) {
        code = 0x8;
        xor1 = xor2;
      }
      bcode = 7;
      if (0 == (xor1 >> 56)) bcode = 6;
      if (0 == (xor1 >> 48)) bcode = 5;
      if (0 == (xor1 >> 40)) bcode = 4;
      if (0 == (xor1 >> 24)) bcode = 3;
      if (0 == (xor1 >> 16)) bcode = 2;
      if (0 == (xor1 >> 8)) bcode = 1;
      if (0 == xor1) bcode = 0;

      *((ull *)&obuf[out & -8]) |= xor1 << ((out & 0x7) << 3);
      *((ull *)&obuf[(out & -8) + 8]) = (xor1 >> (63 - ((out & 0x7) << 3))) >> 1;

      code |= bcode;
      out += bcode + (bcode >> 2);
      obuf[o >> 1] = save | code;
      save = code << 4;
      o++;
    }
    offset += increment;
  }
  if (0 != (o & 1)) {
    obuf[o >> 1] = save;
  }

  *((int *)&obuf[0]) = out;  // number of (compressed) bytes
}


static void *CompressChunks(void *arg)
{
  register int id = (int)(uintptr_t)arg;

  // retrieve the 'parameters' passed via global memory
  CompressChunk(g_threads, g_chunksize, g_comp[id].end, g_comp[id].ibuf, g_comp[id].obuf, g_predsizem1, g_comp[id].fcm, g_comp[id].dfcm);
  pthread_exit(NULL);
}


static void Compress(int num, ull *inbuf, unsigned char **inter, int predsizem1, int threads, int chunksize, ull **fcm, ull **dfcm, void *out_stream, size_t *out_cursor)
{
  register int i, cnt;
  int status;
  pthread_attr_t attr;

  // prepare pthreads
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // can't pass parameters directly in pthreads, so copy them to global structure
  g_predsizem1 = predsizem1;
  g_chunksize = chunksize;
  g_threads = threads;
  for (i = 0; i < threads; i++) {
    g_comp[i].ibuf = &inbuf[i * chunksize];
    g_comp[i].obuf = inter[i];
    g_comp[i].fcm = fcm[i];
    g_comp[i].dfcm = dfcm[i];
    g_comp[i].end = num - (i * chunksize);
    if (0 > g_comp[i].end) g_comp[i].end = 0;
  }

  // launch parallel threads
  for (i = 0; i < threads; i++) {
    pthread_create(&thread[i], &attr, CompressChunks, (void *)(uintptr_t)i);
  }

  // write block header
  cnt = stream_write(&num, 4, 1, out_stream, out_cursor);
  assert(cnt == 1);

  // wait for parallel threads to finish
  for (i = 0; i < threads; i++) {
    pthread_join(thread[i], (void **)&status);
    assert(0 == status);

    // output compressed data
    cnt = *((int *)inter[i]);  // number of (compressed) bytes
    num = stream_write(inter[i], 1, cnt, out_stream, out_cursor);
    assert(num == cnt);
  }

  pthread_attr_destroy(&attr);
}


/*
This is the function that should be called by each parallel thread to compress the chunks of data assigned to this thread.

Inputs
------
threads: the number of parallel threads
chunksize: the number of consecutive doubles that make up a chunk
end: the number of doubles to the end of the obuf
ibuf: a pointer to the first byte in the input buffer that this thread should decompress
obuf: a pointer to the first double of the output buffer that this thread will write (make sure the buffer is large enough)
predsizem1: the fcm and dfcm predictor size (which must be a power of two) minus one [in number of doubles]
fcm: a pointer to the fcm predictor table
dfcm: a pointer to the dfcm predictor table

Output
------
The decompressed data in obuf
*/

static void DecompressChunk(const int threads, int chunksize, const int end, const unsigned char * restrict const ibuf, ull * restrict const obuf, const int predsizem1, ull * restrict const fcm, ull * restrict const dfcm)
{
  register int i, o, alt, in, hash, dhash, bcode, tmp, offset, increment, full, part;
  register ull val, lastval, stride, pred1, pred2, next;

  hash = 0;
  dhash = 0;
  lastval = 0;
  pred1 = 0;
  pred2 = 0;

  full = end / (chunksize * threads);
  part = end - full * chunksize * threads;
  if (part > chunksize)
    part = chunksize;
  in = full * chunksize + part;
  in = ((in + 1) >> 1) + 4;
  i = 8;
  alt = 4;

  offset = 0;
  increment = threads * chunksize;
  while (0 < (end - offset)) {
    if (chunksize > (end - offset)) {
      chunksize = end - offset;
    }
    for (o = 0; o < chunksize; o++) {
      bcode = ibuf[i >> 1] >> alt;
      i++;
      alt ^= 4;

      val = *((ull *)&ibuf[in & -8]);
      next = *((ull *)&ibuf[(in & -8) + 8]);
      tmp = (in & 0x7) << 3;
      val >>= tmp;
      next <<= 63 - tmp;
      val |= next + next;

      if (0 != (bcode & 0x8))
        pred1 = pred2;
      bcode &= 0x7;
      val &= mask[bcode];
      in += bcode + (bcode >> 2);
      val ^= pred1;

      fcm[hash] = val;
      hash = ((hash << HLSHIFT) ^ (val >> HRSHIFT)) & predsizem1;
      pred1 = fcm[hash];

      stride = val - lastval;
      dfcm[dhash] = stride;
      dhash = ((dhash << DLSHIFT) ^ (stride >> DRSHIFT)) & predsizem1;
      pred2 = val + dfcm[dhash];
      lastval = val;

      obuf[offset + o] = val;
    }
    offset += increment;
  }
}


static void *DecompressChunks(void *arg)
{
  register int id = (int)(uintptr_t)arg;

  // retrieve the 'parameters' passed via global memory
  DecompressChunk(g_threads, g_chunksize, g_decomp[id].end, g_decomp[id].ibuf, g_decomp[id].obuf, g_predsizem1, g_decomp[id].fcm, g_decomp[id].dfcm);
  pthread_exit(NULL);
}


static void Decompress(int num, unsigned char **inter, ull *outbuf, int predsizem1, int threads, int chunksize, ull **fcm, ull **dfcm, void *out_stream, size_t *out_cursor)
{
  register int i, cnt;
  int status;
  pthread_attr_t attr;

  // prepare pthreads
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // can't pass parameters directly in pthreads, so copy them to global structure
  g_predsizem1 = predsizem1;
  g_chunksize = chunksize;
  g_threads = threads;
  for (i = 0; i < threads; i++) {
    g_decomp[i].ibuf = inter[i];
    g_decomp[i].obuf = &outbuf[i * chunksize];
    g_decomp[i].fcm = fcm[i];
    g_decomp[i].dfcm = dfcm[i];
    g_decomp[i].end = num - (i * chunksize);
  }

  // launch parallel threads
  for (i = 0; i < threads; i++) {
    pthread_create(&thread[i], &attr, DecompressChunks, (void *)(uintptr_t)i);
  }

  // wait for parallel threads to finish
  for (i = 0; i < threads; i++) {
    pthread_join(thread[i], (void **)&status);
    assert(0 == status);
  }

  // output decompressed data
  cnt = stream_write(outbuf, 8, num, out_stream, out_cursor);
  assert(cnt == num);

  pthread_attr_destroy(&attr);
}


static unsigned char *AlignedAlloc(size_t size, size_t factor)
{
  void *buf;
  int error = posix_memalign(&buf, ALIGNSIZE, size * factor);
  if (error != 0) {
      fprintf(stderr, "posix_memalign(%zu, %zu): %s", ALIGNSIZE, size * factor, strerror(error));
      abort();
  }
  memset(buf, 0, size * factor);
  return buf;
}


size_t pFPC_Compress_Memory(const void *in_stream, size_t in_bytes, void *out_stream, int predsizelg2, int threads, int chunksize)
{
  size_t in_cursor = 0, out_cursor = 0;
  int i, num, chunkrepeat, predsizem1, val, cnt, in;
  register ull **fcm, **dfcm;
  register ull *input, *output;
  register unsigned char **inter;

  // make sure system parameters are okay
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));
  val = 1;
  assert(1 == *((char *)&val));

  // output header
  num = stream_write(&predsizelg2, 1, 1, out_stream, &out_cursor);
  assert(1 == num);
  num = stream_write(&threads, 1, 1, out_stream, &out_cursor);
  assert(1 == num);
  num = stream_write(&chunksize, 4, 1, out_stream, &out_cursor);
  assert(1 == num);

  // compute chunkrepeat
  chunkrepeat = BUFSIZE / (8 * threads * chunksize);
  if ((chunkrepeat * 8 * threads * chunksize) < BUFSIZE)
      chunkrepeat++;

  // allocate buffers
  inter = (unsigned char **)calloc(threads, sizeof(unsigned char *));
  assert(NULL != inter);
  input = (ull *)AlignedAlloc(BUFSIZE >> 3, 8);
  for (i = 0; i < threads; i++) {
      inter[i] = (unsigned char *)AlignedAlloc(8 + ((chunksize * chunkrepeat + 1) / 2) + (chunksize * chunkrepeat * 8) + 7, 1);
  }

  // allocate predictors
  predsizem1 = (1 << predsizelg2) - 1;
  fcm = (ull **)calloc(threads, sizeof(ull *));
  assert(NULL != fcm);
  dfcm = (ull **)calloc(threads, sizeof(ull *));
  assert(NULL != dfcm);
  for (i = 0; i < threads; i++) {
      fcm[i] = (ull *)AlignedAlloc(predsizem1 + 1, 8);
      dfcm[i] = (ull *)AlignedAlloc(predsizem1 + 1, 8);
  }

  // read in and compress data
  num = stream_read(input, 8, BUFSIZE >> 3, in_stream, in_bytes, &in_cursor);
  while (num > 0) {
      Compress(num, input, inter, predsizem1, threads, chunksize, fcm, dfcm, out_stream, &out_cursor);
      num = stream_read(input, 8, BUFSIZE >> 3, in_stream, in_bytes, &in_cursor);
  }

  // free buffers
  for (i = 0; i < threads; i++) {
      free(inter[i]);
  }
  free(inter);
  free(input);

  // free predictors
  for (i = 0; i < threads; i++) {
      free(fcm[i]);
      free(dfcm[i]);
  }
  free(fcm);
  free(dfcm);

  return out_cursor;
}

size_t pFPC_Decompress_Memory(const void *in_stream, size_t in_bytes, void *out_stream)
{
  size_t in_cursor = 0, out_cursor = 0;
  int predsizelg2, threads, chunksize;
  int i, num, chunkrepeat, predsizem1, val, cnt, in;
  register ull **fcm, **dfcm;
  register ull *output;
  register unsigned char **inter;

  // read in header
  predsizelg2 = 0;
  num = stream_read(&predsizelg2, 1, 1, in_stream, in_bytes, &in_cursor);
  assert(1 == num);
  threads = 0;
  num = stream_read(&threads, 1, 1, in_stream, in_bytes, &in_cursor);
  assert(1 == num);
  num = stream_read(&chunksize, 4, 1, in_stream, in_bytes, &in_cursor);
  assert(1 == num);

  // compute chunkrepeat
  chunkrepeat = BUFSIZE / (8 * threads * chunksize);
  if ((chunkrepeat * 8 * threads * chunksize) < BUFSIZE)
      chunkrepeat++;

  // allocate buffers
  inter = (unsigned char **)calloc(threads, sizeof(unsigned char *));
  assert(NULL != inter);
  for (i = 0; i < threads; i++) {
      inter[i] = (unsigned char *)AlignedAlloc(8 + ((chunksize * chunkrepeat + 1) / 2) + (chunksize * chunkrepeat * 8) + 7, 1);
  }
  output = (ull *)AlignedAlloc(BUFSIZE >> 3, 8);

  // allocate predictors
  predsizem1 = (1 << predsizelg2) - 1;
  fcm = (ull **)calloc(threads, sizeof(ull *));
  assert(NULL != fcm);
  dfcm = (ull **)calloc(threads, sizeof(ull *));
  assert(NULL != dfcm);
  for (i = 0; i < threads; i++) {
      fcm[i] = (ull *)AlignedAlloc(predsizem1 + 1, 8);
      dfcm[i] = (ull *)AlignedAlloc(predsizem1 + 1, 8);
  }

  // read in and decompress data
  cnt = stream_read(&num, 4, 1, in_stream, in_bytes, &in_cursor);
  while (cnt == 1) {
      for (i = 0; i < threads; i++) {
          cnt = stream_read(inter[i], 4, 1, in_stream, in_bytes, &in_cursor);
          assert(cnt == 1);
          in = *((int *)(inter[i])) - 4;
          cnt = stream_read(&(inter[i][4]), 1, in, in_stream, in_bytes, &in_cursor);
          assert(cnt == in);
      }
      Decompress(num, inter, output, predsizem1, threads, chunksize, fcm, dfcm, out_stream, &out_cursor);

      cnt = stream_read(&num, 4, 1, in_stream, in_bytes, &in_cursor);
  }

  // free buffers
  for (i = 0; i < threads; i++) {
      free(inter[i]);
  }
  free(inter);
  free(output);

  // free predictors
  for (i = 0; i < threads; i++) {
      free(fcm[i]);
      free(dfcm[i]);
  }
  free(fcm);
  free(dfcm);

  return out_cursor;
}


const char *pFPC_Version_String = "pFPC v1.0";

