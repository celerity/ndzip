# High-Throughput Scientific Data Compressor

ndzip leverages SIMD- and thread parallelism to compress and decompress multidimensional univariate grids of
single- and double-precision IEEE 754 floating-point data at speeds close to memory bandwidth.
Its primary use case is speeding up distributed HPC applications by increasing effective interconnect and bus bandwidth.

This is still experimental, with APIs not suitable for real-world use yet. The compressed file format is not stable and
*will* change in the future.

ndzip performance can be tested on your machine using the included benchmark. Perform a git submodule update and
install the system packages for lz4, xzutils, zlib and zstd to measure against all competitor algorithms

The optimized implementation of ndzip requires AVX2, make sure to compile with `CMAKE_CXX_FLAGS="-march=native"`.
