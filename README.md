# ndzip: A High-Throughput Parallel Lossless Compressor for Scientific Data

ndzip compresses and decompresses multidimensional univariate grids of single- and double-precision IEEE 754
floating-point data. We implement a single-threaded CPU compressor, an OpenMP-backed multi-threaded compressor and a
SYCL-based GPU compressor. All three variants generate and decode bit-identical compressed stream.

ndzip is currently a research project with the primary use case of speeding up distributed HPC applications by
increasing effective interconnect bandwidth.

## Prerequisites

- CMake >= 3.15
- Clang >= 10.0.0
- Linux (tested on x86_64 and POWER9)
- Boost >= 1.66
- [Catch2](https://github.com/catchorg/Catch2) >= 2.13.3 (optional, for unit tests and microbenchmarks)

### Additionaly, for GPU support

- CUDA >= 11.0 (not officially compatible with Clang 10/11, but a lower version will optimize insufficiently!)
- A [fork of hipSYCL](https://github.com/fknorr/hipSYCL/tree/rt-profiling) which includes kernel profiling functionality
- An Nvidia GPU of Compute Capability >= 6.1

## Building

Make sure to set the right build type and enable the full instruction set of the target CPU architecture:

```sh
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
```

If unit tests and microbenchmarks should also be built, add

```sh
-DNDZIP_BUILD_TEST=YES
```

### For GPU support

1. Build and install the required [custom hipSYCL](https://github.com/fknorr/hipSYCL/tree/rt-profiling) version

```
git clone https://github.com/fknorr/hipSYCL --branch rt-profiling
cd hipSYCL
cmake -B build -DCMAKE_INSTALL_PREFIX=../hipSYCL-rt-profiling -DWITH_CUDA_BACKEND=YES -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j
```

2. Build ndzip with GPU support

```
cmake -B build -DCMAKE_PREFIX_PATH='../hipSYCL-rt-profiling/lib/cmake' -DHIPSYCL_PLATFORM=cuda -DCMAKE_CUDA_ARCHITECTURES=75 -DHIPSYCL_GPU_ARCH=sm_75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -march=native"
cmake --build build -j
```

Replace `sm_75` and `75` with the string matching your GPU's Compute Capability. The `-U__FLOAT128__` define is required
due to an [open bug in Clang](https://bugs.llvm.org/show_bug.cgi?id=47559).

## Compressing and decompressing files

```sh
build/compress -n <size> -i <uncompressed-file> -o <compressed-file> [-t float|double]
build/compress -d -n <size> -i <compressed-file> -o <decompressed-file> [-t float|double]
```

`<size>` are one to three arguments depending on the dimensionality of the input grid. In the multi-dimensional case,
the first number specifies the width of the slowest-iterating dimension.

By default, `compress` uses the single-threaded CPU compressor. Passing `-e cpu-mt` or `-e gpu` selects the
multi-threaded CPU compressor or the GPU compressor if available, respectively.

## Running unit tests

Only available if tests have been enabled during build.

```sh
build/encoder_test
build/gpu_bits_test  # only if built with GPU support
build/gpu_ubench     # GPU microbenchmarks, only if built with GPU support
```

## See also

- [Benchmarking ndzip](docs/benchmarking.md)

## References

If you are using ndzip as part of your research, we kindly ask you to cite

- Fabian Knorr, Peter Thoman, and Thomas Fahringer. "ndzip: A High-Throughput Parallel Lossless Compressor for
  Scientific Data". In _2021 Data Compression Conference (DCC)_, IEEE,
  2021. [[DOI]](https://doi.org/10.1109/DCC50243.2021.00018) [Preprint PDF](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf)

- Fabian Knorr, Peter Thoman, and Thomas Fahringer. "ndzip-gpu: Efficient Lossless Compression of Scientific
  Floating-Point Data on GPUs". Accepted at Supercomputing
  2021. [Preprint PDF](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-gpu-efficient-lossless-compression-of-scientific-floating-point-data-on-gpus.pdf)
