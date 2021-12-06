# ndzip: A High-Throughput Parallel Lossless Compressor for Scientific Data

ndzip compresses and decompresses multidimensional univariate grids of single- and double-precision IEEE 754
floating-point data. We implement

- a single-threaded CPU compressor
- an OpenMP-backed multi-threaded compressor
- a SYCL-based GPU compressor (currently hipSYCL + NVIDIA only)
- a CUDA-based GPU compressor (experimental)

All variants generate and decode bit-identical compressed stream.

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
- An Nvidia GPU with Compute Capability >= 6.1
- For the SYCL version: [hipSYCL](https://github.com/illuhad/hipSYCL) >= `8756087f`

## Building

Make sure to set the right build type and enable the full instruction set of the target CPU architecture:

```sh
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
```

If unit tests and microbenchmarks should also be built, add

```sh
-DNDZIP_BUILD_TEST=YES
```

### For GPU support with SYCL

1. Build and install hipSYCL

```
git clone https://github.com/illuhad/hipSYCL
cd hipSYCL
cmake -B build -DCMAKE_INSTALL_PREFIX=../hipSYCL-install -DWITH_CUDA_BACKEND=YES -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j
```

2. Build ndzip with SYCL

```
cmake -B build -DCMAKE_PREFIX_PATH='../hipSYCL-install/lib/cmake' -DHIPSYCL_PLATFORM=cuda -DCMAKE_CUDA_ARCHITECTURES=75 -DHIPSYCL_GPU_ARCH=sm_75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -march=native"
cmake --build build -j
```

Replace `sm_75` and `75` with the string matching your GPU's Compute Capability. The `-U__FLOAT128__` define is required
due to an [open bug in Clang](https://bugs.llvm.org/show_bug.cgi?id=47559).

### For GPU support with CUDA (experimental)

a) Either build ndzip with CUDA + NVCC ...

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
cmake --build build -j
```

Replace `sm_75` and `75` with the string matching your GPU's Compute Capability.

b) ... or with CUDA + Clang

```
cmake -B build -DCMAKE_CUDA_COMPILER="$(which clang++)" -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -march=native"
cmake --build build -j
```

The `-U__FLOAT128__` define is required due to an [open bug in Clang](https://bugs.llvm.org/show_bug.cgi?id=47559).

## Compressing and decompressing files

```sh
build/compress -n <size> -i <uncompressed-file> -o <compressed-file> [-t float|double]
build/compress -d -n <size> -i <compressed-file> -o <decompressed-file> [-t float|double]
```

`<size>` are one to three arguments depending on the dimensionality of the input grid. In the multi-dimensional case,
the first number specifies the width of the slowest-iterating dimension.

By default, `compress` uses the single-threaded CPU compressor. Passing `-e cpu-mt` or `-e sycl` / `-e cuda` selects the
multi-threaded CPU compressor or the GPU compressor if available, respectively.

## Running unit tests

Only available if tests have been enabled during build.

```sh
build/encoder_test
build/sycl_bits_test  # only if built with SYCL support
build/sycl_ubench     # GPU microbenchmarks, only if built with SYCL support
build/cuda_bits_test  # only if built with CUDA support
```

## See also

- [Benchmarking ndzip](docs/benchmarking.md)

## References

If you are using ndzip as part of your research, we kindly ask you to cite

- Fabian Knorr, Peter Thoman, and Thomas Fahringer. "ndzip: A High-Throughput Parallel Lossless Compressor for
  Scientific Data". In _2021 Data Compression Conference (DCC)_, IEEE,
  2021. [[DOI]](https://doi.org/10.1109/DCC50243.2021.00018) [Preprint PDF](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf)

- Knorr, Fabian, Peter Thoman, and Thomas Fahringer. "ndzip-gpu: efficient lossless compression of scientific floating-point data on GPUs". In _SC'21: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_, ACM, 2021. [[DOI]](https://doi.org/10.1145/3458817.3476224) [[Preprint PDF]](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-gpu-efficient-lossless-compression-of-scientific-floating-point-data-on-gpus.pdf)
