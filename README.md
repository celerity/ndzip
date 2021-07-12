# ndzip: A High-Throughput Parallel Lossless Compressor for Scientific Data

ndzip compresses and decompresses multidimensional univariate grids of single- and double-precision IEEE 754 floating-point data.
Its primary use case is speeding up distributed HPC applications by increasing effective interconnect bandwidth.

This repository includes submodules, so make sure to update them:

```
git submodule update --init --recursive
```

## Prerequisites

- CMake >= 3.15
- Clang >= 10.0.0
- Linux (tested on x86_64 and POWER9)
- Boost >= 1.66
- (optional) [Catch2](https://github.com/catchorg/Catch2) >= 2.13.3

### Additionaly, for enabling GPU support

- CUDA >= 11.0 (not officially compatible with Clang 10/11, but a lower version will optimize insufficiently!)
- A [fork of hipSYCL](https://github.com/fknorr/hipSYCL/tree/rt-profiling) which includes kernel profiling functionality 
- An Nvidia GPU of Compute Capability >= 6.1

## Building and installing

Recommended CMake options for optimal CPU performance

```
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
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
cmake -B build -DCMAKE_PREFIX_PATH='../hipSYCL-rt-profiling/lib/cmake' -DHIPSYCL_PLATFORM=cuda -DCMAKE_CUDA_ARCHITECTURES=75 -DHIPSYCL_GPU_ARCH=sm_75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(which clang++) -DCMAKE_CUDA_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__" -DCMAKE_CXX_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -march=native"
cmake --build build -j
```

Replace `sm_75` and `75` with the string matching your GPU's Compute Capability. The `-U__FLOAT128__` define is required
due to an [open bug in Clang](https://bugs.llvm.org/show_bug.cgi?id=47559).

### For benchmarking with NVCOMP

Unfortunately, Clang currently cannot build NVCOMP. A separate binary supporting NVCOMP benchmarks can be built using
NVCC:

```
cmake -B build-nvcc -DCMAKE_CUDA_ARCHITECTURES=75 -DHIPSYCL_GPU_ARCH=sm_75 -DCMAKE_BUILD_TYPE=Release
cmake --build build-nvcc -j
```

## Running a benchmark

Benchmarking requires a collection of datasets, described by a CSV file. Example CSV:

```
miranda.f32;float;1024 1024 1024
magrecon.f32;float;512 512 512
flow.f64;double;16 7680 10240
turbulence.f32;float;256 256 256
rsim.f32;float;2048 11509
rsim.f64;double;2048 11509
wave.f32;float;512 512 512
wave.f64;double;512 512 512
```

The first column is a file name that is searched relative to the CSV path.
The second can be either `float` or `double` to denote the data type.
The third column is a space-separated tuple describing the dataset size. It can have one to three components, where the first refers to the slowest dimension.

`.f32` / `.f64` files are interpreted as little-endian blocks of single/double-precision floating-point data.

Example invocation for comparing GPU compressors (see the output of `benchmark --help` for more):

```
./benchmark scidata.csv -r 5 -a ndzip-gpu mpc gfc cudpp-compress > benchmark-results.csv
```

Results are visualized and aggregated using the `plot_benchmark` script:

```
python3 src/benchmark/plot_benchmark.py benchmark-results.csv
```

## Finding an optimal thread block / group size for your GPU

GPU compressor performance can vary with the (statically determined) number for threads per block.

To specify your own block size instead of the default 256 for single-precision and 384 for double-precision, ndzip must be recompiled with the following flags:

```
-DCMAKE_CXX_FLAGS="(... other CXX flags) -DNDZIP_GPU_GROUP_SIZE=512"
```

## GPU Kernel run-time debugging

Set the environment variable `NDZIP_VERBOSE=1` for any binary to dump individual kernel runtimes.

```
NDZIP_VERBOSE=1 ./benchmark scidata.csv -r 5 -a ndzip-gpu
```

