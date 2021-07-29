# Benchmarking ndzip against third-party compressors

Some third-party compressors are included as submodules, so make sure to update them:

```sh
git submodule update --init --recursive

## Building
```

Benchmarks need to be enabled in the build explicitly. Also, make sure to set the right build type and enable the full instruction set of the target CPU architecture:

```sh
cmake -DNDZIP_ENABLE_BENCHMARK=YES -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" [...]
```

Building GPU benchmarks also requires setting `CMAKE_CUDA_ARCHITECTURES`.

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
