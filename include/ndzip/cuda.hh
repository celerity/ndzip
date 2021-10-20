#pragma once

#include "array.hh"

#include <cuda_runtime.h>
#include <memory>

namespace ndzip::cuda {

template<typename T, unsigned Dims>
struct compressor_scratch_memory;

template<typename T, unsigned Dims>
std::unique_ptr<compressor_scratch_memory<T, Dims>> allocate_compressor_scratch_memory(extent<Dims> data_size);

template<typename T, unsigned Dims>
void compress_async(slice<const T, Dims> in_device_data, void *out_device_stream, size_t *out_device_stream_length,
        compressor_scratch_memory<T, Dims> &scratch, cudaStream_t stream = 0);

template<typename T, unsigned Dims>
void decompress_async(const void *in_device_stream, slice<T, Dims> out_device_data, cudaStream_t stream = 0);

}  // namespace ndzip::cuda


namespace std {

template<typename T, unsigned Dims>
struct default_delete<ndzip::cuda::compressor_scratch_memory<T, Dims>> {
    void operator()(ndzip::cuda::compressor_scratch_memory<T, Dims> *m) const;
};

}  // namespace std
