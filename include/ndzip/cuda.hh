#pragma once

#include "ndzip.hh"

#include <cuda_runtime.h>


namespace ndzip {

template<typename T>
class cuda_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~cuda_compressor() = default;

    virtual void compress(const T *in_device_data, const extent &data_size, compressed_type *out_device_stream,
            index_type *out_device_stream_length)
            = 0;
};

template<typename T>
class cuda_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~cuda_decompressor() = default;

    virtual void decompress(const compressed_type *in_device_stream, T *out_device_data, const extent &data_size) = 0;
};

template<typename T>
std::unique_ptr<cuda_compressor<T>>
make_cuda_compressor(const compressor_requirements &req, cudaStream_t stream = nullptr);

template<typename T>
std::unique_ptr<cuda_decompressor<T>> make_cuda_decompressor(dim_type dims, cudaStream_t stream = nullptr);

}  // namespace ndzip
