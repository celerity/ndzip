#pragma once

#include "array.hh"

#include <cuda_runtime.h>
#include <memory>

namespace ndzip {

template<int Dims>
class cuda_compressor_requirements {
  public:
    cuda_compressor_requirements() = default;

    cuda_compressor_requirements(extent<Dims> single_data_size) {  // NOLINT(google-explicit-constructor)
        include(single_data_size);
    }

    cuda_compressor_requirements(std::initializer_list<extent<Dims>> data_sizes) {
        for (auto ds : data_sizes) {
            include(ds);
        }
    }

    void include(extent<Dims> data_size);

  private:
    template<typename, int>
    friend class cuda_compressor;

    index_type _max_num_hypercubes = 0;
};

template<typename T, int Dims>
class cuda_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    explicit cuda_compressor(cuda_compressor_requirements<Dims> reqs) : cuda_compressor{nullptr, reqs} {}

    explicit cuda_compressor(cudaStream_t stream, cuda_compressor_requirements<Dims> reqs);

    cuda_compressor(cuda_compressor &&) noexcept = default;

    ~cuda_compressor();

    cuda_compressor &operator=(cuda_compressor &&) noexcept = default;

    void compress(
            slice<const T, Dims> in_device_data, compressed_type *out_device_stream, size_t *out_device_stream_length);

  private:
    template<typename, unsigned>
    friend class cuda_encoder;

    struct scratch_buffers;
    cudaStream_t _stream = nullptr;
    std::unique_ptr<scratch_buffers> _scratch;
};

template<typename T, int Dims>
class cuda_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    cuda_decompressor() = default;

    explicit cuda_decompressor(cudaStream_t stream);

    void decompress(const compressed_type *in_device_stream, slice<T, Dims> out_device_data);

  private:
    cudaStream_t _stream = nullptr;
};

}  // namespace ndzip
