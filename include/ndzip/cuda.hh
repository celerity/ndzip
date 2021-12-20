#pragma once

#include "ndzip.hh"

#include <cuda_runtime.h>


namespace ndzip {

template<typename T>
class basic_cuda_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_cuda_compressor() = default;

    virtual void compress(const T *in_device_data, const dynamic_extent &data_size, compressed_type *out_device_stream,
            index_type *out_device_stream_length)
            = 0;
};

template<typename T, int Dims>
class cuda_compressor final : public basic_cuda_compressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    explicit cuda_compressor(compressor_requirements<Dims> reqs) : cuda_compressor{nullptr, reqs} {}

    explicit cuda_compressor(cudaStream_t stream, compressor_requirements<Dims> reqs);

    cuda_compressor(cuda_compressor &&) noexcept = default;

    ~cuda_compressor();

    cuda_compressor &operator=(cuda_compressor &&) noexcept = default;

    void compress(const T *in_device_data, const extent<Dims> &data_size, compressed_type *out_device_stream,
            index_type *out_device_stream_length);

    virtual void compress(const T *in_device_data, const dynamic_extent &data_size, compressed_type *out_device_stream,
            index_type *out_device_stream_length) override {
        compress(in_device_data, extent<Dims>{data_size}, out_device_stream, out_device_stream_length);
    }

  private:
    template<typename, dim_type>
    friend class cuda_offloader;

    struct scratch_buffers;
    cudaStream_t _stream = nullptr;
    std::unique_ptr<scratch_buffers> _scratch;
};

template<typename T>
class basic_cuda_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_cuda_decompressor() = default;

    virtual void
    decompress(const compressed_type *in_device_stream, T *out_device_data, const dynamic_extent &data_size)
            = 0;

  private:
    cudaStream_t _stream = nullptr;
};

template<typename T, int Dims>
class cuda_decompressor final : public basic_cuda_decompressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    cuda_decompressor() = default;

    explicit cuda_decompressor(cudaStream_t stream) : _stream(stream) {}

    void decompress(const compressed_type *in_device_stream, T *out_device_data, const extent<Dims> &data_size);

    void decompress(
            const compressed_type *in_device_stream, T *out_device_data, const dynamic_extent &data_size) override {
        decompress(in_device_stream, out_device_data, extent<Dims>{data_size});
    }

  private:
    cudaStream_t _stream = nullptr;
};

}  // namespace ndzip
