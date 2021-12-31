#pragma once

#include "ndzip.hh"


namespace ndzip {

template<typename T>
class offloader {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~offloader() = default;

    index_type compress(const value_type *data, const extent &data_size, compressed_type *stream,
            kernel_duration *duration = nullptr) {
        return do_compress(data, data_size, stream, duration);
    }

    index_type decompress(const compressed_type *stream, index_type length, value_type *data, const extent &data_size,
            kernel_duration *duration = nullptr) {
        return do_decompress(stream, length, data, data_size, duration);
    }

  protected:
    virtual index_type
    do_compress(const value_type *data, const extent &data_size, compressed_type *stream, kernel_duration *duration)
            = 0;

    virtual index_type do_decompress(const compressed_type *stream, index_type length, value_type *data,
            const extent &data_size, kernel_duration *duration)
            = 0;
};

enum class target {
    cpu,
#if NDZIP_HIPSYCL_SUPPORT
    sycl,
#endif
#if NDZIP_CUDA_SUPPORT
    cuda,
#endif
};

template<typename T>
std::unique_ptr<offloader<T>> make_cpu_offloader(dim_type dims, unsigned num_threads = 0);

#if NDZIP_HIPSYCL_SUPPORT
template<typename T>
std::unique_ptr<offloader<T>> make_sycl_offloader(dim_type dimensions, bool enable_profiling = false);
#endif

#if NDZIP_CUDA_SUPPORT
template<typename T>
std::unique_ptr<offloader<T>> make_cuda_offloader(dim_type dimensions);
#endif

template<typename T>
std::unique_ptr<offloader<T>> make_offloader(target target, dim_type dimensions, bool enable_profiling = false) {
    switch (target) {
        case target::cpu: return make_cpu_offloader<T>(dimensions);
#if NDZIP_HIPSYCL_SUPPORT
        case target::sycl: return make_sycl_offloader<T>(dimensions, enable_profiling);
#endif
#if NDZIP_CUDA_SUPPORT
        case target::cuda: return make_cuda_offloader<T>(dimensions);
#endif
        default: throw std::runtime_error("ndzip::make_offloader: invalid target");
    }
}

}  // namespace ndzip
