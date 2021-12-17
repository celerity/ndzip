#pragma once

#include "ndzip.hh"


namespace ndzip {

template<typename T>
class offloader {
  public:
    using data_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~offloader() = default;

    index_type compress(const slice<const data_type, dynamic_extent> &data, compressed_type *stream,
            kernel_duration *duration = nullptr) {
        return do_compress(data, stream, duration);
    }

    index_type decompress(const compressed_type *stream, index_type length,
            const slice<data_type, dynamic_extent> &data, kernel_duration *duration = nullptr) {
        return do_decompress(stream, length, data, duration);
    }

  protected:
    virtual index_type
    do_compress(const slice<const data_type, dynamic_extent> &data, compressed_type *stream, kernel_duration *duration)
            = 0;

    virtual index_type do_decompress(const compressed_type *stream, index_type length,
            const slice<data_type, dynamic_extent> &data, kernel_duration *duration)
            = 0;
};

template<typename T, dim_type Dims>
class cpu_offloader final : public offloader<T> {
  public:
    using data_type = T;
    using compressed_type = detail::bits_type<T>;
    constexpr static dim_type dimensions = Dims;

    cpu_offloader() = default;

    explicit cpu_offloader(index_type num_threads) : _co{num_threads}, _de{num_threads} {}

  protected:
    index_type do_compress(const slice<const data_type, dynamic_extent> &data, compressed_type *stream,
            kernel_duration *duration) override {
        // TODO duration
        return _co.compress(slice<const data_type, extent<Dims>>{data}, stream);
    }

    index_type do_decompress(
            const compressed_type *stream, [[maybe_unused]] index_type, const slice<data_type, dynamic_extent> &data,
            kernel_duration *duration) override {
        // TODO duration
        return _de.decompress(stream, slice<data_type, extent<Dims>>{data});
    }

  private:
    compressor<T, Dims> _co;
    decompressor<T, Dims> _de;
};

#if NDZIP_HIPSYCL_SUPPORT

template<typename T, dim_type Dims>
class sycl_offloader final : public offloader<T> {
  public:
    using data_type = T;
    using compressed_type = detail::bits_type<T>;
    constexpr static dim_type dimensions = Dims;

    explicit sycl_offloader(bool enable_profiling = false);
    ~sycl_offloader();
    sycl_offloader(sycl_offloader &&) noexcept = default;
    sycl_offloader &operator=(sycl_offloader &&) noexcept = default;

  protected:
    index_type do_compress(const slice<const data_type, dynamic_extent> &data, compressed_type *stream,
            kernel_duration *duration) override;

    index_type do_decompress(const compressed_type *stream, index_type length,
            const slice<data_type, dynamic_extent> &data, kernel_duration *duration) override;

  private:
    struct impl;
    std::unique_ptr<impl> _pimpl;
};

#endif  // NDZIP_HIPSYCL_SUPPORT

#if NDZIP_CUDA_SUPPORT

template<typename T, dim_type Dims>
class cuda_offloader final : public offloader<T> {
  public:
    using data_type = T;
    using compressed_type = detail::bits_type<T>;
    constexpr static dim_type dimensions = Dims;

  protected:
    index_type do_compress(const slice<const data_type, dynamic_extent> &data, compressed_type *stream,
            kernel_duration *duration) override;

    index_type do_decompress(const compressed_type *stream, index_type length,
            const slice<data_type, dynamic_extent> &data, kernel_duration *duration) override;
};

#endif  // NDZIP_CUDA_SUPPORT

}  // namespace ndzip

namespace ndzip::detail {
template<template<typename, dim_type> typename TargetOffloader, typename T, typename ...CtorParams>
std::unique_ptr<offloader<T>> make_target_offloader(dim_type dimensions, CtorParams ...args) {
    switch (dimensions) {
        case 1: return std::make_unique<TargetOffloader<T, 1>>(args...);
        case 2: return std::make_unique<TargetOffloader<T, 2>>(args...);
        case 3: return std::make_unique<TargetOffloader<T, 3>>(args...);
        default: throw std::runtime_error("ndzip::make_offloader: invalid dimensionality");
    }
}
}  // namespace ndzip::detail

namespace ndzip {

template<typename T>
std::unique_ptr<offloader<T>> make_cpu_offloader(dim_type dimensions) {
    return detail::make_target_offloader<cpu_offloader, T>(dimensions);
}

template<typename T>
std::unique_ptr<offloader<T>> make_cpu_offloader(dim_type dimensions, size_t num_threads) {
    return detail::make_target_offloader<cpu_offloader, T>(dimensions, num_threads);
}

#if NDZIP_HIPSYCL_SUPPORT
template<typename T>
std::unique_ptr<offloader<T>> make_sycl_offloader(dim_type dimensions, bool enable_profiling = false) {
    return detail::make_target_offloader<sycl_offloader, T>(dimensions, enable_profiling);
}
#endif

#if NDZIP_CUDA_SUPPORT
template<typename T>
std::unique_ptr<offloader<T>> make_cuda_offloader(dim_type dimensions) {
    return detail::make_target_offloader<cuda_offloader, T>(dimensions);
}
#endif

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
std::unique_ptr<offloader<T>> make_offloader(dim_type dimensions, target target, bool enable_profiling = false) {
    switch (target) {
        case target::cpu: return detail::make_target_offloader<cpu_offloader, T>(dimensions);
#if NDZIP_HIPSYCL_SUPPORT
        case target::sycl: return detail::make_target_offloader<sycl_offloader, T>(dimensions, enable_profiling);
#endif
#if NDZIP_CUDA_SUPPORT
        case target::cuda: return detail::make_target_offloader<cuda_offloader, T>(dimensions);
#endif
        default: throw std::runtime_error("ndzip::make_offloader: invalid target");
    }
}

}  // namespace ndzip
