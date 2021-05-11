#pragma once

#include "array.hh"

#include <chrono>
#include <memory>

namespace ndzip {

using kernel_duration = std::chrono::duration<uint64_t, std::nano>;

template<typename T, unsigned Dims>
class sycl_encoder {
  public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    explicit sycl_encoder(bool report_kernel_duration = false);
    ~sycl_encoder();
    sycl_encoder(sycl_encoder &&) noexcept = default;
    sycl_encoder &operator=(sycl_encoder &&) noexcept = default;

    size_t compress(const slice<const data_type, dimensions> &item, void *stream,
            kernel_duration *out_kernel_duration = nullptr) const;

    size_t decompress(
            const void *raw_stream, size_t bytes, const slice<data_type, dimensions> &data,
            kernel_duration *out_kernel_duration = nullptr) const;

  private:
    struct impl;
    std::unique_ptr<impl> _pimpl;
};

}  // namespace ndzip
