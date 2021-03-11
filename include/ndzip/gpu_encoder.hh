#pragma once

#include "array.hh"

#include <memory>

namespace ndzip {

template<typename T, unsigned Dims>
class gpu_encoder {
  public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    gpu_encoder();
    ~gpu_encoder();
    gpu_encoder(gpu_encoder &&) noexcept = default;
    gpu_encoder &operator=(gpu_encoder &&) noexcept = default;

    size_t compress(const slice<const data_type, dimensions> &item, void *stream) const;

    size_t decompress(
            const void *raw_stream, size_t bytes, const slice<data_type, dimensions> &data) const;

  private:
    struct impl;
    std::unique_ptr<impl> _pimpl;
};

}  // namespace ndzip
