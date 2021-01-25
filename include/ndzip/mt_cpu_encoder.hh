#pragma once

#include "array.hh"

#include <memory>

namespace ndzip {

template<typename T, unsigned Dims>
class mt_cpu_encoder {
  public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    mt_cpu_encoder();

    explicit mt_cpu_encoder(size_t num_threads);

    mt_cpu_encoder(mt_cpu_encoder &&) noexcept = default;

    ~mt_cpu_encoder();

    mt_cpu_encoder &operator=(mt_cpu_encoder &&) noexcept = default;

    size_t compress(const slice<const data_type, dimensions> &data, void *stream) const;

    size_t decompress(
            const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const;

  private:
    struct impl;
    std::unique_ptr<impl> _pimpl;
};

}  // namespace ndzip
