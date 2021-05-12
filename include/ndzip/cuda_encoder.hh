#pragma once

#include "array.hh"

namespace ndzip {

template<typename T, unsigned Dims>
class cuda_encoder {
  public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    size_t compress(const slice<const data_type, dimensions> &item, void *stream,
            kernel_duration *out_kernel_duration = nullptr) const;

    size_t decompress(
            const void *raw_stream, size_t bytes, const slice<data_type, dimensions> &data,
            kernel_duration *out_kernel_duration = nullptr) const;
};

}  // namespace ndzip
