#pragma once

#include "ndzip.hh"

namespace ndzip {

template<typename T, dim_type Dims>
class cuda_encoder {
  public:
    using data_type = T;
    constexpr static dim_type dimensions = Dims;

    size_t compress(const slice<const data_type, extent<dimensions>> &item, void *stream,
            kernel_duration *out_kernel_duration = nullptr) const;

    size_t decompress(const void *raw_stream, size_t bytes, const slice<data_type, extent<dimensions>> &data,
            kernel_duration *out_kernel_duration = nullptr) const;
};

}  // namespace ndzip
