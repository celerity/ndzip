#pragma once

#include "ndzip.hh"


namespace ndzip {

template<typename T, unsigned Dims>
class cpu_encoder {
  public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    cpu_encoder() = default;

    explicit cpu_encoder(size_t num_threads): _co{num_threads}, _de{num_threads} {}

    size_t compress(const slice<const data_type, extent<dimensions>> &data, void *stream) {
        return _co.compress(data, static_cast<detail::bits_type<T>*>(stream));
    }

    size_t decompress(const void *stream, size_t bytes, const slice<data_type, extent<dimensions>> &data) {
        return _de.decompress(static_cast<const detail::bits_type<T>*>(stream), data);
    }

  private:
    compressor<T, Dims> _co;
    decompressor<T, Dims> _de;
};

}  // namespace ndzip
