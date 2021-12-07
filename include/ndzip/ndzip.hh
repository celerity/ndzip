#pragma once

#include "array.hh"

#include <memory>

namespace ndzip {

template<typename T, int Dims>
class compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    compressor();

    explicit compressor(size_t num_threads);

    size_t compress(const slice<const value_type, Dims> &data, compressed_type *stream) {
        return _pimpl->compress(data, stream);
    }

  private:
    struct impl {
        virtual ~impl() = default;
        virtual size_t compress(const slice<const value_type, Dims> &data, compressed_type *stream) = 0;
    };
    struct st_impl;
    struct mt_impl;

    std::unique_ptr<impl> _pimpl;
};

template<typename T, int Dims>
class decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    decompressor();

    explicit decompressor(size_t num_threads);

    size_t decompress(const compressed_type *stream, const slice<value_type, Dims> &data) {
        return _pimpl->decompress(stream, data);
    }

  private:
    struct impl {
        virtual ~impl() = default;
        virtual size_t decompress(const compressed_type *stream, const slice<value_type, Dims> &data) = 0;
    };
    struct st_impl;
    struct mt_impl;

    std::unique_ptr<impl> _pimpl;
};

}  // namespace ndzip