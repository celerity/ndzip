#pragma once

#include "common.hh"

#include <vector>


namespace hcde::detail {

template<typename T, size_t Align>
class aligned_buffer {
    static_assert(Align >= alignof(T) && Align % alignof(T) == 0);

    public:
        explicit aligned_buffer(size_t size) {
            assert(size % Align == 0);
            _memory = std::aligned_alloc(Align, size * sizeof(T));
            if (!_memory) {
                throw std::bad_alloc();
            }
        }

        aligned_buffer(const aligned_buffer &) = delete;
        aligned_buffer &operator=(const aligned_buffer &) = delete;

        ~aligned_buffer() {
            std::free(_memory);
        }

        T *data() {
            return static_cast<T*>(__builtin_assume_aligned(_memory, Align));
        }

        const T *data() const {
            return static_cast<const T*>(__builtin_assume_aligned(_memory, Align));
        }

    private:
        void *_memory;
};

template<typename T>
using simd_aligned_buffer = aligned_buffer<T, 32>;

template<typename profile>
[[gnu::noinline]]
void load_hypercube(const hypercube<profile> &hc,
    const slice<const typename profile::data_type, profile::dimensions> &data,
    detail::simd_aligned_buffer<typename profile::bits_type> &cube) {
    using bits_type = typename profile::bits_type;

    bits_type *cube_pos = cube.data();
    hc.for_each_cell(data, [&](auto *cell) {
        *cube_pos++ = load_value<profile>(cell);
    });
}

}


template<typename T, unsigned Dims>
size_t hcde::cpu_encoder<T, Dims>::compressed_size_bound(const extent<dimensions> &size) const {
    using profile = detail::profile<T, Dims>;
    detail::file<profile> file(size);
    size_t bound = file.file_header_length();
    bound += file.num_hypercubes() * profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename T, unsigned Dims>
size_t hcde::cpu_encoder<T, Dims>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr static auto side_length = profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, profile::dimensions);

    detail::file<profile> file(data.size());
    size_t stream_pos = file.file_header_length();

    detail::simd_aligned_buffer<bits_type> cube(hc_size);
    file.for_each_hypercube([&](auto hc, auto hc_index) {
        detail::load_hypercube<profile>(hc, data, cube);

        bits_type *cube_pos = cube.data();
        auto header_pos = stream_pos;
        stream_pos += detail::encode_block<profile>(cube_pos, static_cast<char *>(stream) + stream_pos);

        if (hc_index > 0) {
            auto file_offset_address = static_cast<char *>(stream)
                    + (hc_index - 1) * sizeof(detail::file_offset_type);
            auto file_offset = static_cast<detail::file_offset_type>(header_pos);
            detail::store_unaligned(file_offset_address, detail::endian_transform(file_offset));
        }
    });

    if (file.num_hypercubes() > 0) {
        auto border_offset_address = static_cast<char *>(stream) + (file.num_hypercubes() - 1) * sizeof(detail::file_offset_type);
        detail::store_unaligned(border_offset_address, detail::endian_transform(stream_pos));
    }
    stream_pos += detail::pack_border(static_cast<char *>(stream) + stream_pos, data, side_length);
    return stream_pos;
}


template<typename T, unsigned Dims>
size_t hcde::cpu_encoder<T, Dims>::decompress(const void *stream, size_t bytes,
    const slice<data_type, dimensions> &data) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    constexpr static auto side_length = profile::hypercube_side_length;
    detail::file<profile> file(data.size());

    size_t stream_pos = file.file_header_length(); // simply skip the header
    file.for_each_hypercube([&](auto hc) {
        bits_type cube[detail::ipow(side_length, profile::dimensions)] = {};
        stream_pos += detail::decode_block<profile>(static_cast<const char *>(stream) + stream_pos, cube);
        bits_type *cube_ptr = cube;
        hc.for_each_cell(data, [&](auto *cell) { detail::store_value<profile>(cell, *cube_ptr++); });
    });
    stream_pos += detail::unpack_border(data, static_cast<const char *>(stream) + stream_pos,
        side_length);
    return stream_pos;
}


namespace hcde {
    extern template class cpu_encoder<float, 1>;
    extern template class cpu_encoder<float, 2>;
    extern template class cpu_encoder<float, 3>;
    extern template class cpu_encoder<double, 1>;
    extern template class cpu_encoder<double, 2>;
    extern template class cpu_encoder<double, 3>;
}
