#pragma once

#include "common.hh"

#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#   include <immintrin.h>
#endif


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

        T &operator[](size_t i) {
            return data()[i];
        };

        const T &operator[](size_t i) const {
            return data()[i];
        };

    private:
        void *_memory;
};

constexpr static const size_t simd_width = 32;

template<typename T>
using simd_aligned_buffer = aligned_buffer<T, simd_width>;

template<typename profile>
[[gnu::noinline]]
void load_hypercube(const hypercube<profile> &hc,
    const slice<const typename profile::data_type, profile::dimensions> &data,
    detail::simd_aligned_buffer<typename profile::bits_type> &cube) {

}

template<typename T>
[[gnu::always_inline]]
void transpose_bits_trivial(const T *__restrict vs, T *__restrict out) {
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        out[i] = 0;
        for (unsigned j = 0; j < bitsof<T>; ++j) {
            out[i] |= ((vs[j] >> (bitsof<T>-1-i)) & 1u) << (bitsof<T>-1-j);
        }
    }
    if (sizeof(T) == 4) {
        puts("out:");
        for (unsigned i = 0; i < 32; ++i) {
            printf("%08x ", out[i]);
            if (i % 8 == 7)
                puts("");
        }
        puts("");
    }
}

#ifdef __AVX2__

[[gnu::always_inline]]
inline void transpose_bits_32_avx2(const uint32_t *__restrict vs, uint32_t *__restrict out) {
    __m256i unpck0[4];
    __builtin_memcpy(unpck0, vs, sizeof unpck0);

    // 1. In order to generate 32 transpositions using only 32 *movemask* instructions,
    // first shuffle bytes so that the i-th 256-bit vector contains the i-th byte of
    // each float

    __m256i shuf[4];
    {
        // interpret each 128-bit lane as a 4x4 matrix and transpose it
        // also correct endianess for later
        uint8_t idx8[] = {
                12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3,
                12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3,
        };
        __m256i idx;
        __builtin_memcpy(&idx, idx8, sizeof idx);
        for (unsigned i = 0; i < 4; ++i) {
            shuf[i] = _mm256_shuffle_epi8(unpck0[i], idx);
        }
    }

    __m256i perm[4];
    {
        // interleave doublewords within each 256-bit vector
        // quadword i will contain elements {i, i+4, i+8, ... i+28}
        // uint32_t idx32[] = {0, 4, 1, 5, 2, 6, 3, 7};
        uint32_t idx32[] = {7, 3, 6, 2, 5, 1, 4, 0};
        __m256i idx;
        __builtin_memcpy(&idx, idx32, sizeof idx);
        for (unsigned i = 0; i < 4; ++i) {
            perm[i] = _mm256_permutevar8x32_epi32(shuf[i], idx);
        }
    }

    __m256i unpck1[4];
    for (unsigned i = 0; i < 4; i += 2) {
        // interleave quadwords of neighboring 256-bit vectors
        // each double-quadword will contain elements with stride 4
        unpck1[i+1] = _mm256_unpackhi_epi64(perm[i+1], perm[i+0]);
        unpck1[i+0] = _mm256_unpacklo_epi64(perm[i+1], perm[i+0]);
    }

    __m256i perm2[4] = {
        // combine matching 128-bit lanes
        _mm256_permute2x128_si256(unpck1[2], unpck1[0], 0x20),
        _mm256_permute2x128_si256(unpck1[3], unpck1[1], 0x20),
        _mm256_permute2x128_si256(unpck1[2], unpck1[0], 0x31),
        _mm256_permute2x128_si256(unpck1[3], unpck1[1], 0x31),
    };

    // 2. Transpose by extracting the 32 MSBs of each byte of each 256-byte vector

    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            out[i*8+j] = _mm256_movemask_epi8(perm2[i]);
            perm2[i] = _mm256_slli_epi32(perm2[i], 1);
        }
    }
}

#endif // __AVX2__

template<typename T>
void transpose_bits(const T *__restrict in, T *__restrict out) {
#ifdef __AVX2__
    if constexpr(sizeof(T) == 4) {
        transpose_bits_32_avx2(in, out);
    } else
#endif
    {
        return transpose_bits_trivial(in, out);
    }
}

template<typename T>
[[gnu::always_inline]]
size_t compact_zero_words(const T *shifted, std::byte *out0) {
    auto out = out0 + sizeof(T);
    T head = 0;
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        if (shifted[i] != 0) {
            head |= 1u << i;
            store_aligned(out, shifted[i]);
            out += sizeof(T);
        }
    }
    store_aligned(out0, head);
    return out - out0;
}

template<typename T>
[[gnu::always_inline]]
size_t expand_zero_words(const std::byte *in0, T *shifted) {
    auto in = in0 + sizeof(T);
    auto head = load_aligned<T>(in0);
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        if ((head >> i) & 1u) {
            shifted[i] = load_aligned<T>(in);
            in += sizeof(T);
        } else {
            shifted[i] = 0;
        }
    }
    return in - in0;
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

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    detail::file<profile> file(data.size());
    size_t stream_pos = file.file_header_length();

    detail::simd_aligned_buffer<bits_type> cube(hc_size);
    file.for_each_hypercube([&](auto hc, auto hc_index) {
        auto header_pos = stream_pos;

        hc.for_each_cell(data, [&](auto *cell, size_t i) {
            memcpy(&cube[i], cell, sizeof(T));
        });

        detail::block_transform(cube.data(), Dims, profile::hypercube_side_length);

        for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
            bits_type transposed[detail::bitsof<bits_type>];
            detail::transpose_bits(cube.data() + i, transposed);
            stream_pos += detail::compact_zero_words(transposed, static_cast<std::byte *>(stream) + stream_pos);
        }

        if (hc_index > 0) {
            auto file_offset_address = static_cast<std::byte *>(stream)
                    + (hc_index - 1) * sizeof(detail::file_offset_type);
            auto file_offset = static_cast<detail::file_offset_type>(header_pos);
            detail::store_aligned(file_offset_address, detail::endian_transform(file_offset));
        }
    });

    if (file.num_hypercubes() > 0) {
        auto file_offset_address
                = static_cast<char *>(stream) + (file.num_hypercubes() - 1) * sizeof(detail::file_offset_type);
        detail::store_aligned(file_offset_address, detail::endian_transform(stream_pos));
    }
    stream_pos += detail::pack_border(static_cast<char *>(stream) + stream_pos, data, side_length);
    return stream_pos;
}


template<typename T, unsigned Dims>
size_t hcde::cpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr static auto side_length = profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, profile::dimensions);

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    detail::file<profile> file(data.size());

    detail::simd_aligned_buffer<bits_type> cube(hc_size);
    size_t stream_pos = file.file_header_length(); // simply skip the header
    file.for_each_hypercube([&](auto hc) {
        for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
            bits_type transposed[detail::bitsof<bits_type>];
            stream_pos += detail::expand_zero_words(static_cast<const std::byte*>(stream) + stream_pos, transposed);
            detail::transpose_bits(transposed, cube.data() + i);
        }

        detail::inverse_block_transform(cube.data(), Dims, profile::hypercube_side_length);

        hc.for_each_cell(data, [&](auto *cell, size_t i) {
            memcpy(cell, &cube[i], sizeof(T));
        });
    });
    stream_pos += detail::unpack_border(data, static_cast<const std::byte *>(stream) + stream_pos, side_length);
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
