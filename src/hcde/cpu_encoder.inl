#pragma once

#include "common.hh"

#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#   include <immintrin.h>
#endif


namespace hcde::detail::cpu {

constexpr static const size_t simd_width_bytes = 32;

template<typename T>
[[gnu::always_inline]] T *assume_simd_aligned(T *x) {
    assert(reinterpret_cast<uintptr_t>(x) % simd_width_bytes == 0);
    return static_cast<T *>(__builtin_assume_aligned(x, simd_width_bytes));
}

template<typename T>
class simd_aligned_buffer {
    static_assert(simd_width_bytes >= alignof(T) && simd_width_bytes % alignof(T) == 0);

    public:
        explicit simd_aligned_buffer(size_t size) {
            assert(size % simd_width_bytes == 0);
            _memory = std::aligned_alloc(simd_width_bytes, size * sizeof(T));
            if (!_memory) {
                throw std::bad_alloc();
            }
        }

        simd_aligned_buffer(const simd_aligned_buffer &) = delete;
        simd_aligned_buffer &operator=(const simd_aligned_buffer &) = delete;

        ~simd_aligned_buffer() {
            std::free(_memory);
        }

        T *data() {
            return assume_simd_aligned(static_cast<T*>(_memory));
        }

        const T *data() const {
            return assume_simd_aligned(static_cast<const T*>(_memory));
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

template<typename Profile>
[[gnu::noinline]]
void load_hypercube(const hypercube<Profile> &hc,
                    const slice<const typename Profile::data_type, Profile::dimensions> &data,
                    typename Profile::bits_type *cube)
{
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    map_hypercube_slices(hc, data, cube, [](const data_type *src, bits_type *dest, size_t n_elems) {
        memcpy(assume_simd_aligned(dest), src, n_elems * sizeof(data_type));
    });
}

template<typename Profile>
[[gnu::noinline]]
void store_hypercube(const hypercube<Profile> &hc,
                     const typename Profile::bits_type *cube,
                     const slice<typename Profile::data_type, Profile::dimensions> &data)
{
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    map_hypercube_slices(hc, data, cube, [](data_type *dest, const bits_type *src, size_t n_elems) {
        memcpy(dest, assume_simd_aligned(src), n_elems * sizeof(data_type));
    });
}


template<typename T>
void block_transform_step(T *x, size_t n, size_t s) {
    T a, b;
    b = x[0*s];
    for (size_t i = 1; i < n; ++i) {
        a = b;
        b = x[i*s];
        x[i*s] = a ^ b;
    }
}

#ifdef __AVX2__

[[gnu::always_inline]]
inline __m256i load_aligned_8x32(const uint32_t *p) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}

[[gnu::always_inline]]
inline void store_aligned_8x32(uint32_t *p, __m256i x) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(p), x);
}

[[gnu::always_inline]]
inline __m128i load_aligned_4x32(const uint32_t *p) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}

[[gnu::always_inline]]
inline void store_aligned_4x32(uint32_t *p, __m128i x) {
    _mm_store_si128(reinterpret_cast<__m128i*>(p), x);
}

[[gnu::always_inline]]
inline void block_transform_2d_32_avx2(uint32_t *x) {
    constexpr auto side_length = profile<float, 2>::hypercube_side_length;
    constexpr auto n_256bit_lanes = sizeof(uint32_t) * side_length / simd_width_bytes;
    constexpr auto words_per_256bit_lane = simd_width_bytes / sizeof(uint32_t);
    constexpr auto n_128bit_lanes = n_256bit_lanes * 2;
    constexpr auto words_per_128bit_lane = words_per_256bit_lane / 2;

    __m256i lanes_a[n_256bit_lanes];
    __m256i lanes_b[n_256bit_lanes];
    __builtin_memcpy(lanes_b, assume_simd_aligned(x), sizeof lanes_b);

    for (size_t i = 1; i < side_length; ++i) {
        __builtin_memcpy(lanes_a, lanes_b, sizeof lanes_b);
        __builtin_memcpy(lanes_b, assume_simd_aligned(x + i * side_length), sizeof lanes_b);
        for (size_t j = 0; j < n_256bit_lanes; ++j) {
            lanes_a[j] = _mm256_xor_si256(lanes_a[j], lanes_b[j]);
        }
        __builtin_memcpy(assume_simd_aligned(x + i * side_length), lanes_a, sizeof lanes_a);
    }

    for (size_t i = 0; i < side_length; ++i) {
        auto *line = x + i * side_length;
        __m128i bottom_a, bottom_b, top;
        bottom_b = load_aligned_4x32(line);
        top = _mm_slli_si128(bottom_b, 4);
        for (unsigned j = 0; j < n_128bit_lanes - 1; ++j) {
            store_aligned_4x32(line + j * words_per_128bit_lane, _mm_xor_si128(bottom_b, top));
            bottom_a = bottom_b;
            bottom_b = load_aligned_4x32(line + (j + 1) * words_per_128bit_lane);
            top = _mm_alignr_epi8(bottom_b, bottom_a, 12);
        }
        store_aligned_4x32(line + side_length - words_per_128bit_lane, _mm_xor_si128(bottom_b, top));
    }
}

#endif // __AVX2__

template<typename Profile>
[[gnu::noinline]]
void block_transform(typename Profile::bits_type *x) {
    constexpr size_t n = Profile::hypercube_side_length;

    x = assume_simd_aligned(x);
    if constexpr (Profile::dimensions == 1) {
        block_transform_step(x, n, 1);
    } else if constexpr (Profile::dimensions == 2) {
#ifdef __AVX2__
        if constexpr (sizeof(typename Profile::bits_type) == 4) {
            return block_transform_2d_32_avx2(x);
        }
#endif
        for (size_t i = 0; i < n*n; i += n) {
            block_transform_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n; ++i) {
            block_transform_step(x + i, n, n);
        }
    } else if constexpr (Profile::dimensions == 3) {
        for (size_t i = 0; i < n*n*n; i += n*n) {
            for (size_t j = 0; j < n; ++j) {
                block_transform_step(x + i + j, n, n);
            }
        }
        for (size_t i = 0; i < n*n*n; i += n) {
            block_transform_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n*n; ++i) {
            block_transform_step(x + i, n, n * n);
        }
    }
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
}

#ifdef __AVX2__

[[gnu::always_inline]]
inline void transpose_bits_32_avx2(const uint32_t *__restrict vs, uint32_t *__restrict out) {
    __m256i unpck0[4];
    __builtin_memcpy(unpck0, assume_simd_aligned(vs), sizeof unpck0);

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

    // is _mm256_inserti128_si256 faster here?
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
[[gnu::noinline]]
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
            head |= T{1} << i;
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
        if ((head >> i) & T{1}) {
            shifted[i] = load_aligned<T>(in);
            in += sizeof(T);
        } else {
            shifted[i] = 0;
        }
    }
    return in - in0;
}

template<typename Profile>
[[gnu::noinline]]
size_t zero_bit_encode(const typename Profile::bits_type *cube, std::byte *stream) {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, Profile::dimensions);

    size_t pos = 0;
    for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
        alignas(simd_width_bytes) bits_type transposed[detail::bitsof<bits_type>];
        detail::cpu::transpose_bits(cube + i, transposed);
        pos += detail::cpu::compact_zero_words(transposed, stream + pos);
    }

    return pos;
}

template<typename Profile>
[[gnu::noinline]]
size_t zero_bit_decode(const std::byte *stream, typename Profile::bits_type *cube) {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, Profile::dimensions);

    size_t pos = 0;
    for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
        alignas(simd_width_bytes) bits_type transposed[detail::bitsof<bits_type>];
        pos += detail::cpu::expand_zero_words(stream + pos, transposed);
        detail::cpu::transpose_bits(transposed, cube + i);
    }
    return pos;
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

    detail::cpu::simd_aligned_buffer<bits_type> cube(hc_size);
    file.for_each_hypercube([&](auto hc, auto hc_index) {
        auto header_pos = stream_pos;
        detail::cpu::load_hypercube(hc, data, cube.data());
        detail::cpu::block_transform<profile>(cube.data());
        stream_pos += detail::cpu::zero_bit_encode<profile>(cube.data(), static_cast<std::byte *>(stream) + stream_pos);

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

    detail::cpu::simd_aligned_buffer<bits_type> cube(hc_size);
    size_t stream_pos = file.file_header_length(); // simply skip the header
    file.for_each_hypercube([&](auto hc) {
        stream_pos += detail::cpu::zero_bit_decode<profile>(static_cast<const std::byte*>(stream) + stream_pos,
                                                            cube.data());
        detail::inverse_block_transform(cube.data(), Dims, profile::hypercube_side_length);
        detail::cpu::store_hypercube(hc, cube.data(), data);
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
