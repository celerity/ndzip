#pragma once

#include <hcde.hh>

#include <climits>
#include <cstring>
#include <limits>

#define HCDE_BIG_ENDIAN 0
#define HCDE_LITTLE_ENDIAN 1

#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
    defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
#   define HCDE_ENDIAN HCDE_BIG_ENDIAN
#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN || \
    defined(__LITTLE_ENDIAN__) || \
    defined(__ARMEL__) || \
    defined(__THUMBEL__) || \
    defined(__AARCH64EL__) || \
    defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__)
#   define HCDE_ENDIAN HCDE_LITTLE_ENDIAN
#else
#   error "Unknown endian"
#endif


namespace hcde::detail {

template<unsigned Dims, typename Fn>
void for_each_in_hypercube(unsigned side_length, const Fn &fn) {
    if constexpr (Dims == 1) {
        for (unsigned i = 0; i < side_length; ++i) {
            fn(extent<1>{i});
        }
    } else if constexpr (Dims == 2) {
        for (unsigned i = 0; i < side_length; ++i) {
            for (unsigned j = 0; j < side_length; ++j) {
                fn(extent<2>{i, j});
            }
        }
    } else if constexpr (Dims == 3) {
        for (unsigned i = 0; i < side_length; ++i) {
            for (unsigned j = 0; j < side_length; ++j) {
                for (unsigned k = 0; k < side_length; ++k) {
                    fn(extent<3>{i, j, k});
                }
            }
        }
    } else if constexpr (Dims == 4) {
        for (unsigned i = 0; i < side_length; ++i) {
            for (unsigned j = 0; j < side_length; ++j) {
                for (unsigned k = 0; k < side_length; ++k) {
                    for (unsigned l = 0; l < side_length; ++l) {
                        fn(extent<4>{i, j, k, l});
                    }
                }
            }
        }
    } else {
        static_assert(Dims != Dims);
    }
}

inline unsigned significant_bits(unsigned long long value) {
    return CHAR_BIT * sizeof(unsigned long long) - __builtin_clzll(value);
}

inline unsigned significant_bits(unsigned long value) {
    return CHAR_BIT * sizeof(unsigned long) - __builtin_clzl(value);
}

inline unsigned significant_bits(unsigned int value) {
    return CHAR_BIT * sizeof(unsigned int) - __builtin_clz(value);
}

inline unsigned significant_bits(unsigned short value) {
    return significant_bits(static_cast<unsigned int>(value));
}

inline unsigned significant_bits(unsigned char value) {
    return significant_bits(static_cast<unsigned int>(value));
}

template<typename Integer>
Integer shl(Integer value, unsigned shift) {
    return shift == CHAR_BIT * sizeof(Integer) ? 0 : value << shift;
}

template<typename Integer>
Integer shr(Integer value, unsigned shift) {
    return shift == CHAR_BIT * sizeof(Integer) ? 0 : value >> shift;
}

template<typename Integer>
Integer load_bits(const void *src, size_t src_bit_offset, size_t n_bits) {
    static_assert(sizeof(Integer) <= 4);
    std::uint64_t a;
    memcpy(a, static_cast<const char*>(src) + src_bit_offset / integer_n_bits, sizeof a);
    if (HCDE_ENDIAN == HCDE_LITTLE_ENDIAN) {
        a = __builtin_bswap64(a);
    }

    auto integer_n_bits = CHAR_BIT * sizeof(Integer);
    assert(n_bits <= integer_n_bits);
    Integer a[2];
    auto shift = src_bit_offset % integer_n_bits;
    return shr(shl(a[0], shift) | shr(a[1], (integer_n_bits - shift)), integer_n_bits - n_bits);
}

template<typename Integer>
void store_bits_linear(void *dest, size_t dest_bit_offset, size_t n_bits, Integer value) {
    auto integer_n_bits = CHAR_BIT * sizeof(Integer);
    assert(n_bits <= integer_n_bits);
    auto shift = dest_bit_offset % integer_n_bits;
    Integer a[2];
    auto addr = static_cast<char*>(__builtin_assume_aligned(dest, sizeof(Integer)))
        + dest_bit_offset / integer_n_bits;
    memcpy(a, addr, sizeof a[0]);
    assert(shl(a[0], integer_n_bits - shift) == 0);
    a[0] |= shr(value, shift);
    a[1] = shl(value, integer_n_bits - shift);
    memcpy(addr, a, sizeof a);
}

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
auto fast_profile<T, Dims>::load_value(const data_type *data) const -> bits_type {
    bits_type bits;
    memcpy(&bits, data, sizeof bits);
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits << 1u) | (bits >> (sizeof bits * CHAR_BIT - 1));
    } else if constexpr (std::is_signed_v<data_type>) {
        bits += static_cast<bits_type>(std::numeric_limits<data_type>::min());
    }
    return bits;
}


template<typename T, unsigned Dims>
void fast_profile<T, Dims>::store_value(data_type *data, bits_type bits) const {
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits >> 1u) | (bits << (sizeof bits * CHAR_BIT - 1));
    } else if constexpr (std::is_signed_v<data_type>) {
        bits -= static_cast<bits_type>(std::numeric_limits<data_type>::min());
    }
    memcpy(data, &bits, sizeof bits);
}


template<typename T, unsigned Dims>
size_t fast_profile<T, Dims>::encode_block(const bits_type *bits, void *stream) const {
    auto ref = bits[0];
    bits_type domain = 0;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        domain |= bits[i];
    }
    auto remainder_width = detail::significant_bits(domain ^ ref);
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    memcpy(stream, &ref, sizeof ref);
    size_t bit_offset = CHAR_BIT * sizeof ref;
    detail::store_bits_linear(stream, bit_offset, width_width, remainder_width);
    bit_offset += width_width;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        detail::store_bits_linear(stream, bit_offset, remainder_width, bits[i] ^ ref);
        bit_offset += remainder_width;
    }
    return (bit_offset + CHAR_BIT-1) / CHAR_BIT;
}


template<typename T, unsigned Dims>
void fast_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    bits_type ref;
    memcpy(&ref, stream, sizeof ref);
    size_t bit_offset = CHAR_BIT * sizeof ref;
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    auto remainder_width = detail::load_bits<unsigned>(stream, bit_offset, width_width);
    bit_offset += width_width;
    bits[0] = ref;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        bits[i] = detail::load_bits<bits_type>(stream, bit_offset, remainder_width) ^ ref;
    }
}

} // namespace hcde
