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

#ifdef __SIZEOF_INT128__
using native_uint128_t = unsigned __int128;
#   define HCDE_HAVE_NATIVE_UINT128_T 1
#else
#   define HCDE_HAVE_NATIVE_UINT128_T 0
#endif


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


template<unsigned Dims, typename Fn>
void for_each_hypercube_offset(extent<Dims> size, unsigned side_length, const Fn &fn) {
    if constexpr (Dims == 1) {
        for (unsigned i = 0; i < size[0] / side_length; ++i) {
            fn(extent<1>{i * side_length});
        }
    } else if constexpr (Dims == 2) {
        for (unsigned i = 0; i < size[0] / side_length; ++i) {
            for (unsigned j = 0; j < size[1] / side_length; ++j) {
                fn(extent<2>{i * side_length, j * side_length});
            }
        }
    } else if constexpr (Dims == 3) {
        for (unsigned i = 0; i < size[0] / side_length; ++i) {
            for (unsigned j = 0; j < size[1] / side_length; ++j) {
                for (unsigned k = 0; k < size[2] / side_length; ++k) {
                    fn(extent<3>{i * side_length, j * side_length, k * side_length});
                }
            }
        }
    } else if constexpr (Dims == 4) {
        for (unsigned i = 0; i < size[0] / side_length; ++i) {
            for (unsigned j = 0; j < size[1] / side_length; ++j) {
                for (unsigned k = 0; k < size[2] / side_length; ++k) {
                    for (unsigned l = 0; l < size[3] / side_length; ++l) {
                        fn(extent<4>{i * side_length, j * side_length, k * side_length,
                                l * side_length});
                    }
                }
            }
        }
    } else {
        static_assert(Dims != Dims);
    }
}


template<typename Integer>
Integer endian_transform(Integer value);

class alignas(16) emulated_uint128 {
    public:
        constexpr emulated_uint128() noexcept = default;

        constexpr explicit emulated_uint128(uint64_t v)
            : _c{0, v}
        {
        }

        constexpr emulated_uint128 operator>>(unsigned shift) const {
            assert(shift < 128);
            if (shift == 0) {
                return *this;
            } else if (shift < 64) {
                return {_c[0] >> shift, (_c[0] << (64 - shift)) | (_c[1] >> shift)};
            } else {
                return {0, _c[0] >> (shift - 64)};
            }
        }

        constexpr emulated_uint128 operator<<(unsigned shift) const {
            assert(shift < 128);
            if (shift == 0) {
                return *this;
            } else if (shift < 64) {
                return {(_c[0] << shift) | (_c[1] >> (64 - shift)), _c[1] << shift};
            } else {
                return {_c[1] << (shift - 64), 0};
            }
        }

        constexpr emulated_uint128 operator~() const {
            return {~_c[0], ~_c[1]};
        }

        constexpr emulated_uint128 operator|(emulated_uint128 other) const {
            return {_c[0] | other._c[0], _c[1] | other._c[1]};
        }

        constexpr emulated_uint128 operator&(emulated_uint128 other) const {
            return {_c[0] & other._c[0], _c[1] & other._c[1]};
        }

        constexpr explicit operator uint64_t() const {
            return _c[1];
        }

    private:
        friend emulated_uint128 endian_transform<emulated_uint128>(emulated_uint128);

        constexpr emulated_uint128(uint64_t hi, uint64_t lo)
            : _c{hi, lo}
        {
        }

        uint64_t _c[2]{};
};

template<typename T>
constexpr inline size_t bitsof = CHAR_BIT * sizeof(T);

template<typename Integer>
unsigned significant_bits(Integer value) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    // On x86, this is branchless `bitsof<Integer> - lzcnt(x)`, but __builtin_clz has UB for x == 0.
    // Hoisting the zero-check out of the `if constexpr` causes both Clang 10 and GCC 10 to
    // mis-optimize this and emit a branch / cmove. This version is correctly optimized by Clang.
    if constexpr (std::is_same_v<Integer, unsigned long long>) {
        return bitsof<unsigned long long>
            - (value ? __builtin_clzll(value) : bitsof<unsigned long long>);
    } else if constexpr (std::is_same_v<Integer, unsigned long>) {
        return bitsof<unsigned long> - (value ? __builtin_clzl(value) : bitsof<unsigned long>);
    } else if constexpr (std::is_same_v<Integer, unsigned int>) {
        return bitsof<unsigned int> - (value ? __builtin_clz(value) : bitsof<unsigned int>);
    } else {
        static_assert(sizeof(Integer) <= sizeof(unsigned int));
        return significant_bits(static_cast<unsigned int>(value));
    }
}

template<typename Integer>
Integer endian_transform(Integer value) {
    if constexpr (HCDE_ENDIAN == HCDE_LITTLE_ENDIAN) {
        if constexpr (std::is_same_v<Integer, emulated_uint128>) {
            return {__builtin_bswap64(value._c[0]), __builtin_bswap64(value._c[1])};
#if HCDE_HAVE_NATIVE_UINT128_T
        } else if constexpr (std::is_same_v<Integer, native_uint128_t>) {
            return (native_uint128_t{__builtin_bswap64(value)} << 64u)
             | (native_uint128_t{__builtin_bswap64(value >> 64u)});
#endif
        } else if constexpr (std::is_same_v<Integer, uint64_t>) {
            return __builtin_bswap64(value);
        } else if constexpr (std::is_same_v<Integer, uint32_t>) {
            return __builtin_bswap32(value);
        } else if constexpr (std::is_same_v<Integer, uint16_t>) {
            return __builtin_bswap16(value);
        } else {
            static_assert(std::is_same_v<Integer, uint8_t>);
            return value;
        }
    } else {
        return value;
    }
}

#if HCDE_HAVE_NATIVE_UINT128_T
using uint128_t = native_uint128_t;
#else
using uint128_t = emulated_uint128;
#endif

template<size_t Align, typename POD>
POD load_aligned(const void *src, size_t byte_offset) {
    static_assert(std::is_trivially_copyable_v<POD>);
    assert(reinterpret_cast<uintptr_t>(src) % Align == 0);
    assert(byte_offset % Align == 0);
    POD a;
    memcpy(&a, static_cast<const char*>(__builtin_assume_aligned(src, Align)) + byte_offset,
            sizeof(POD));
    return a;
}

template<size_t Align, typename POD>
void store_aligned(void *dest, size_t byte_offset, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    assert(reinterpret_cast<uintptr_t>(dest) % Align == 0);
    assert(byte_offset % Align == 0);
    memcpy(static_cast<char*>(__builtin_assume_aligned(dest, Align)) + byte_offset, &a,
            sizeof(POD));
}

template<typename Integer>
using next_larger_uint = std::conditional_t<std::is_same_v<Integer, uint8_t>, uint16_t,
      std::conditional_t<std::is_same_v<Integer, uint16_t>, uint32_t,
      std::conditional_t<std::is_same_v<Integer, uint32_t>, uint64_t,
      std::conditional_t<std::is_same_v<Integer, uint64_t>, uint128_t, void>>>>;

template<typename Integer>
Integer load_bits(const void *src, size_t src_bit_offset, size_t n_bits) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using window = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    auto src_offset = (src_bit_offset / bitsof<Integer>) * sizeof(Integer);
    auto a = endian_transform<window>(load_aligned<sizeof(Integer), window>(src, src_offset));
    auto shift = bitsof<window> - (src_bit_offset % bitsof<Integer>) - n_bits;
    return static_cast<Integer>((a >> shift) & ~(~window{} << n_bits));
}

template<typename Integer>
void store_bits_linear(void *dest, size_t dest_bit_offset, size_t n_bits, Integer value) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using window = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    assert(load_bits<Integer>(dest, dest_bit_offset, n_bits) == 0);
    auto dest_offset = (dest_bit_offset / bitsof<Integer>) * sizeof(Integer);
    auto a = load_aligned<sizeof(Integer), window>(dest, dest_offset);
    auto shift = bitsof<window> - (dest_bit_offset % bitsof<Integer>) - n_bits;
    a = a | endian_transform(static_cast<window>(static_cast<window>(value) << shift));
    store_aligned<sizeof(Integer)>(dest, dest_offset, a);
}

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
auto fast_profile<T, Dims>::load_value(const data_type *data) const -> bits_type {
    bits_type bits;
    memcpy(&bits, data, sizeof bits);
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits << 1u) | (bits >> (detail::bitsof<bits_type> - 1));
    } else if constexpr (std::is_signed_v<data_type>) {
        bits += static_cast<bits_type>(std::numeric_limits<data_type>::min());
    }
    return bits;
}


template<typename T, unsigned Dims>
void fast_profile<T, Dims>::store_value(data_type *data, bits_type bits) const {
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits >> 1u) | (bits << (detail::bitsof<bits_type> - 1));
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
    size_t bit_offset = detail::bitsof<T>;
    detail::store_bits_linear<bits_type>(stream, bit_offset, width_width, remainder_width);
    bit_offset += width_width;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        detail::store_bits_linear<bits_type>(stream, bit_offset, remainder_width, bits[i] ^ ref);
        bit_offset += remainder_width;
    }
    return (bit_offset + CHAR_BIT-1) / CHAR_BIT;
}


template<typename T, unsigned Dims>
size_t fast_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    bits_type ref;
    memcpy(&ref, stream, sizeof ref);
    size_t bit_offset = detail::bitsof<T>;
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    auto remainder_width = detail::load_bits<bits_type>(stream, bit_offset, width_width);
    bit_offset += width_width;
    bits[0] = ref;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        bits[i] = detail::load_bits<bits_type>(stream, bit_offset, remainder_width) ^ ref;
        bit_offset += remainder_width;
    }
    return (bit_offset + CHAR_BIT-1) / CHAR_BIT;
}

} // namespace hcde
