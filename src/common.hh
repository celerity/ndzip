#pragma once

#include <hcde.hh>

#include <climits>
#include <iostream>
#include <cstring>
#include <limits>
#include <optional>

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


template<typename T, unsigned Dims, typename Fn>
void for_each_in_hypercube(const slice<T, Dims> &data, const extent<Dims> &offset,
        unsigned side_length, const Fn &fn)
{
    if constexpr (Dims == 1) {
        auto *pointer = &data[offset];
        for (unsigned i = 0; i < side_length; ++i) {
            fn(pointer[i]);
        }
    } else if constexpr (Dims == 2) {
        auto stride = data.size()[1];
        auto *pointer = &data[offset];
        for (unsigned i = 0; i < side_length; ++i) {
            for (unsigned j = 0; j < side_length; ++j) {
                fn(pointer[j]);
            }
            pointer += stride;
        }
    } else if constexpr (Dims == 3) {
        auto stride0 = data.size()[1] * data.size()[2];
        auto stride1 = data.size()[2];
        auto *pointer0 = &data[offset];
        for (unsigned i = 0; i < side_length; ++i) {
            auto pointer1 = pointer0;
            for (unsigned j = 0; j < side_length; ++j) {
                for (unsigned k = 0; k < side_length; ++k) {
                    fn(pointer1[k]);
                }
                pointer1 += stride1;
            }
            pointer0 += stride0;
        }
    } else if constexpr (Dims == 4) {
        auto stride0 = data.size()[1] * data.size()[2] * data.size()[3];
        auto stride1 = data.size()[2] * data.size()[3]; auto stride2 = data.size()[3];
        auto *pointer0 = &data[offset];
        for (unsigned i = 0; i < side_length; ++i) {
            auto pointer1 = pointer0;
            for (unsigned j = 0; j < side_length; ++j) {
                auto pointer2 = pointer1;
                for (unsigned k = 0; k < side_length; ++k) {
                    for (unsigned l = 0; l < side_length; ++l) {
                        fn(pointer2[l]);
                    }
                    pointer2 += stride2;
                }
                pointer1 += stride1;
            }
            pointer0 += stride0;
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
POD load_aligned(const void *src) {
    static_assert(std::is_trivially_copyable_v<POD>);
    assert(reinterpret_cast<uintptr_t>(src) % Align == 0);
    POD a;
    memcpy(&a, static_cast<const char*>(__builtin_assume_aligned(src, Align)), sizeof(POD));
    return a;
}

template<size_t Align, typename POD>
void store_aligned(void *dest, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    assert(reinterpret_cast<uintptr_t>(dest) % Align == 0);
    memcpy(static_cast<char*>(__builtin_assume_aligned(dest, Align)), &a, sizeof(POD));
}

template<typename Integer>
using next_larger_uint = std::conditional_t<std::is_same_v<Integer, uint8_t>, uint16_t,
      std::conditional_t<std::is_same_v<Integer, uint16_t>, uint32_t,
      std::conditional_t<std::is_same_v<Integer, uint32_t>, uint64_t,
      std::conditional_t<std::is_same_v<Integer, uint64_t>, uint128_t, void>>>>;

template<typename Void, size_t Align>
class basic_bit_ptr {
    static_assert(std::is_void_v<Void>);

    public:
        using address_type = Void*;
        using bit_offset_type = size_t;
        constexpr static size_t byte_alignment = Align;
        constexpr static size_t bit_alignment = Align * CHAR_BIT;

        constexpr basic_bit_ptr(std::nullptr_t) noexcept {}

        constexpr basic_bit_ptr() noexcept = default;

        basic_bit_ptr(address_type aligned_address, size_t bit_offset)
            : _aligned_address(aligned_address)
            , _bit_offset(bit_offset)
        {
            assert(reinterpret_cast<uintptr_t>(aligned_address) % byte_alignment == 0);
            assert(bit_offset < bit_alignment);
        }

        template<typename OtherVoid,
            std::enable_if_t<std::is_const_v<Void> && !std::is_const_v<OtherVoid>, int> = 0>
        basic_bit_ptr(const basic_bit_ptr<OtherVoid, Align> &other)
            : _aligned_address(other._aligned_address)
            , _bit_offset(other._bit_offset)
        {
        }

        static basic_bit_ptr from_unaligned_pointer(address_type unaligned) {
            auto misalign = reinterpret_cast<uintptr_t>(unaligned) % byte_alignment;
            return basic_bit_ptr(reinterpret_cast<byte_address_type>(unaligned) - misalign,
                    misalign * CHAR_BIT);
        }

        address_type aligned_address() const {
            return _aligned_address;
        }

        size_t bit_offset() const {
            return _bit_offset;
        }

        void advance(size_t n_bits) {
            _aligned_address = reinterpret_cast<byte_address_type>(_aligned_address)
                + (_bit_offset + n_bits) / bit_alignment * byte_alignment;
            _bit_offset = (_bit_offset + n_bits) % bit_alignment;

            assert(reinterpret_cast<uintptr_t>(_aligned_address) % byte_alignment == 0);
            assert(_bit_offset < bit_alignment);
        }

        friend bool operator==(basic_bit_ptr lhs, basic_bit_ptr rhs) {
            return lhs._aligned_address == rhs._aligned_address && lhs._bit_offset == rhs._bit_offset;
        }

        friend bool operator!=(basic_bit_ptr lhs, basic_bit_ptr rhs) {
            return !operator==(lhs, rhs);
        }

    private:
        using byte_address_type = std::conditional_t<std::is_const_v<Void>, const char, char>*;

        Void *_aligned_address = nullptr;
        size_t _bit_offset = 0;

        friend class basic_bit_ptr<std::add_const_t<Void>, Align>;
};

template<size_t Align>
using bit_ptr = basic_bit_ptr<void, Align>;

template<size_t Align>
using const_bit_ptr = basic_bit_ptr<const void, Align>;

template<typename Void, size_t Align>
size_t ceil_byte_offset(const void *from, basic_bit_ptr<Void, Align> to) {
    return reinterpret_cast<const char*>(to.aligned_address())
        - reinterpret_cast<const char*>(from) + (to.bit_offset() + (CHAR_BIT - 1)) / CHAR_BIT;
}

template<typename Integer>
Integer load_bits(const_bit_ptr<sizeof(Integer)> src, size_t n_bits) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using window = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    auto a = endian_transform<window>(
            load_aligned<sizeof(Integer), window>(src.aligned_address()));
    auto shift = bitsof<window> - src.bit_offset() - n_bits;
    return static_cast<Integer>((a >> shift) & ~(~window{} << n_bits));
}

template<typename Integer>
void store_bits_linear(bit_ptr<sizeof(Integer)> dest, size_t n_bits, Integer value) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using window = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    assert((window{value} >> n_bits) == 0);
    assert(load_bits<Integer>(dest, n_bits) == 0);
    auto a = load_aligned<sizeof(Integer), window>(dest.aligned_address());
    auto shift = bitsof<window> - dest.bit_offset() - n_bits;
    a = a | endian_transform(static_cast<window>(static_cast<window>(value) << shift));
    store_aligned<sizeof(Integer)>(dest.aligned_address(), a);
}

template<unsigned Dims, typename Fn>
void for_each_border_slice_recursive(const extent<Dims> &size, extent<Dims> pos,
        unsigned side_length, unsigned d, unsigned smallest_dim_with_border, const Fn &fn)
{
    auto border_begin = size[d] / side_length * side_length;
    auto border_end = size[d];

    if (d < smallest_dim_with_border) {
        for (pos[d] = 0; pos[d] < border_begin; ++pos[d]) {
            for_each_border_slice_recursive(size, pos, side_length, d + 1,
                    smallest_dim_with_border, fn);
        }
    }

    if (border_begin < border_end) {
        auto begin_pos = pos;
        begin_pos[d] = border_begin;
        auto end_pos = pos;
        end_pos[d] = border_end;
        auto offset = linear_index(size, begin_pos);
        auto count = linear_index(size, end_pos) - offset;
        fn(offset, count);
    }
}

template<unsigned Dims, typename Fn>
void for_each_border_slice(const extent<Dims> &size, unsigned side_length, const Fn &fn) {
    std::optional<unsigned> smallest_dim_with_border;
    for (unsigned d = 0; d < Dims; ++d) {
        if (size[d] / side_length == 0) {
            // special case: the whole array is a border
            fn(0, size.linear_offset());
            return;
        }
        if (size[d] % side_length != 0) {
            smallest_dim_with_border = static_cast<int>(d);
        }
    }
    if (smallest_dim_with_border) {
        for_each_border_slice_recursive(size, extent<Dims>{}, side_length, 0,
                *smallest_dim_with_border, fn);
    }
}

template<typename DataType, unsigned Dims>
size_t pack_border(void *dest, const slice<DataType, Dims> &src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t dest_offset = 0;
    for_each_border_slice(src.size(), side_length, [&](size_t src_offset, size_t count) {
        memcpy(static_cast<char*>(dest) + dest_offset, src.data() + src_offset,
                count * sizeof(DataType));
        dest_offset += count * sizeof(DataType);
    });
    return dest_offset;
}

template<typename DataType, unsigned Dims>
size_t unpack_border(const slice<DataType, Dims> &dest, const void *src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t src_offset = 0;
    for_each_border_slice(dest.size(), side_length, [&](size_t dest_offset, size_t count) {
        memcpy(dest.data() + dest_offset, static_cast<const char*>(src) + src_offset,
                count * sizeof(DataType));
        src_offset += count * sizeof(DataType);
    });
    return src_offset;
}

template<unsigned Dims>
size_t border_element_count(const extent<Dims> &e, unsigned side_length) {
    size_t n_cube_elems = 1;
    size_t n_all_elems = 1;
    for (unsigned d = 0; d < Dims; ++d) {
        n_cube_elems *= e[d] / side_length * side_length;
        n_all_elems *= e[d];
    }
    return n_all_elems - n_cube_elems;
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
        domain |= bits[i] ^ ref;
    }
    auto remainder_width = detail::significant_bits(domain);
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    memcpy(stream, &ref, sizeof ref);
    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    dest.advance(detail::bitsof<T>);
    detail::store_bits_linear<bits_type>(dest, width_width, remainder_width);
    dest.advance(width_width);
    if (remainder_width > 0) { // store_bits_linear does not allow n_bits == 0
        for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            detail::store_bits_linear<bits_type>(dest, remainder_width, bits[i] ^ ref);
            dest.advance(remainder_width);
        }
    }
    return ceil_byte_offset(stream, dest);
}


template<typename T, unsigned Dims>
size_t fast_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    bits_type ref;
    memcpy(&ref, stream, sizeof ref);
    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    src.advance(detail::bitsof<T>);
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    auto remainder_width = detail::load_bits<bits_type>(src, width_width);
    src.advance(width_width);
    if (remainder_width > 0) { // load_bits does not allow n_bits == 0
        bits[0] = ref;
        for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            bits[i] = detail::load_bits<bits_type>(src, remainder_width) ^ ref;
            src.advance(remainder_width);
        }
    } else {
        for (size_t i = 0; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            bits[i] = ref;
        }
    }
    return ceil_byte_offset(stream, src);
}

} // namespace hcde
