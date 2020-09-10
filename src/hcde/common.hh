#pragma once

#include <hcde.hh>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <limits>
#include <optional>


#define HCDE_BIG_ENDIAN 0
#define HCDE_LITTLE_ENDIAN 1

#if defined(__BYTE_ORDER)
#   if __BYTE_ORDER == __BIG_ENDIAN
#       define HCDE_ENDIAN HCDE_BIG_ENDIAN
#   else
#       define HCDE_ENDIAN HCDE_LITTLE_ENDIAN
#   endif
#elif defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
#   define HCDE_ENDIAN HCDE_BIG_ENDIAN
#elif defined(__LITTLE_ENDIAN__) || \
    defined(__ARMEL__) || \
    defined(__THUMBEL__) || \
    defined(__AARCH64EL__) || \
    defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__)
#   define HCDE_ENDIAN HCDE_LITTLE_ENDIAN
#else
#   error "Unknown endianess"
#endif

#define HCDE_WARP_SIZE (size_t{32})
#define HCDE_SUPERBLOCK_SIZE HCDE_WARP_SIZE


namespace hcde::detail {

using file_offset_type = uint64_t;
using superblock_offset_type = uint32_t;

constexpr auto superblock_size = HCDE_SUPERBLOCK_SIZE;


#ifdef __SIZEOF_INT128__
using native_uint128_t = unsigned __int128;
#   define HCDE_HAVE_NATIVE_UINT128_T 1
#else
#   define HCDE_HAVE_NATIVE_UINT128_T 0
#endif


template<typename Fn, typename Index, typename T>
[[gnu::always_inline]] void invoke_for_element(Fn &&fn, Index index, T &&value) {
    if constexpr (std::is_invocable_v<Fn, T, Index>) {
        fn(std::forward<T>(value), index);
    } else {
        fn(std::forward<T>(value));
    }
}


template<typename Integer>
Integer endian_transform(Integer value);

class alignas(16) emulated_uint128 {
    public:
        constexpr emulated_uint128() noexcept = default;

        constexpr explicit emulated_uint128(uint64_t v)
            : _c{0, v} {
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
            : _c{hi, lo} {
        }

        uint64_t _c[2]{};
};

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

template<typename POD>
POD load_unaligned(const void *src) {
    static_assert(std::is_trivially_copyable_v<POD>);
    POD a;
    memcpy(&a, src, sizeof(POD));
    return a;
}

template<size_t Align, typename POD>
POD load_aligned(const void *src) {
    assert(reinterpret_cast<uintptr_t>(src) % Align == 0);
    return load_unaligned<POD>(__builtin_assume_aligned(src, Align));
}

template<typename POD>
POD load_aligned(const void *src) {
    return load_aligned<alignof(POD), POD>(src);
}

template<typename POD>
void store_unaligned(void *dest, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    memcpy(dest, &a, sizeof(POD));
}

template<size_t Align, typename POD>
void store_aligned(void *dest, POD a) {
    assert(reinterpret_cast<uintptr_t>(dest) % Align == 0);
    store_unaligned(__builtin_assume_aligned(dest, Align), a);
}

template<typename POD>
void store_aligned(void *dest, POD a) {
    store_aligned<alignof(POD), POD>(dest, a);
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
        using address_type = Void *;
        using bit_offset_type = size_t;
        constexpr static size_t byte_alignment = Align;
        constexpr static size_t bit_alignment = Align * CHAR_BIT;

        constexpr basic_bit_ptr(std::nullptr_t) noexcept {}

        constexpr basic_bit_ptr() noexcept = default;

        basic_bit_ptr(address_type aligned_address, size_t bit_offset)
            : _aligned_address(aligned_address)
            , _bit_offset(bit_offset) {
            assert(reinterpret_cast<uintptr_t>(aligned_address) % byte_alignment == 0);
            assert(bit_offset < bit_alignment);
        }

        template<typename OtherVoid,
            std::enable_if_t<std::is_const_v<Void> && !std::is_const_v<OtherVoid>, int> = 0>
        basic_bit_ptr(const basic_bit_ptr<OtherVoid, Align> &other)
            : _aligned_address(other._aligned_address)
            , _bit_offset(other._bit_offset) {
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
        using byte_address_type = std::conditional_t<std::is_const_v<Void>, const char, char> *;

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
    return reinterpret_cast<const char *>(to.aligned_address())
        - reinterpret_cast<const char *>(from) + (to.bit_offset() + (CHAR_BIT - 1)) / CHAR_BIT;
}

template<typename Integer>
Integer load_bits(const_bit_ptr<sizeof(Integer)> src, size_t n_bits) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using word = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    auto a = endian_transform<word>(
        load_aligned<sizeof(Integer), word>(src.aligned_address()));
    auto shift = bitsof<word> - src.bit_offset() - n_bits;
    return static_cast<Integer>((a >> shift) & ~(~word{} << n_bits));
}

template<typename Integer>
void store_bits_linear(bit_ptr<sizeof(Integer)> dest, size_t n_bits, Integer value) {
    static_assert(std::is_integral_v<Integer> && std::is_unsigned_v<Integer>);
    using word = next_larger_uint<Integer>;
    assert(n_bits > 0 && n_bits <= bitsof<Integer>);
    assert((word{value} >> n_bits) == 0);
    assert(load_bits<Integer>(dest, n_bits) == 0);
    auto a = load_aligned<sizeof(Integer), word>(dest.aligned_address());
    auto shift = bitsof<word> - dest.bit_offset() - n_bits;
    // TODO This is a read-modify-write op. We can probably eliminate the entire transfer
    // from memory by keeping the last word in a register. This might incur a branch.
    a = a | endian_transform(static_cast<word>(static_cast<word>(value) << shift));
    store_aligned<sizeof(Integer)>(dest.aligned_address(), a);
}

template<unsigned Dims, typename Fn>
void for_each_border_slice_recursive(const extent<Dims> &size, extent<Dims> pos,
    unsigned side_length, unsigned d, unsigned smallest_dim_with_border, const Fn &fn) {
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
            fn(0, num_elements(size));
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
[[gnu::noinline]]
size_t pack_border(void *dest, const slice<DataType, Dims> &src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t dest_offset = 0;
    for_each_border_slice(src.size(), side_length, [&](size_t src_offset, size_t count) {
        memcpy(static_cast<char *>(dest) + dest_offset, src.data() + src_offset,
            count * sizeof(DataType));
        dest_offset += count * sizeof(DataType);
    });
    return dest_offset;
}

template<typename DataType, unsigned Dims>
[[gnu::noinline]]
size_t unpack_border(const slice<DataType, Dims> &dest, const void *src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t src_offset = 0;
    for_each_border_slice(dest.size(), side_length, [&](size_t dest_offset, size_t count) {
        memcpy(dest.data() + dest_offset, static_cast<const char *>(src) + src_offset,
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

template<unsigned Dims>
extent<Dims> global_hypercube_offset(const extent<Dims> &hypercubes_per_dim, unsigned side_length, size_t index) {
    // less-or-equal: Allow generating one-past-end offset
    assert(index <= hypercubes_per_dim[0]);
    extent<Dims> off;
    for (unsigned d = 1; d < Dims; ++d) {
        off[d - 1] = index / hypercubes_per_dim[d] * side_length;
        index %= hypercubes_per_dim[d];
    }
    off[Dims - 1] = index * side_length;
    return off;
}

template<typename Profile>
class hypercube {
    public:
        explicit hypercube(const extent<Profile::dimensions> &offset)
            : _offset(offset) {
        }

        template<typename DataType, typename Fn>
        [[gnu::always_inline]] void for_each_cell(const slice<DataType, Profile::dimensions> &data,
            const Fn &fn) const {
            constexpr auto side_length = Profile::hypercube_side_length;
            if constexpr (Profile::dimensions == 1) {
                auto *pointer = &data[_offset];
                for (unsigned i = 0; i < side_length; ++i) {
                    invoke_for_element(fn, i, pointer + i);
                }
            } else if constexpr (Profile::dimensions == 2) {
                auto stride = data.size()[1];
                auto *pointer = &data[_offset];
                for (unsigned i = 0; i < side_length; ++i) {
                    for (unsigned j = 0; j < side_length; ++j) {
                        invoke_for_element(fn, i * side_length + j, pointer + j);
                    }
                    pointer += stride;
                }
            } else if constexpr (Profile::dimensions == 3) {
                auto stride0 = data.size()[1] * data.size()[2];
                auto stride1 = data.size()[2];
                auto *pointer0 = &data[_offset];
                for (unsigned i = 0; i < side_length; ++i) {
                    auto pointer1 = pointer0;
                    for (unsigned j = 0; j < side_length; ++j) {
                        for (unsigned k = 0; k < side_length; ++k) {
                            invoke_for_element(fn, (i * side_length + j) * side_length + k, pointer1 + k);
                        }
                        pointer1 += stride1;
                    }
                    pointer0 += stride0;
                }
            } else if constexpr (Profile::dimensions == 4) {
                auto stride0 = data.size()[1] * data.size()[2] * data.size()[3];
                auto stride1 = data.size()[2] * data.size()[3];
                auto stride2 = data.size()[3];
                auto *pointer0 = &data[_offset];
                for (unsigned i = 0; i < side_length; ++i) {
                    auto pointer1 = pointer0;
                    for (unsigned j = 0; j < side_length; ++j) {
                        auto pointer2 = pointer1;
                        for (unsigned k = 0; k < side_length; ++k) {
                            for (unsigned l = 0; l < side_length; ++l) {
                                invoke_for_element(fn, ((i * side_length + j) * side_length + k) * side_length + l,
                                    pointer2 + l);
                            }
                            pointer2 += stride2;
                        }
                        pointer1 += stride1;
                    }
                    pointer0 += stride0;
                }
            } else {
                static_assert(Profile::dimensions != Profile::dimensions);
            }
        }

        extent<Profile::dimensions> global_offset() const {
            return _offset;
        }

    private:
        extent<Profile::dimensions> _offset;
};

template<typename Profile>
class superblock {
    public:
        superblock(extent<Profile::dimensions> hypercubes_per_dim, size_t first_hc_index)
            : _hypercubes_per_dim(hypercubes_per_dim)
            , _first_hc_index(first_hc_index)
        {
        }

        size_t num_hypercubes() const {
            // For some bizarre reason, replacing HCDE_SUPERBLOCK_SIZE with detail::superblock_size
            // causes a miscompilation with hipSYCL where superblock_size is treated as containing
            // zero.
            return std::min(_hypercubes_per_dim[0] - _first_hc_index, HCDE_SUPERBLOCK_SIZE);
        }

        size_t num_hypercubes_per_dim(size_t d) const {
            return _hypercubes_per_dim[d];
        }

        template<typename Fn>
        void for_each_hypercube(Fn &&fn) const {
            for (size_t index = 0; index < num_hypercubes(); ++index) {
                invoke_for_element(fn, index, hypercube<Profile>{global_hypercube_offset(
                    _hypercubes_per_dim, Profile::hypercube_side_length, _first_hc_index + index)});
            }
        }

        hypercube<Profile> hypercube_at(size_t hc_offset_index) const {
            // less-or-equal: Allow generating one-past-end offset
            assert(hc_offset_index <= HCDE_SUPERBLOCK_SIZE);
            return hypercube<Profile>{global_hypercube_offset(_hypercubes_per_dim, Profile::hypercube_side_length,
                _first_hc_index + hc_offset_index)};
        }

    private:
        extent<Profile::dimensions> _hypercubes_per_dim;
        size_t _first_hc_index;
};

template<typename Profile>
class file {
    public:
        explicit file(extent<Profile::dimensions> size)
            : _size(size)
        {
            size_t hypercubes = 1;
            for (unsigned nd = 0; nd < Profile::dimensions; ++nd) {
                auto d = Profile::dimensions - 1 - nd;
                hypercubes *= _size[d] / Profile::hypercube_side_length;
                _hypercubes_per_dim[d] = hypercubes;
            }
        }

        size_t num_superblocks() const {
            return (_hypercubes_per_dim[0] + superblock_size - 1) / superblock_size;
        }

        template<typename Fn>
        void for_each_superblock(Fn &&fn) const {
            for (size_t start = 0; start < _hypercubes_per_dim[0]; start += superblock_size) {
                invoke_for_element(fn, start / superblock_size, superblock<Profile>(_hypercubes_per_dim, start));
            }
        }

        constexpr size_t num_hypercubes() const {
            return _hypercubes_per_dim[0];
        }

        constexpr size_t max_num_hypercubes_per_superblock() const {
            return superblock_size;
        }

        constexpr size_t file_header_length() const {
            return std::max(size_t{1}, num_superblocks()) * sizeof(file_offset_type);
        }

        constexpr size_t superblock_header_length() const {
            return max_num_hypercubes_per_superblock()
                * sizeof(typename Profile::hypercube_offset_type);
        }

        constexpr size_t combined_length_of_all_headers() const {
            return file_header_length() + num_superblocks() * superblock_header_length();
        }

    private:
        extent<Profile::dimensions> _size;
        extent<Profile::dimensions> _hypercubes_per_dim;
};

template<typename T, typename Enable=void>
struct positional_bits_repr;

template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>> {
    using numeric_type = T;
    using bits_type = T;

    static bits_type to_bits(numeric_type x) {
        return x;
    }

    static numeric_type from_bits(bits_type x) {
        return x;
    }
};

template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>>> {
    using numeric_type = T;
    using bits_type = std::make_unsigned_t<T>;

    static bits_type to_bits(numeric_type x) {
        return static_cast<bits_type>(x)
            - static_cast<bits_type>(std::numeric_limits<numeric_type>::min());
    }

    static numeric_type from_bits(bits_type x) {
        return static_cast<numeric_type>(
            x + static_cast<bits_type>(std::numeric_limits<numeric_type>::min()));
    }
};


template<typename T>
struct float_bits_traits;

template<>
struct float_bits_traits<float> {
    static_assert(sizeof(float) == 4);
    static_assert(std::numeric_limits<float>::is_iec559); // IEE 754

    using bits_type = uint32_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 8;
    constexpr static unsigned mantissa_bits = 23;
};

template<>
struct float_bits_traits<double> {
    static_assert(sizeof(double) == 8);
    static_assert(std::numeric_limits<double>::is_iec559); // IEE 754

    using bits_type = uint64_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 11;
    constexpr static unsigned mantissa_bits = 52;
};


template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    constexpr static unsigned sign_bits = float_bits_traits<T>::sign_bits;
    constexpr static unsigned exponent_bits = float_bits_traits<T>::exponent_bits - 3;
    constexpr static unsigned mantissa_bits = float_bits_traits<T>::mantissa_bits + 3;

    using numeric_type = T;
    using bits_type = typename float_bits_traits<T>::bits_type;

    static bits_type to_bits(numeric_type x) {
        bits_type bits;
        memcpy(&bits, &x, sizeof bits);
        auto exponent = (bits << 1u) & ~(~bits_type{0} >> exponent_bits);
        auto sign = (bits & ~(~bits_type{0} >> sign_bits)) >> exponent_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        return exponent | sign | mantissa;
    }

    static numeric_type from_bits(bits_type bits) {
        auto sign = (bits << exponent_bits) & ~(~bits_type{0} >> sign_bits);
        auto exponent = (bits & ~(~bits_type{0} >> exponent_bits)) >> sign_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        bits = sign | exponent | mantissa;
        numeric_type x;
        memcpy(&x, &bits, sizeof bits);
        return x;
    }
};


template<typename T, unsigned Dims>
class profile {
    public:
        using data_type = T;
        using bits_type = detail::bits_type<T>;
        using hypercube_offset_type = uint32_t;

        constexpr static unsigned dimensions = Dims;
        constexpr static unsigned hypercube_side_length = Dims == 1 ? 4096 : Dims == 2 ? 64 : 16;
        constexpr static size_t compressed_block_size_bound
            = ((detail::bitsof<data_type> + (sizeof(T) <= 4 ? 4: 5))
                * detail::ipow(hypercube_side_length, Dims) + CHAR_BIT-1) / CHAR_BIT;
};


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

template<typename T>
void inverse_block_transform_step(T *x, size_t n, size_t s) {
    for (size_t i = 1; i < n; ++i) {
        x[i*s] ^= x[(i-1)*s];
    }
}

template<typename T>
void block_transform(T *x, unsigned dims, size_t n) {
    if (dims == 1) {
        block_transform_step(x, n, 1);
    } else if (dims == 2) {
        for (size_t i = 0; i < n*n; i += n) {
            block_transform_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n; ++i) {
            block_transform_step(x + i, n, n);
        }
    } else if (dims == 3) {
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
void inverse_block_transform(T *x, unsigned dims, size_t n) {
    if (dims == 1) {
        inverse_block_transform_step(x, n, 1);
    } else if (dims == 2) {
        for (size_t i = 0; i < n; ++i) {
            inverse_block_transform_step(x + i, n, n);
        }
        for (size_t i = 0; i < n*n; i += n) {
            inverse_block_transform_step(x + i, n, 1);
        }
    } else if (dims == 3) {
        for (size_t i = 0; i < n*n; ++i) {
            inverse_block_transform_step(x + i, n, n * n);
        }
        for (size_t i = 0; i < n*n*n; i += n) {
            inverse_block_transform_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n*n*n; i += n*n) {
            for (size_t j = 0; j < n; ++j) {
                inverse_block_transform_step(x + i + j, n, n);
            }
        }
    }
}

template<typename Profile>
size_t run_length_encode(const typename Profile::bits_type *coeffs, void *stream) {
    using bits_type = typename Profile::bits_type;

    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);

    unsigned width_width;
    unsigned width_min;
    if (sizeof(bits_type) <= 4) {
        width_width = 4;
        width_min = 19;
    } else {
        width_width = 5;
        width_min = 35;
    }

    auto remainder_dest = dest;
    remainder_dest.advance(width_width * detail::ipow(Profile::hypercube_side_length, Profile::dimensions));
    for (unsigned i = 0; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width = detail::significant_bits(coeffs[i]);
        if (width == 0) {
            dest.advance(width_width);
        } else {
            auto width_code = width < width_min ? bits_type{1} : width + 2 - width_min;
            auto verbatim_bits = width < width_min ? width_min - 1 : width - 1;
            detail::store_bits_linear<bits_type>(dest, width_width, width_code);
            dest.advance(width_width);
            detail::store_bits_linear<bits_type>(remainder_dest, verbatim_bits,
                coeffs[i] & ~(~bits_type{} << (verbatim_bits)));
            remainder_dest.advance(verbatim_bits);
        }
    }

    return ceil_byte_offset(stream, remainder_dest);
}

template<typename Profile>
size_t run_length_decode(const void *stream, typename Profile::bits_type *coeffs) {
    using bits_type = typename Profile::bits_type;

    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);

    unsigned width_width;
    unsigned width_min;
    if (sizeof(bits_type) <= 4) {
        width_width = 4;
        width_min = 19;
    } else {
        width_width = 5;
        width_min = 35;
    }

    auto remainder_src = src;
    remainder_src.advance(width_width * detail::ipow(Profile::hypercube_side_length, Profile::dimensions));
    for (unsigned i = 0; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width_code = detail::load_bits<bits_type>(src, width_width);
        src.advance(width_width);
        if (width_code == 0) {
            coeffs[i] = 0;
        } else if (width_code == 1) {
            coeffs[i] = detail::load_bits<bits_type>(remainder_src, width_min - 1);
            remainder_src.advance(width_min - 1);
        } else {
            auto width = width_code + width_min - 2;
            coeffs[i] = (bits_type{1} << (width - 1)) | detail::load_bits<bits_type>(remainder_src, width - 1);
            remainder_src.advance(width - 1);
        }
    }

    return ceil_byte_offset(stream, remainder_src);
}


template<typename Profile>
size_t encode_block(typename Profile::bits_type *bits, void *stream) {
    detail::block_transform(bits, Profile::dimensions, Profile::hypercube_side_length);
    return detail::run_length_encode<Profile>(bits, stream);
}

template<typename Profile>
size_t decode_block(const void *stream, typename Profile::bits_type *bits) {
    auto end = detail::run_length_decode<Profile>(stream, bits);
    detail::inverse_block_transform(bits, Profile::dimensions, Profile::hypercube_side_length);
    return end;
}

template<typename Profile>
typename Profile::bits_type load_value(const typename Profile::data_type *data) {
    return detail::positional_bits_repr<typename Profile::data_type>::to_bits(*data);
}

template<typename Profile>
void store_value(typename Profile::data_type *data, typename Profile::bits_type bits) {
    *data = detail::positional_bits_repr<typename Profile::data_type>::from_bits(bits);
}

template<typename Profile>
void load_superblock_warp(size_t tid, const superblock<Profile> &sb,
    const slice<const typename Profile::data_type, Profile::dimensions> &src,
    typename Profile::bits_type *dest) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto warp_size = HCDE_WARP_SIZE;
    auto global_offset = sb.hypercube_at(0).global_offset();
    if constexpr (Profile::dimensions == 1) {
        auto start = linear_offset(src.size(), global_offset);
        auto width = std::min(warp_size * HCDE_SUPERBLOCK_SIZE, src.size()[0] - global_offset[0]);
        auto src_ptr = src.data() + start;
        for (size_t i = tid; i < width; i += warp_size) {
            dest[i] = load_value<Profile>(src_ptr + i);
        }
    } else if constexpr (Profile::dimensions == 2) {
        const auto num_hcs = sb.num_hypercubes();
        auto hcs_remaining = num_hcs;
        while (hcs_remaining > 0) {
            auto hcs_in_this_row = std::min(hcs_remaining,
                (src.size()[1] - global_offset[1]) / side_length);
            auto first_hc_index_in_this_row = num_hcs - hcs_remaining;
            auto line_width = hcs_in_this_row * side_length;
            auto src_pos = linear_offset(src.size(), global_offset);
            for (size_t line_in_hc = 0; line_in_hc < side_length; ++line_in_hc) {
                for (size_t j = tid; j < line_width; j += warp_size) {
                    auto hc_index = first_hc_index_in_this_row + j / side_length;
                    auto col_in_hc = j % side_length;
                    dest[hc_index * hc_size + line_in_hc * side_length + col_in_hc]
                        = load_value<Profile>(src.data() + src_pos + j);
                }
                src_pos += src.size()[1];
            }
            global_offset[1] = 0;
            global_offset[0] += Profile::hypercube_side_length;
            hcs_remaining -= hcs_in_this_row;
        }
    } else if constexpr (Profile::dimensions == 3) {
        const auto num_hcs = sb.num_hypercubes();
        auto hcs_remaining = num_hcs;
        while (hcs_remaining > 0) {
            auto hcs_in_this_row = std::min(hcs_remaining,
                (src.size()[2] - global_offset[2]) / side_length);
            auto first_hc_index_in_this_row = num_hcs - hcs_remaining;
            auto line_width = hcs_in_this_row * side_length;
            auto local_offset = global_offset;
            for (size_t plane_in_hc = 0; plane_in_hc < side_length; ++plane_in_hc) {
                auto src_pos = linear_offset(src.size(), local_offset);
                for (size_t line_in_hc = 0; line_in_hc < side_length; ++line_in_hc) {
                    for (size_t j = tid; j < line_width; j += warp_size) {
                        auto hc_index = first_hc_index_in_this_row + j / side_length;
                        auto col_in_hc = j % side_length;
                        dest[hc_index * hc_size + plane_in_hc * side_length * side_length
                            + line_in_hc * side_length + col_in_hc]
                            = load_value<Profile>(src.data() + src_pos + j);
                    }
                    src_pos += src.size()[2];
                }
                ++local_offset[0];
            }
            global_offset[2] = 0;
            global_offset[1] += Profile::hypercube_side_length;
            if (global_offset[1] + Profile::hypercube_side_length > src.size()[1]) {
                global_offset[1] = 0;
                global_offset[0] += Profile::hypercube_side_length;
            }
            hcs_remaining -= hcs_in_this_row;
        }
    } else {
        static_assert(Profile::dimensions != Profile::dimensions, "unimplemented");
    }
}

} // namespace hcde::detail
