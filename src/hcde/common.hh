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


namespace hcde::detail {

#ifdef __SIZEOF_INT128__
using native_uint128_t = unsigned __int128;
#   define HCDE_HAVE_NATIVE_UINT128_T 1
#else
#   define HCDE_HAVE_NATIVE_UINT128_T 0
#endif


template<typename T, unsigned Dims, typename Fn>
void for_each_in_hypercube(const slice<T, Dims> &data, const extent<Dims> &offset,
    unsigned side_length, const Fn &fn) {
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
        auto stride1 = data.size()[2] * data.size()[3];
        auto stride2 = data.size()[3];
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
void store_unaligned(void *dest, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    memcpy(dest, &a, sizeof(POD));
}

template<size_t Align, typename POD>
void store_aligned(void *dest, POD a) {
    assert(reinterpret_cast<uintptr_t>(dest) % Align == 0);
    store_unaligned(__builtin_assume_aligned(dest, Align), a);
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

template<typename Profile>
class superblock {
    public:
        superblock(extent<Profile::dimensions> start, extent<Profile::dimensions> end,
            unsigned split_dimension)
            : _start(start)
              , _end(end)
              , _split_dimension(split_dimension) {
        }

        size_t num_hypercubes() const {
            const auto side_length = Profile::hypercube_side_length;
            size_t n_cubes = (_end[_split_dimension] - _start[_split_dimension]) / side_length;
            for (unsigned d = _split_dimension + 1; d < Profile::dimensions; ++d) {
                assert(_start[d] == 0);
                n_cubes *= _end[d] / side_length;
            }
            return n_cubes;
        }

        template<typename Fn>
        void for_each_hypercube(Fn &&fn) const {
            auto dims = Profile::dimensions;
            auto side_length = Profile::hypercube_side_length;
            for (auto off = _start;
                off[_split_dimension] + side_length <= _end[_split_dimension];
                off[_split_dimension] += side_length) {
                if (dims >= 1 && _split_dimension == dims - 1) {
                    fn(off);
                } else {
                    for (off[_split_dimension + 1] = 0;
                        off[_split_dimension + 1] + side_length <= _end[_split_dimension + 1];
                        off[_split_dimension + 1] += side_length) {
                        if (dims >= 2 && _split_dimension == dims - 2) {
                            fn(off);
                        } else {
                            for (off[_split_dimension + 2] = 0;
                                off[_split_dimension + 2] + side_length
                                    <= _end[_split_dimension + 2];
                                off[_split_dimension + 2] += side_length) {
                                if (dims >= 3 && _split_dimension == dims - 3) {
                                    fn(off);
                                } else {
                                    for (off[_split_dimension + 3] = 0;
                                        off[_split_dimension + 3] + side_length
                                            <= _end[_split_dimension + 3];
                                        off[_split_dimension + 3] += side_length) {
                                        assert(dims == 4 && _split_dimension == dims - 4);
                                        fn(off);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        extent<Profile::dimensions> hypercube_offset_at(size_t hypercube_index) const {
            auto dims = Profile::dimensions;
            auto side_length = Profile::hypercube_side_length;
            auto cell_index = hypercube_index * side_length;

            auto off = _start;
            if (dims >= 1 && _split_dimension == dims - 1) {
                off[_split_dimension] += cell_index;
            } else if (dims >= 2 && _split_dimension == dims - 1) {
                auto stride = _end[_split_dimension + 1] - _start[_split_dimension + 1];
                off[_split_dimension] += cell_index / stride;
                off[_split_dimension + 1] = cell_index % stride;
            } else if (dims >= 3 && _split_dimension == dims - 2) {
                auto stride1 = (_end[_split_dimension + 2] - _start[_split_dimension + 2]);
                auto stride0 = (_end[_split_dimension + 1] - _start[_split_dimension + 1])
                    * stride1;
                off[_split_dimension] += cell_index / stride0;
                off[_split_dimension + 1] = (cell_index % stride0) / stride1;
                off[_split_dimension + 2] = (cell_index % stride0) % stride1;
            } else if (dims >= 4 && _split_dimension == dims - 3) {
                auto stride2 = (_end[_split_dimension + 2] - _start[_split_dimension + 2]);
                auto stride1 = (_end[_split_dimension + 2] - _start[_split_dimension + 2]) * stride2;
                auto stride0 = (_end[_split_dimension + 1] - _start[_split_dimension + 1]) * stride1;
                off[_split_dimension] += cell_index / stride0;
                off[_split_dimension + 1] = (cell_index % stride0) / stride1;
                off[_split_dimension + 2] = ((cell_index % stride0) % stride1) / stride2;
                off[_split_dimension + 3] = ((cell_index % stride0) % stride1) % stride2;
            } else {
                assert(false);
            }
            return off;
        }

    private:
        extent<Profile::dimensions> _start;
        extent<Profile::dimensions> _end;
        unsigned _split_dimension;
};

template<typename Profile>
class file {
    public:
        using superblock_offset_type = uint64_t;

        explicit file(extent<Profile::dimensions> size)
            : _size(size) {
        }

        size_t num_superblocks() const {
            auto split_dim = superblock_split_dimension();
            auto side = Profile::hypercube_side_length;

            size_t n_sbs = _size[split_dim] / superblock_split_granularity();
            for (unsigned d = 0; d < split_dim; ++d) {
                n_sbs *= _size[d] / side;
            }
            return n_sbs;
        }

        template<typename Fn>
        void for_each_superblock(Fn &&fn) const {
            auto d = superblock_split_dimension();
            auto step = superblock_split_granularity();
            auto side = Profile::hypercube_side_length;

            if (Profile::dimensions > 0 && d == 0) {
                for (extent<Profile::dimensions> start; start[0] + step <= _size[0]; start[0] += step) {
                    auto end = _size;
                    end[0] = start[0] + step;
                    fn(superblock < Profile > {start, end, d});
                }
            } else if (Profile::dimensions > 1 && d == 1) {
                for (extent<Profile::dimensions> start; start[0] + side <= _size[0]; start[0] += side) {
                    for (start[1] = 0; start[1] + step <= _size[1]; start[1] += step) {
                        auto end = _size;
                        end[0] = start[0] + side;
                        end[1] = start[1] + step;
                        fn(superblock < Profile > {start, end, d});
                    }
                }
            } else if (Profile::dimensions > 2 && d == 2) {
                for (extent<Profile::dimensions> start; start[0] + side <= _size[0]; start[0] += side) {
                    for (start[1] = 0; start[1] + side <= _size[1]; start[1] += side) {
                        for (start[2] = 0; start[2] + step <= _size[2]; start[2] += step) {
                            auto end = _size;
                            end[0] = start[0] + side;
                            end[1] = start[1] + side;
                            end[2] = start[2] + step;
                            fn(superblock < Profile > {start, end, d});
                        }
                    }
                }
            } else {
                assert(Profile::dimensions > 3 && d == 3);
                for (extent<Profile::dimensions> start; start[0] + side <= _size[0]; start[0] += side) {
                    for (start[1] = 0; start[1] + side <= _size[1]; start[1] += side) {
                        for (start[2] = 0; start[2] + side <= _size[2]; start[2] += side) {
                            for (start[3] = 0; start[3] + step <= _size[3]; start[3] += step) {
                                auto end = _size;
                                end[0] = start[0] + side;
                                end[1] = start[1] + side;
                                end[2] = start[2] + side;
                                end[3] = start[3] + step;
                                fn(superblock < Profile > {start, end, d});
                            }
                        }
                    }
                }
            }
        }

        constexpr size_t num_hypercubes() const {
            size_t n_cubes = 1;
            for (unsigned d = 0; d < Profile::dimensions; ++d) {
                n_cubes *= _size[d] / Profile::hypercube_side_length;
            }
            return n_cubes;
        }

        constexpr size_t max_num_hypercubes_per_superblock() const {
            return 63; // TODO
        }

        constexpr size_t file_header_length() const {
            return std::max(size_t{1}, num_superblocks()) * sizeof(superblock_offset_type);
        }

        constexpr size_t superblock_header_length() const {
            return max_num_hypercubes_per_superblock()
                * sizeof(typename Profile::hypercube_offset_type);
        }

        constexpr size_t combined_length_of_all_headers() const {
            return file_header_length() + num_superblocks() * superblock_header_length();
        }

        constexpr unsigned superblock_split_target_size() const {
            return 32; // assumed GPU warp size
        }

        constexpr unsigned superblock_split_dimension() const {
            return superblock_split().first;
        }

        constexpr unsigned superblock_split_granularity() const {
            return superblock_split().second;
        }

    private:
        extent<Profile::dimensions> _size;

        constexpr std::pair<unsigned, size_t> superblock_split() const {
            size_t n_cubes = 1;
            unsigned d = 0;
            for (unsigned nd = 0; nd < Profile::dimensions; ++nd) {
                d = Profile::dimensions - 1 - nd;
                n_cubes *= _size[d] / Profile::hypercube_side_length;
                if (n_cubes > superblock_split_target_size() / 2) {
                    auto granularity = _size[d] / Profile::hypercube_side_length
                        * Profile::hypercube_side_length;
                    return {d, granularity};
                }
            }
            return {0, _size[0] / Profile::hypercube_side_length * Profile::hypercube_side_length};
        }
};

} // namespace hcde::detail
