#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

#include <ndzip/array.hh>


#define NDZIP_BIG_ENDIAN 0
#define NDZIP_LITTLE_ENDIAN 1

#if defined(__BYTE_ORDER)
#if __BYTE_ORDER == __BIG_ENDIAN
#define NDZIP_ENDIAN NDZIP_BIG_ENDIAN
#else
#define NDZIP_ENDIAN NDZIP_LITTLE_ENDIAN
#endif
#elif defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) \
        || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
#define NDZIP_ENDIAN NDZIP_BIG_ENDIAN
#elif defined(__LITTLE_ENDIAN__) || defined(__ARMEL__) || defined(__THUMBEL__) \
        || defined(__AARCH64EL__) || defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__)
#define NDZIP_ENDIAN NDZIP_LITTLE_ENDIAN
#else
#error "Unknown endianess"
#endif


namespace ndzip::detail {

template<typename Integer>
NDZIP_UNIVERSAL constexpr inline Integer ipow(Integer base, unsigned exponent) {
    Integer power{1};
    while (exponent) {
        if (exponent & 1u) { power *= base; }
        base *= base;
        exponent >>= 1u;
    }
    return power;
}

template<typename T>
using bits_type = std::conditional_t<sizeof(T) == 1, uint8_t,
        std::conditional_t<sizeof(T) == 2, uint16_t,
                std::conditional_t<sizeof(T) == 4, uint32_t,
                        std::conditional_t<sizeof(T) == 8, uint64_t, void>>>>;

template<typename Integer>
NDZIP_UNIVERSAL constexpr Integer div_ceil(Integer p, Integer q) {
    return (p + q - 1) / q;
}

template<typename Integer>
NDZIP_UNIVERSAL constexpr Integer ceil(Integer x, Integer multiple) {
    return div_ceil(x, multiple) * multiple;
}

template<typename Integer>
NDZIP_UNIVERSAL constexpr Integer floor(Integer x, Integer multiple) {
    return x / multiple * multiple;
}

template<typename T>
constexpr inline index_type bits_of = static_cast<index_type>(CHAR_BIT * sizeof(T));

template<typename T>
constexpr inline index_type bytes_of = static_cast<index_type>(sizeof(T));

template<typename Integer>
NDZIP_UNIVERSAL constexpr Integer floor_power_of_two(Integer x) {
    for (index_type s = 1; s < bits_of<Integer>; ++s) {
        x &= ~(x >> s);
    }
    return (x + 1) / 2;
}

template<typename Fn, typename Index, typename T>
[[gnu::always_inline]] void invoke_for_element(Fn &&fn, Index index, T &&value) {
    if constexpr (std::is_invocable_v<Fn, T, Index>) {
        fn(std::forward<T>(value), index);
    } else {
        fn(std::forward<T>(value));
    }
}

template<typename Integer>
Integer endian_transform(Integer value) {
    // TODO endian correction is inactive until we figure out
    //  - how compressed chunks must be transformed
    //  - how to unit-test big-endian on a little-endian machine
    if constexpr (false && NDZIP_ENDIAN == NDZIP_LITTLE_ENDIAN) {
        if constexpr (std::is_same_v<Integer, uint64_t>) {
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


template<typename POD>
[[gnu::always_inline]] NDZIP_UNIVERSAL POD load_unaligned(const void *src) {
    static_assert(std::is_trivially_copyable_v<POD>);
    POD a;
    memcpy(&a, src, sizeof(POD));
    return a;
}

template<size_t Align, typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL POD load_aligned(const Memory *src) {
    assert(reinterpret_cast<uintptr_t>(src) % Align == 0);
    return load_unaligned<POD>(__builtin_assume_aligned(src, Align));
}

template<typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL POD load_aligned(const Memory *src) {
    return load_aligned<alignof(POD), POD>(src);
}

template<typename POD>
[[gnu::always_inline]] NDZIP_UNIVERSAL void store_unaligned(void *dest, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    memcpy(dest, &a, sizeof(POD));
}

template<size_t Align, typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL void store_aligned(Memory *dest, POD a) {
    assert(reinterpret_cast<uintptr_t>(dest) % Align == 0);
    store_unaligned(__builtin_assume_aligned(dest, Align), a);
}

template<typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL void store_aligned(Memory *dest, POD a) {
    store_aligned<alignof(POD), POD>(dest, a);
}

template<unsigned Dims, typename Fn>
void for_each_border_slice_recursive(const extent<Dims> &size, extent<Dims> pos,
        unsigned side_length, unsigned d, unsigned smallest_dim_with_border, const Fn &fn) {
    auto border_begin = size[d] / side_length * side_length;
    auto border_end = size[d];

    if (d < smallest_dim_with_border) {
        for (pos[d] = 0; pos[d] < border_begin; ++pos[d]) {
            for_each_border_slice_recursive(
                    size, pos, side_length, d + 1, smallest_dim_with_border, fn);
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
        if (size[d] % side_length != 0) { smallest_dim_with_border = static_cast<int>(d); }
    }
    if (smallest_dim_with_border) {
        for_each_border_slice_recursive(
                size, extent<Dims>{}, side_length, 0, *smallest_dim_with_border, fn);
    }
}

template<typename DataType, unsigned Dims>
[[gnu::noinline]] size_t
pack_border(void *dest, const slice<DataType, Dims> &src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t dest_offset = 0;
    for_each_border_slice(src.size(), side_length, [&](index_type src_offset, index_type count) {
        memcpy(static_cast<char *>(dest) + dest_offset, src.data() + src_offset,
                count * sizeof(DataType));
        dest_offset += count * sizeof(DataType);
    });
    return dest_offset;
}

template<typename DataType, unsigned Dims>
[[gnu::noinline]] size_t
unpack_border(const slice<DataType, Dims> &dest, const void *src, unsigned side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    size_t src_offset = 0;
    for_each_border_slice(dest.size(), side_length, [&](index_type dest_offset, index_type count) {
        memcpy(dest.data() + dest_offset, static_cast<const char *>(src) + src_offset,
                count * sizeof(DataType));
        src_offset += count * sizeof(DataType);
    });
    return src_offset;
}

template<unsigned Dims>
index_type border_element_count(const extent<Dims> &e, unsigned side_length) {
    index_type n_cube_elems = 1;
    index_type n_all_elems = 1;
    for (unsigned d = 0; d < Dims; ++d) {
        n_cube_elems *= e[d] / side_length * side_length;
        n_all_elems *= e[d];
    }
    return n_all_elems - n_cube_elems;
}

template<typename Profile, unsigned ThisDim, typename F>
[[gnu::always_inline]] void iter_hypercubes(const extent<Profile::dimensions> &size,
        extent<Profile::dimensions> &off, index_type &i, F &f) {
    if constexpr (ThisDim == Profile::dimensions) {
        invoke_for_element(f, i, off);
        ++i;
    } else {
        for (off[ThisDim] = 0; off[ThisDim] + Profile::hypercube_side_length <= size[ThisDim];
                off[ThisDim] += Profile::hypercube_side_length) {
            iter_hypercubes<Profile, ThisDim + 1>(size, off, i, f);
        }
    }
}

template<typename Profile>
struct stream {
    using bits_type = std::conditional_t<std::is_const_v<Profile>,
            const typename Profile::bits_type, typename Profile::bits_type>;
    using offset_type = std::conditional_t<std::is_const_v<Profile>, const index_type, index_type>;
    using byte_type = std::conditional_t<std::is_const_v<Profile>, const std::byte, std::byte>;

    static_assert(
            sizeof(bits_type) >= sizeof(offset_type) && alignof(bits_type) >= alignof(offset_type));

    index_type num_hypercubes;
    bits_type *buffer;

    NDZIP_UNIVERSAL offset_type *header() { return reinterpret_cast<offset_type *>(buffer); }

    NDZIP_UNIVERSAL index_type offset_after(index_type hc_index) { return header()[hc_index]; }

    NDZIP_UNIVERSAL void set_offset_after(index_type hc_index, index_type position) {
        // TODO memcpy this, else potential aliasing UB!
        header()[hc_index] = position;
    }

    // requires header() to be initialized
    NDZIP_UNIVERSAL bits_type *hypercube(index_type hc_index) {
        auto base_byte_offset = ceil(num_hypercubes * bytes_of<offset_type>, bytes_of<bits_type>);
        auto *base = reinterpret_cast<bits_type *>(
                reinterpret_cast<byte_type *>(buffer) + base_byte_offset);
        if (hc_index == 0) {
            return base;
        } else {
            return base + offset_after(hc_index - 1);
        }
    }

    NDZIP_UNIVERSAL index_type hypercube_size(index_type hc_index) {
        return hc_index == 0 ? offset_after(0)
                             : offset_after(hc_index) - offset_after(hc_index - 1);
    }

    // requires header() to be initialized
    NDZIP_UNIVERSAL bits_type *border() { return hypercube(num_hypercubes); }
};

template<typename Profile>
class file {
  public:
    explicit file(extent<Profile::dimensions> size) : _size(size) {}

    index_type num_hypercubes() const {
        index_type num = 1;
        for (unsigned d = 0; d < Profile::dimensions; ++d) {
            num *= _size[d] / Profile::hypercube_side_length;
        }
        return num;
    }

    template<typename Fn>
    void for_each_hypercube(Fn &&f) const {
        index_type i = 0;
        extent<Profile::dimensions> off{};
        iter_hypercubes<Profile, 0>(_size, off, i, f);
    }

    constexpr size_t file_header_length() const { return num_hypercubes() * sizeof(index_type); }

    const extent<Profile::dimensions> &size() const { return _size; };

  private:
    extent<Profile::dimensions> _size;
};

template<typename T, unsigned Dims>
class profile {
  public:
    using data_type = T;
    using bits_type = detail::bits_type<T>;

    constexpr static unsigned dimensions = Dims;
    constexpr static unsigned hypercube_side_length = Dims == 1 ? 4096 : Dims == 2 ? 64 : 16;
    constexpr static size_t compressed_block_size_bound = detail::ipow(hypercube_side_length, Dims)
            / bits_of<bits_type> * (bits_of<bits_type> + 1) * sizeof(bits_type);
};


template<typename T>
NDZIP_UNIVERSAL T rotate_left_1(T v) {
    return (v << 1u) | (v >> (bits_of<T> - 1u));
}

template<typename T>
NDZIP_UNIVERSAL T rotate_right_1(T v) {
    return (v >> 1u) | (v << (bits_of<T> - 1u));
}

template<typename T>
NDZIP_UNIVERSAL T complement_negative(T v) {
    return v >> (bits_of<T> - 1u) ? v ^ (~T{} >> 1u) : v;
}

template<typename T>
inline void block_transform_step(T *x, index_type n, index_type s) {
    T a, b;
    b = x[0 * s];
    for (index_type i = 1; i < n; ++i) {
        a = b;
        b = x[i * s];
        x[i * s] = b - a;
    }
}

template<typename T>
inline void inverse_block_transform_step(T *x, index_type n, index_type s) {
    for (index_type i = 1; i < n; ++i) {
        x[i * s] += x[(i - 1) * s];
    }
}

template<typename T>
inline void block_transform(T *x, unsigned dims, index_type n) {
    for (index_type i = 0; i < ipow(n, dims); ++i) {
        x[i] = rotate_left_1(x[i]);
    }

    if (dims == 1) {
        block_transform_step(x, n, 1);
    } else if (dims == 2) {
        for (index_type i = 0; i < n * n; i += n) {
            block_transform_step(x + i, n, 1);
        }
        for (index_type i = 0; i < n; ++i) {
            block_transform_step(x + i, n, n);
        }
    } else if (dims == 3) {
        for (index_type i = 0; i < n * n * n; i += n * n) {
            for (index_type j = 0; j < n; ++j) {
                block_transform_step(x + i + j, n, n);
            }
        }
        for (index_type i = 0; i < n * n * n; i += n) {
            block_transform_step(x + i, n, 1);
        }
        for (index_type i = 0; i < n * n; ++i) {
            block_transform_step(x + i, n, n * n);
        }
    }

    for (index_type i = 0; i < ipow(n, dims); ++i) {
        x[i] = complement_negative(x[i]);
    }
}

template<typename T>
inline void inverse_block_transform(T *x, unsigned dims, index_type n) {
    for (index_type i = 0; i < ipow(n, dims); ++i) {
        x[i] = complement_negative(x[i]);
    }

    if (dims == 1) {
        inverse_block_transform_step(x, n, 1);
    } else if (dims == 2) {
        for (index_type i = 0; i < n; ++i) {
            inverse_block_transform_step(x + i, n, n);
        }
        for (index_type i = 0; i < n * n; i += n) {
            inverse_block_transform_step(x + i, n, 1);
        }
    } else if (dims == 3) {
        for (index_type i = 0; i < n * n; ++i) {
            inverse_block_transform_step(x + i, n, n * n);
        }
        for (index_type i = 0; i < n * n * n; i += n) {
            inverse_block_transform_step(x + i, n, 1);
        }
        for (index_type i = 0; i < n * n * n; i += n * n) {
            for (index_type j = 0; j < n; ++j) {
                inverse_block_transform_step(x + i + j, n, n);
            }
        }
    }

    for (index_type i = 0; i < ipow(n, dims); ++i) {
        x[i] = rotate_right_1(x[i]);
    }
}


template<typename Profile, typename SliceDataType, typename CubeDataType, typename F>
[[gnu::always_inline]] void for_each_hypercube_slice(const extent<Profile::dimensions> &hc_offset,
        const slice<SliceDataType, Profile::dimensions> &data, CubeDataType *cube_ptr, F &&f) {
    constexpr auto side_length = Profile::hypercube_side_length;

    auto slice_ptr = &data[hc_offset];
    if constexpr (Profile::dimensions == 1) {
        f(slice_ptr, cube_ptr, side_length);
    } else if constexpr (Profile::dimensions == 2) {
        const auto stride = data.size()[1];
        for (index_type i = 0; i < side_length; ++i) {
            f(slice_ptr, cube_ptr, side_length);
            slice_ptr += stride;
            cube_ptr += side_length;
        }
    } else if constexpr (Profile::dimensions == 3) {
        const auto stride0 = data.size()[1] * data.size()[2];
        const auto stride1 = data.size()[2];
        for (index_type i = 0; i < side_length; ++i) {
            auto slice_ptr1 = slice_ptr;
            for (index_type j = 0; j < side_length; ++j) {
                f(slice_ptr1, cube_ptr, side_length);
                slice_ptr1 += stride1;
                cube_ptr += side_length;
            }
            slice_ptr += stride0;
        }
    } else {
        static_assert(Profile::dimensions != Profile::dimensions, "unimplemented");
    }
}

template<unsigned Dims>
NDZIP_UNIVERSAL extent<Dims> extent_from_linear_id(index_type linear_id, const extent<Dims> &size) {
    extent<Dims> ext;
    for (unsigned nd = 0; nd < Dims; ++nd) {
        auto d = Dims - 1 - nd;
        ext[d] = linear_id % size[d];
        linear_id /= size[d];
    }
    return ext;
}


// static_assert as an expression for use in variable template definitions
template<bool Assertion>
NDZIP_UNIVERSAL constexpr void static_check() {
    static_assert(Assertion);
}


NDZIP_UNIVERSAL inline unsigned popcount(unsigned int x) {
    return __builtin_popcount(x);
}

NDZIP_UNIVERSAL inline unsigned popcount(unsigned long x) {
    return __builtin_popcountl(x);
}

NDZIP_UNIVERSAL inline unsigned popcount(unsigned long long x) {
    return __builtin_popcountll(x);
}


template<typename U, typename T>
[[gnu::always_inline]] NDZIP_UNIVERSAL U bit_cast(T v) {
    static_assert(std::is_trivially_copy_constructible_v<U> && sizeof(U) == sizeof(T));
    U cast;
    __builtin_memcpy(&cast, &v, sizeof cast);
    return cast;
}

inline bool verbose()
{
    auto env = getenv("NDZIP_VERBOSE");
    return env && *env;
}

}  // namespace ndzip::detail
