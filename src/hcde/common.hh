#pragma once

#include <hcde/hcde.hh>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
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

template<typename Integer>
constexpr inline Integer ipow(Integer base, unsigned exponent) {
    Integer power{1};
    while (exponent) {
        if (exponent & 1u) {
            power *= base;
        }
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

template<typename T>
constexpr inline size_t bitsof = CHAR_BIT * sizeof(T);

using file_offset_type = uint64_t;


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
    if constexpr (HCDE_ENDIAN == HCDE_LITTLE_ENDIAN) {
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

template<typename Profile, unsigned ThisDim, typename F>
[[gnu::always_inline]]
void iter_hypercubes(const extent<Profile::dimensions> &size,
        extent<Profile::dimensions> &off, size_t &i, F &f)
{
    if constexpr (ThisDim == Profile::dimensions) {
        invoke_for_element(f, i, off);
        ++i;
    } else {
        for (off[ThisDim] = 0;
                off[ThisDim] + Profile::hypercube_side_length <= size[ThisDim];
                off[ThisDim] += Profile::hypercube_side_length)
        {
            iter_hypercubes<Profile, ThisDim+1>(size, off, i, f);
        }
    }
}

template<typename Profile>
class file {
    public:
        explicit file(extent<Profile::dimensions> size)
            : _size(size)
        {
        }

        size_t num_hypercubes() const {
            size_t num = 1;
            for (unsigned d = 0; d < Profile::dimensions; ++d) {
                num *= _size[d] / Profile::hypercube_side_length;
            }
            return num;
        }

        template<typename Fn>
        void for_each_hypercube(Fn &&f) const {
            size_t i = 0;
            extent<Profile::dimensions> off{};
            iter_hypercubes<Profile, 0>(_size, off, i, f);
        }

        constexpr size_t file_header_length() const {
            return num_hypercubes() * sizeof(file_offset_type);
        }

        const extent<Profile::dimensions> &size() const {
            return _size;
        };

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
                / bitsof<bits_type> * (bitsof<bits_type> + 1) * sizeof(bits_type);
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
[[gnu::noinline]]
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
[[gnu::noinline]]
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


#define HCDE_WARP_SIZE (size_t{32})

template<typename Profile>
void load_hypercube_warp(size_t tid, const extent<Profile::dimensions> &hc_offset,
    const slice<const typename Profile::data_type, Profile::dimensions> &src,
    typename Profile::bits_type *dest) {
    using bits_type = typename Profile::bits_type;
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto warp_size = HCDE_WARP_SIZE;
    if constexpr (Profile::dimensions == 1) {
        auto start = linear_offset(src.size(), hc_offset);
        auto src_ptr = src.data() + start;
        for (size_t i = tid; i < Profile::hypercube_side_length; i += warp_size) {
            memcpy(&dest[i], src_ptr + i, sizeof(bits_type));
        }
    } else if constexpr (Profile::dimensions == 2) {
        auto start = linear_offset(src.size(), hc_offset);
        auto dest_ptr = dest;
        for (size_t i = 0; i < side_length; ++i) {
            auto src_ptr = src.data() + start;
            for (size_t j = tid; j < side_length; j += warp_size) {
                memcpy(&dest[j], src_ptr + j, sizeof(bits_type));
            }
            start += src.size()[1];
            dest_ptr += side_length;
        }
    } else if constexpr (Profile::dimensions == 3) {
        auto src_off = hc_offset;
        auto dest_ptr = dest;
        for (size_t i = 0; i < side_length; ++i) {
            auto start = linear_offset(src.size(), src_off);
            for (size_t j = 0; j < side_length; ++j) {
                auto src_ptr = src.data() + start;
                for (size_t j = tid; j < side_length; j += warp_size) {
                    memcpy(&dest[j], src_ptr + j, sizeof(bits_type));
                }
                start += src.size()[2];
                dest_ptr += side_length;
            }
            src_off[0] += 1;
        }
    } else {
        static_assert(Profile::dimensions != Profile::dimensions, "unimplemented");
    }
}


template<typename Profile, typename SliceDataType, typename CubeDataType, typename F>
[[gnu::always_inline]]
void map_hypercube_slices(const extent<Profile::dimensions> &hc_offset,
                          const slice<SliceDataType, Profile::dimensions> &data,
                          CubeDataType *cube_ptr, F &&f)
{
    constexpr auto side_length = Profile::hypercube_side_length;

    auto slice_ptr = &data[hc_offset];
    if constexpr (Profile::dimensions == 1) {
        f(slice_ptr, cube_ptr, side_length);
    } else if constexpr (Profile::dimensions == 2) {
        const auto stride = data.size()[1];
        for (size_t i = 0; i < side_length; ++i) {
            f(slice_ptr, cube_ptr, side_length);
            slice_ptr += stride;
            cube_ptr += side_length;
        }
    } else if constexpr (Profile::dimensions == 3) {
        const auto stride0 = data.size()[1] * data.size()[2];
        const auto stride1 = data.size()[2];
        for (size_t i = 0; i < side_length; ++i) {
            auto slice_ptr1 = slice_ptr;
            for (size_t j = 0; j < side_length; ++j) {
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

} // namespace hcde::detail
