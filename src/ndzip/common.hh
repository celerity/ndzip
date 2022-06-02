#pragma once

#include "cuda_workaround.hh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

#include <ndzip/ndzip.hh>


namespace ndzip::detail {

template<dim_type Dims>
class static_extent {
  public:
    static_assert(Dims > 0);
    static_assert(Dims <= max_dimensionality);

    using const_iterator = const index_type *;
    using iterator = index_type *;

    constexpr static_extent() noexcept = default;

    template<typename... Init,
            std::enable_if_t<((sizeof...(Init) == Dims) && ... && std::is_convertible_v<Init, index_type>), int> = 0>
    NDZIP_UNIVERSAL constexpr static_extent(Init... components) noexcept
        : _components{static_cast<index_type>(components)...} {}

    NDZIP_UNIVERSAL explicit static_extent(const extent &dyn) {
        assert(dyn._dims == Dims);
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] = dyn._components[d];
        }
    }

    NDZIP_UNIVERSAL static static_extent broadcast(index_type scalar) {
        static_extent e;
        for (dim_type d = 0; d < Dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    NDZIP_UNIVERSAL constexpr dim_type dimensions() const { return Dims; }

    NDZIP_UNIVERSAL index_type &operator[](dim_type d) { return _components[d]; }

    NDZIP_UNIVERSAL index_type operator[](dim_type d) const { return _components[d]; }

    NDZIP_UNIVERSAL static_extent &operator+=(const static_extent &other) {
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] += other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend static_extent operator+(const static_extent &left, const static_extent &right) {
        auto result = left;
        result += right;
        return result;
    }

    NDZIP_UNIVERSAL static_extent &operator-=(const static_extent &other) {
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] -= other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend static_extent operator-(const static_extent &left, const static_extent &right) {
        auto result = left;
        result -= right;
        return result;
    }

    NDZIP_UNIVERSAL static_extent &operator*=(index_type other) {
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] *= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend static_extent operator*(const static_extent &left, index_type right) {
        auto result = left;
        result *= right;
        return result;
    }

    NDZIP_UNIVERSAL friend static_extent operator*(index_type left, const static_extent &right) {
        auto result = right;
        result *= left;
        return result;
    }

    NDZIP_UNIVERSAL static_extent &operator/=(index_type other) {
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] /= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend static_extent operator/(const static_extent &left, index_type right) {
        auto result = left;
        result /= right;
        return result;
    }

    NDZIP_UNIVERSAL friend bool operator==(const static_extent &left, const static_extent &right) {
        bool eq = true;
        for (dim_type d = 0; d < Dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const static_extent &left, const static_extent &right) {
        return !operator==(left, right);
    }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + Dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + Dims; }

    NDZIP_UNIVERSAL constexpr operator extent() const {  // NOLINT(google-explicit-constructor)
        extent dyn(Dims);
        for (dim_type d = 0; d < Dims; ++d) {
            dyn[d] = _components[d];
        }
        return dyn;
    }

  private:
    friend class ndzip::extent;
    index_type _components[Dims] = {};
};

template<typename... Init>
static_extent(const Init &...) -> static_extent<sizeof...(Init)>;

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
        x |= x >> s;
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


template<typename POD>
[[gnu::always_inline]] NDZIP_UNIVERSAL POD load_unaligned(const void *src) {
    static_assert(std::is_trivially_copyable_v<POD>);
    POD a;
    __builtin_memcpy(&a, src, sizeof(POD));
    return a;
}

template<typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL POD load_aligned(const Memory *src) {
    static_assert(sizeof(POD) >= sizeof(Memory) && sizeof(POD) % sizeof(Memory) == 0);

    // GCC explicitly allows type punning through unions
    union pun {
        Memory mem[sizeof(POD) / sizeof(Memory)];
        POD value;
        NDZIP_UNIVERSAL pun() : mem{{}} {}
    } pun;
    for (size_t i = 0; i < sizeof(POD) / sizeof(Memory); ++i) {
        pun.mem[i] = src[i];
    }
    return pun.value;
}

template<typename POD>
[[gnu::always_inline]] NDZIP_UNIVERSAL void store_unaligned(void *dest, POD a) {
    static_assert(std::is_trivially_copyable_v<POD>);
    __builtin_memcpy(dest, &a, sizeof(POD));
}

template<typename POD, typename Memory>
[[gnu::always_inline]] NDZIP_UNIVERSAL void store_aligned(Memory *dest, POD a) {
    static_assert(sizeof(POD) >= sizeof(Memory) && sizeof(POD) % sizeof(Memory) == 0);

    // GCC explicitly allows type punning through unions
    union pun {
        POD value{};
        Memory mem[sizeof(POD) / sizeof(Memory)];
        NDZIP_UNIVERSAL explicit pun(POD v) : value(v) {}
    } pun(a);
    for (size_t i = 0; i < sizeof(POD) / sizeof(Memory); ++i) {
        dest[i] = pun.mem[i];
    }
}

template<dim_type Dims, typename Fn>
void for_each_border_slice_recursive(const static_extent<Dims> &size, static_extent<Dims> pos, index_type side_length,
        dim_type d, dim_type smallest_dim_with_border, const Fn &fn) {
    auto border_begin = size[d] / side_length * side_length;
    auto border_end = size[d];

    if (d < smallest_dim_with_border) {
        for (pos[d] = 0; pos[d] < border_begin; ++pos[d]) {
            for_each_border_slice_recursive(size, pos, side_length, d + 1, smallest_dim_with_border, fn);
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

template<dim_type Dims, typename Fn>
void for_each_border_slice(const static_extent<Dims> &size, index_type side_length, const Fn &fn) {
    std::optional<dim_type> smallest_dim_with_border;
    for (dim_type d = 0; d < Dims; ++d) {
        if (size[d] / side_length == 0) {
            // special case: the whole array is a border
            fn(0, num_elements(size));
            return;
        }
        if (size[d] % side_length != 0) { smallest_dim_with_border = static_cast<int>(d); }
    }
    if (smallest_dim_with_border) {
        for_each_border_slice_recursive(size, static_extent<Dims>{}, side_length, 0, *smallest_dim_with_border, fn);
    }
}

template<typename DataType, dim_type Dims>
[[gnu::noinline]] index_type pack_border(
        compressed_type<DataType> *dest, DataType *src, const static_extent<Dims> &src_size, index_type side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    index_type dest_offset = 0;
    for_each_border_slice(src_size, side_length, [&](index_type src_offset, index_type count) {
        memcpy(dest + dest_offset, src + src_offset, count * sizeof(DataType));
        dest_offset += count;
    });
    return dest_offset;
}

template<typename DataType, dim_type Dims>
[[gnu::noinline]] index_type unpack_border(DataType *dest, const static_extent<Dims> &dest_size,
        const compressed_type<DataType> *src, index_type side_length) {
    static_assert(std::is_trivially_copyable_v<DataType>);
    index_type src_offset = 0;
    for_each_border_slice(dest_size, side_length, [&](index_type dest_offset, index_type count) {
        memcpy(dest + dest_offset, src + src_offset, count * sizeof(DataType));
        src_offset += count;
    });
    return src_offset;
}

template<dim_type Dims>
index_type border_element_count(const static_extent<Dims> &e, dim_type side_length) {
    index_type n_cube_elems = 1;
    index_type n_all_elems = 1;
    for (dim_type d = 0; d < Dims; ++d) {
        n_cube_elems *= e[d] / side_length * side_length;
        n_all_elems *= e[d];
    }
    return n_all_elems - n_cube_elems;
}

inline dim_type get_dimensionality(const compressor_requirements &req) {
    if (req._dims == -1) { throw std::runtime_error{"Cannot construct a compressor with empty requirements"}; }
    return req._dims;
}

inline index_type get_num_hypercubes(const compressor_requirements &req) {
    return req._max_num_hypercubes;
}

template<typename Profile>
struct stream {
    using bits_type = std::conditional_t<std::is_const_v<Profile>, const typename Profile::bits_type,
            typename Profile::bits_type>;
    using offset_type = std::conditional_t<std::is_const_v<Profile>, const index_type, index_type>;
    using byte_type = std::conditional_t<std::is_const_v<Profile>, const std::byte, std::byte>;

    static_assert(sizeof(bits_type) >= sizeof(offset_type) && alignof(bits_type) >= alignof(offset_type));

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
        auto *base = reinterpret_cast<bits_type *>(reinterpret_cast<byte_type *>(buffer) + base_byte_offset);
        if (hc_index == 0) {
            return base;
        } else {
            return base + offset_after(hc_index - 1);
        }
    }

    NDZIP_UNIVERSAL index_type hypercube_size(index_type hc_index) {
        return hc_index == 0 ? offset_after(0) : offset_after(hc_index) - offset_after(hc_index - 1);
    }

    // requires header() to be initialized
    NDZIP_UNIVERSAL bits_type *border() { return hypercube(num_hypercubes); }
};

template<dim_type Dims>
struct hypercube_side_length_s;

template<>
struct hypercube_side_length_s<1> : public std::integral_constant<index_type, 4096> {};

template<>
struct hypercube_side_length_s<2> : public std::integral_constant<index_type, 64> {};

template<>
struct hypercube_side_length_s<3> : public std::integral_constant<index_type, 16> {};

template<dim_type Dims>
constexpr static index_type hypercube_side_length = hypercube_side_length_s<Dims>::value;

template<typename T, dim_type Dims>
class profile {
  public:
    using value_type = T;
    using bits_type = detail::bits_type<T>;

    constexpr static dim_type dimensions = Dims;
    constexpr static index_type hypercube_side_length = detail::hypercube_side_length<dimensions>;
    constexpr static size_t compressed_block_length_bound
            = detail::ipow(hypercube_side_length, Dims) / bits_of<bits_type> * (bits_of<bits_type> + 1);
};

template<dim_type Dims>
index_type num_hypercubes(const static_extent<Dims> &array_size) {
    index_type num = 1;
    for (dim_type d = 0; d < Dims; ++d) {
        num *= array_size[d] / hypercube_side_length<Dims>;
    }
    return num;
}

inline index_type num_hypercubes(const extent &size) {
    switch (size.dimensions()) {
        // This is actually independent of data type
        case 1: return num_hypercubes(static_extent<1>{size});
        case 2: return num_hypercubes(static_extent<2>{size});
        case 3: return num_hypercubes(static_extent<3>{size});
        default: abort();
    }
}

template<dim_type Dims, dim_type ThisDim, typename F>
[[gnu::always_inline]] void
iter_hypercubes(const static_extent<Dims> &size, static_extent<Dims> &off, index_type &i, F &f) {
    if constexpr (ThisDim == Dims) {
        invoke_for_element(f, i, off);
        ++i;
    } else {
        for (off[ThisDim] = 0; off[ThisDim] + hypercube_side_length<Dims> <= size[ThisDim];
                off[ThisDim] += hypercube_side_length<Dims>) {
            iter_hypercubes<Dims, ThisDim + 1>(size, off, i, f);
        }
    }
}

template<dim_type Dims, typename Fn>
void for_each_hypercube(const static_extent<Dims> &array_size, Fn &&f) {
    index_type i = 0;
    static_extent<Dims> off{};
    iter_hypercubes<Dims, 0>(array_size, off, i, f);
}


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
inline void block_transform(T *x, dim_type dims, index_type n) {
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
inline void inverse_block_transform(T *x, dim_type dims, index_type n) {
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
[[gnu::always_inline]] void for_each_hypercube_slice(const static_extent<Profile::dimensions> &hc_offset,
        SliceDataType *data, const static_extent<Profile::dimensions> &data_size, CubeDataType *cube_ptr, F &&f) {
    constexpr auto side_length = Profile::hypercube_side_length;

    auto slice_ptr = &data[linear_index(data_size, hc_offset)];
    if constexpr (Profile::dimensions == 1) {
        f(slice_ptr, cube_ptr, side_length);
    } else if constexpr (Profile::dimensions == 2) {
        const auto stride = data_size[1];
        for (index_type i = 0; i < side_length; ++i) {
            f(slice_ptr, cube_ptr, side_length);
            slice_ptr += stride;
            cube_ptr += side_length;
        }
    } else if constexpr (Profile::dimensions == 3) {
        const auto stride0 = data_size[1] * data_size[2];
        const auto stride1 = data_size[2];
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

template<dim_type Dims>
NDZIP_UNIVERSAL static_extent<Dims> extent_from_linear_id(index_type linear_id, const static_extent<Dims> &size) {
    static_extent<Dims> ext;
    for (dim_type nd = 0; nd < Dims; ++nd) {
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
#ifdef __CUDA_ARCH__
    // NVCC regards __builtin_popcount as a __host__ function
    return __popc(static_cast<int>(x));
#else
    return __builtin_popcount(x);
#endif
}

NDZIP_UNIVERSAL inline unsigned popcount(unsigned long x) {
#ifdef __CUDA_ARCH__
    // NVCC regards __builtin_popcountl as a __host__ function
    if (sizeof(unsigned long) == sizeof(unsigned int)) {
        return __popc(static_cast<int>(x));
    } else {
        static_assert(sizeof(unsigned long) == sizeof(unsigned long long));
        return __popcll(static_cast<long long>(x));
    }
#else
    return __builtin_popcountl(x);
#endif
}

NDZIP_UNIVERSAL inline unsigned popcount(unsigned long long x) {
#ifdef __CUDA_ARCH__
    // NVCC regards __builtin_popcountll as a __host__ function
    return __popcll(static_cast<long long>(x));
#else
    return __builtin_popcountll(x);
#endif
}


template<typename U, typename T>
[[gnu::always_inline]] NDZIP_UNIVERSAL U bit_cast(T v) {
    static_assert(std::is_trivially_copy_constructible_v<U> && sizeof(U) == sizeof(T));
    U cast;
    __builtin_memcpy(&cast, &v, sizeof cast);
    return cast;
}

inline bool verbose() {
    auto env = getenv("NDZIP_VERBOSE");
    return env && *env;
}


template<template<typename> typename Base, template<typename> typename Impl, typename T, typename... Params>
std::unique_ptr<Base<T>> make_with_profile(dim_type dims, Params &&...args) {
    switch (dims) {
        case 1: return std::make_unique<Impl<profile<T, 1>>>(std::forward<Params>(args)...);
        case 2: return std::make_unique<Impl<profile<T, 2>>>(std::forward<Params>(args)...);
        case 3: return std::make_unique<Impl<profile<T, 3>>>(std::forward<Params>(args)...);
        default: throw std::runtime_error{"Invalid dimensionality"};
    }
}

}  // namespace ndzip::detail
