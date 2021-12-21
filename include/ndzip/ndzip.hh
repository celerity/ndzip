#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>


#if defined(__CUDA__) || defined(__NVCC__)
#define NDZIP_UNIVERSAL __host__ __device__
#else
#define NDZIP_UNIVERSAL
#endif


namespace ndzip {

using dim_type = int;
using index_type = uint32_t;

}  // namespace ndzip

namespace ndzip::detail {

template<dim_type Dims>
class static_extent;

}  // namespace ndzip::detail

namespace ndzip {

inline constexpr dim_type max_dimensionality = 3;

class extent {
  public:
    using const_iterator = const index_type *;
    using iterator = index_type *;

    constexpr extent() noexcept = default;

    NDZIP_UNIVERSAL constexpr explicit extent(dim_type dims) noexcept : _dims{dims} {
        assert(dims > 0);
        assert(dims <= max_dimensionality);
    }

    NDZIP_UNIVERSAL static constexpr extent broadcast(dim_type dims, index_type scalar) {
        extent e(dims);
        for (dim_type d = 0; d < dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    NDZIP_UNIVERSAL constexpr extent(std::initializer_list<index_type> components) noexcept
        : _dims{static_cast<dim_type>(components.size())} {
        assert(size(components) < static_cast<size_t>(max_dimensionality));
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] = data(components)[d];
        }
    }

    NDZIP_UNIVERSAL constexpr dim_type dimensions() const { return _dims; }

    NDZIP_UNIVERSAL index_type &operator[](dim_type d) {
        assert(d < _dims);
        return _components[d];
    }

    NDZIP_UNIVERSAL index_type operator[](dim_type d) const {
        assert(d < _dims);
        return _components[d];
    }

    NDZIP_UNIVERSAL extent &operator+=(const extent &other) {
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] += other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator+(const extent &left, const extent &right) {
        auto result = left;
        result += right;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator-=(const extent &other) {
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] -= other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator-(const extent &left, const extent &right) {
        auto result = left;
        result -= right;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator*=(index_type other) {
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] *= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator*(const extent &left, index_type right) {
        auto result = left;
        result *= right;
        return result;
    }

    NDZIP_UNIVERSAL friend extent operator*(index_type left, const extent &right) {
        auto result = right;
        result *= left;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator/=(index_type other) {
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] /= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator/(const extent &left, index_type right) {
        auto result = left;
        result /= right;
        return result;
    }

    NDZIP_UNIVERSAL friend bool operator==(const extent &left, const extent &right) {
        assert(left._dims == right._dims);
        bool eq = true;
        for (dim_type d = 0; d < left._dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const extent &left, const extent &right) { return !operator==(left, right); }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + _dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + _dims; }

  private:
    template<dim_type Dims>
    friend class detail::static_extent;

    dim_type _dims = 1;
    index_type _components[max_dimensionality] = {};
};


template<typename Extent>
NDZIP_UNIVERSAL index_type num_elements(const Extent &size) {
    index_type n = 1;
    for (dim_type d = 0; d < size.dimensions(); ++d) {
        n *= size[d];
    }
    return n;
}

// TODO what is the difference between linear_offset and linear_index?
template<typename Extent>
NDZIP_UNIVERSAL index_type linear_offset(const Extent &position, const Extent &space) {
    assert(position.dimensions() == space.dimensions());
    index_type offset = 0;
    index_type stride = 1;
    for (dim_type nd = 0; nd < position.dimensions(); ++nd) {
        auto d = position.dimensions() - 1 - nd;
        offset += stride * position[d];
        stride *= space[d];
    }
    return offset;
}

class compressor_requirements;

}  // namespace ndzip

namespace ndzip::detail {

template<typename Extent>
NDZIP_UNIVERSAL index_type linear_index(const Extent &size, const Extent &pos) {
    assert(size.dimensions() == pos.dimensions());
    index_type l = pos[0];
    for (dim_type d = 1; d < size.dimensions(); ++d) {
        l = l * size[d] + pos[d];
    }
    return l;
}

template<size_t Size>
struct bits_type_s;

template<>
struct bits_type_s<1> {
    using type = uint8_t;
};

template<>
struct bits_type_s<2> {
    using type = uint16_t;
};

template<>
struct bits_type_s<4> {
    using type = uint32_t;
};

template<>
struct bits_type_s<8> {
    using type = uint64_t;
};

template<typename T>
using bits_type = typename bits_type_s<sizeof(T)>::type;

dim_type get_dimensionality(const compressor_requirements &req);
index_type get_num_hypercubes(const compressor_requirements &req);

}  // namespace ndzip::detail

namespace ndzip {

template<typename T>
using compressed_type = ndzip::detail::bits_type<T>;

template<typename T>
index_type compressed_length_bound(const extent &e);

template<typename T>
class compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~compressor() = default;

    virtual index_type compress(const value_type *data, const extent &data_size, compressed_type *stream) = 0;
};

template<typename T>
class decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~decompressor() = default;

    virtual index_type decompress(const compressed_type *stream, value_type *data, const extent &data_size) = 0;
};

template<typename T>
std::unique_ptr<compressor<T>> make_compressor(dim_type dims, unsigned num_threads = 0);

template<typename T>
std::unique_ptr<decompressor<T>> make_decompressor(dim_type dims, unsigned num_threads = 0);

class compressor_requirements {
  public:
    compressor_requirements() = default;
    compressor_requirements(const ndzip::extent &single_data_size);  // NOLINT(google-explicit-constructor)
    compressor_requirements(std::initializer_list<extent> data_sizes);

    void include(const extent &data_size);

  private:
    friend dim_type detail::get_dimensionality(const compressor_requirements &);
    friend index_type detail::get_num_hypercubes(const compressor_requirements &);

    dim_type _dims = -1;
    index_type _max_num_hypercubes = 0;
};

using kernel_duration = std::chrono::duration<uint64_t, std::nano>;

}  // namespace ndzip